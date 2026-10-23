"""DVC stage ``optimize_portfolio``: today's allocation + out-of-sample evidence.

Runs daily after ``process`` (Day 7 scheduler). Each run:

1. Builds a (date x ticker) price matrix from ``data/processed/prices``
   (dates where any ticker is missing are dropped, so all returns are aligned).
2. Optimises the latest ``window_days`` returns over the full long-only grid
   (see :mod:`finsight.portfolio.mpt`) and compares it with equal weight
   **in sample** (same window).
3. Checks the grid against a continuous optimiser (SciPy SLSQP).
4. Runs the **walk-forward** backtest over the whole history: max-Sharpe vs
   equal weight vs minimum volatility **out of sample** -- the honest test of
   whether a 30-day Sharpe optimisation adds value.
5. Writes ``reports/portfolio.json`` (DVC metrics), plot data and figures,
   logs everything to the MLflow experiment ``finsight-portfolio`` and
   upserts the allocation into MongoDB ``portfolios`` (one document per
   ``as_of`` date) for ``GET /api/portfolio``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.runs import stage_run
from finsight.portfolio.mpt import (
    EQUAL_WEIGHT,
    MAX_SHARPE,
    MIN_VOLATILITY,
    continuous_max_sharpe,
    frontier,
    optimise,
    performance,
    walk_forward,
    weight_grid,
)

log = get_logger(__name__)


def load_returns(path: Path) -> pd.DataFrame:
    """Aligned simple daily returns, one column per ticker (sorted)."""
    frame = pd.read_parquet(path, columns=["ticker", "date", "price"])
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    prices = frame.pivot(index="date", columns="ticker", values="price").sort_index()
    prices = prices[sorted(prices.columns)].dropna(how="any")
    if (prices <= 0).any().any():
        raise ValueError("non-positive prices in processed data")
    return prices.pct_change().dropna(how="any")


def run_optimisation(params: dict[str, Any], prices_path: Path) -> dict[str, Any]:
    """Everything except side effects (files, MLflow, Mongo)."""
    returns = load_returns(prices_path)
    window, step = int(params["window_days"]), float(params["weight_step"])
    rf, periods = float(params["risk_free_rate"]), int(params["trading_days_per_year"])
    fallback = params["no_positive_sharpe_fallback"]
    if len(returns) < window:
        raise ValueError(f"need {window} aligned daily returns, have {len(returns)}")
    tickers = list(returns.columns)

    started = time.perf_counter()
    grid = weight_grid(len(tickers), step)
    recent = returns.iloc[-window:]
    result = optimise(recent, grid, rf, periods, fallback)
    grid_seconds = time.perf_counter() - started

    selected, ew = result["selected"], result[EQUAL_WEIGHT]
    top = np.argsort(-result["sharpe"])[:5]
    continuous = continuous_max_sharpe(result["mu"], result["cov"], rf, [ew.weights, *grid[top]])
    backtest = walk_forward(
        returns,
        grid,
        window=window,
        rebalance=int(params["rebalance_days"]),
        rf=rf,
        periods=periods,
        fallback=fallback,
    )
    oos = {s: performance(backtest["daily"][s], rf, periods) for s in backtest["daily"].columns}

    report = {
        "as_of": str(recent.index[-1].date()),
        "window_start": str(recent.index[0].date()),
        "window_end": str(recent.index[-1].date()),
        "window_days": window,
        "weight_step": step,
        "risk_free_rate": rf,
        "grid_size": int(len(grid)),
        "grid_seconds": round(grid_seconds, 3),
        "fallback_used": result["fallback_used"],
        "allocation": selected.as_dict(tickers),
        "in_sample": {
            name: result[name].as_dict(tickers)
            for name in (MAX_SHARPE, MIN_VOLATILITY, EQUAL_WEIGHT)
        },
        "continuous_check": (
            {
                "sharpe": round(continuous["sharpe"], 6),
                "grid_sharpe_gap": round(continuous["sharpe"] - result[MAX_SHARPE].sharpe, 6),
                "weights": {
                    t: round(float(w), 4)
                    for t, w in zip(tickers, continuous["weights"], strict=True)
                },
            }
            if continuous["success"]
            else {"sharpe": None, "grid_sharpe_gap": None}
        ),
        "walk_forward": {
            "rebalance_days": int(params["rebalance_days"]),
            "rebalances": len(backtest["rebalances"]),
            "first_day": str(backtest["daily"].index[0])[:10],
            "last_day": str(backtest["daily"].index[-1])[:10],
            "performance": oos,
            "turnover": backtest["turnover"],
            "beats_equal_weight_sharpe": bool(
                oos[MAX_SHARPE]["sharpe"] > oos[EQUAL_WEIGHT]["sharpe"]
            ),
        },
    }
    return {
        "report": report,
        "result": result,
        "grid": grid,
        "backtest": backtest,
        "tickers": tickers,
    }


def compact(report: dict[str, Any]) -> dict[str, Any]:
    """The small JSON shown by ``dvc metrics show``."""
    alloc, ew = report["allocation"], report["in_sample"][EQUAL_WEIGHT]
    wf = report["walk_forward"]["performance"]
    return {
        "as_of": report["as_of"],
        "method": alloc["method"],
        "sharpe": alloc["sharpe"],
        "expected_return": alloc["expected_return"],
        "volatility": alloc["volatility"],
        "equal_weight_sharpe": ew["sharpe"],
        "grid_sharpe_gap": report["continuous_check"]["grid_sharpe_gap"],
        "oos_sharpe_max_sharpe": wf[MAX_SHARPE]["sharpe"],
        "oos_sharpe_equal_weight": wf[EQUAL_WEIGHT]["sharpe"],
        "oos_sharpe_min_volatility": wf[MIN_VOLATILITY]["sharpe"],
        "oos_beats_equal_weight": report["walk_forward"]["beats_equal_weight_sharpe"],
    }


# --------------------------------------------------------------------------- #
# Figures and plot data
# --------------------------------------------------------------------------- #
def _plt() -> Any:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def allocation_figure(report: dict[str, Any]) -> Any:
    """Pie chart of today's allocation (zero weights omitted)."""
    plt = _plt()
    alloc = report["allocation"]
    weights = {t: w for t, w in alloc["weights"].items() if w > 0}
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie(list(weights.values()), labels=list(weights), autopct="%1.0f%%", startangle=90)
    ax.set_title(
        f"{alloc['method'].replace('_', ' ')} portfolio, as of {report['as_of']}\n"
        f"Sharpe {alloc['sharpe']:.2f} | return {alloc['expected_return']:.1%} | "
        f"volatility {alloc['volatility']:.1%}",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def frontier_points(outcome: dict[str, Any], sample: int = 3000) -> pd.DataFrame:
    """Deterministic sample of the grid cloud plus the efficient upper envelope."""
    result = outcome["result"]
    rng = np.random.default_rng(0)
    idx = np.sort(
        rng.choice(len(outcome["grid"]), size=min(sample, len(outcome["grid"])), replace=False)
    )
    cloud = pd.DataFrame(
        {
            "volatility": np.round(result["vol"][idx], 6),
            "expected_return": np.round(result["ret"][idx], 6),
            "series": "grid sample",
        }
    )
    env = frontier(result["ret"], result["vol"]).assign(series="efficient frontier")
    return pd.concat([cloud, env], ignore_index=True)


def frontier_figure(outcome: dict[str, Any], points: pd.DataFrame) -> Any:
    """Grid cloud coloured by Sharpe with the selected and baseline portfolios."""
    plt = _plt()
    result, report = outcome["result"], outcome["report"]
    cloud = points[points["series"] == "grid sample"]
    env = points[points["series"] == "efficient frontier"]
    rf = report["risk_free_rate"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sc = ax.scatter(
        cloud["volatility"],
        cloud["expected_return"],
        c=(cloud["expected_return"] - rf) / cloud["volatility"],
        s=4,
        cmap="viridis",
        alpha=0.6,
    )
    fig.colorbar(sc, ax=ax, label="Sharpe ratio")
    ax.plot(
        env["volatility"], env["expected_return"], color="black", linewidth=1.2, label="frontier"
    )
    for name, marker in ((MAX_SHARPE, "*"), (MIN_VOLATILITY, "s"), (EQUAL_WEIGHT, "o")):
        a = result[name]
        ax.scatter(
            a.volatility, a.expected_return, marker=marker, s=140, edgecolor="black", label=name
        )
    ax.set_xlabel(f"annualised volatility ({report['window_days']}-day window)")
    ax.set_ylabel("annualised expected return")
    ax.set_title(f"{report['grid_size']:,} long-only portfolios (step {report['weight_step']})")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def backtest_curve(outcome: dict[str, Any]) -> pd.DataFrame:
    """Growth of $1 per strategy over the walk-forward period."""
    growth = (1 + outcome["backtest"]["daily"]).cumprod().round(6)
    growth.index = [str(d)[:10] for d in growth.index]
    return growth.rename_axis("date").reset_index()


def backtest_figure(curve: pd.DataFrame) -> Any:
    """Line chart of :func:`backtest_curve`."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    dates = pd.to_datetime(curve["date"])
    for column in (MAX_SHARPE, EQUAL_WEIGHT, MIN_VOLATILITY):
        ax.plot(dates, curve[column], label=column, linewidth=1.6 if column == MAX_SHARPE else 1.1)
    ax.axhline(1.0, color="#999", linewidth=0.8)
    ax.set_ylabel("growth of $1 (out of sample)")
    ax.set_title("Walk-forward backtest: re-optimised every rebalance on the trailing window")
    ax.legend(fontsize=8, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #
def publish(db: Any, report: dict[str, Any], run_id: str | None) -> None:
    """Upsert the allocation for ``as_of`` into ``portfolios`` (idempotent)."""
    from finsight.common.mongo import PORTFOLIOS  # noqa: PLC0415

    as_of = datetime.fromisoformat(report["as_of"]).replace(tzinfo=UTC)
    doc = {
        **{k: v for k, v in report.items() if k != "as_of"},
        "as_of": as_of,
        "mlflow_run_id": run_id,
        "published_at": datetime.now(UTC),
    }
    db[PORTFOLIOS].replace_one({"as_of": as_of}, doc, upsert=True)


def log_to_mlflow(
    params: dict[str, Any], outcome: dict[str, Any], root: Path, figures: dict
) -> str:
    """One MLflow run per optimisation: parameters, metrics, weights, figures."""
    import mlflow  # noqa: PLC0415

    from finsight.models.tracking import lineage_tags  # noqa: PLC0415

    report = outcome["report"]
    with mlflow.start_run(run_name=f"portfolio-{report['as_of']}") as run:
        mlflow.set_tags(
            {
                **lineage_tags(root, params["prices_path"], 0),
                "stage": "optimize_portfolio",
                "as_of": report["as_of"],
                "method": report["allocation"]["method"],
                "oos_beats_equal_weight": str(
                    report["walk_forward"]["beats_equal_weight_sharpe"]
                ).lower(),
            }
        )
        mlflow.log_params(
            {
                "window_days": report["window_days"],
                "weight_step": report["weight_step"],
                "risk_free_rate": report["risk_free_rate"],
                "grid_size": report["grid_size"],
                "rebalance_days": report["walk_forward"]["rebalance_days"],
                "fallback": params["no_positive_sharpe_fallback"],
                "tickers": ",".join(outcome["tickers"]),
            }
        )
        flat = {
            k: v
            for k, v in compact(report).items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
        for strategy, perf in report["walk_forward"]["performance"].items():
            flat.update({f"oos_{strategy}_{k}": v for k, v in perf.items()})
        flat.update({f"weight_{t}": w for t, w in report["allocation"]["weights"].items()})
        mlflow.log_metrics({k: float(v) for k, v in flat.items() if v is not None})
        mlflow.log_dict(report, "portfolio/report.json")
        mlflow.log_dict(
            {"rebalances": outcome["backtest"]["rebalances"]}, "portfolio/rebalances.json"
        )
        for name, fig in figures.items():
            mlflow.log_figure(fig, f"plots/{name}.png")
        return run.info.run_id


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.portfolio.optimize``)."""
    parser = argparse.ArgumentParser(description="Optimise the portfolio (MPT grid search)")
    parser.add_argument("--no-publish", action="store_true", help="skip the MongoDB upsert")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("optimize_portfolio", settings.log_level, settings.root / "logs")
    params = dict(settings.section("portfolio"))
    params["prices_path"] = settings.section("etl")["processed_prices_path"]
    root = settings.root
    try:
        with stage_run("optimize_portfolio", settings) as run:
            outcome = run_optimisation(params, root / params["prices_path"])
            report = outcome["report"]
            plots = root / "reports" / "plots"
            plots.mkdir(parents=True, exist_ok=True)
            points = frontier_points(outcome)
            points.to_csv(plots / "portfolio_frontier.csv", index=False)
            curve = backtest_curve(outcome)
            curve.to_csv(plots / "portfolio_backtest.csv", index=False)
            figures = {
                "portfolio_allocation": allocation_figure(report),
                "portfolio_frontier": frontier_figure(outcome, points),
                "portfolio_backtest": backtest_figure(curve),
            }
            for name, fig in figures.items():
                fig.savefig(plots / f"{name}.png", dpi=110)
            run_id = None
            if not args.no_mlflow:
                from finsight.models.tracking import configure_tracking  # noqa: PLC0415

                configure_tracking(params["experiment_name"])
                run_id = log_to_mlflow(params, outcome, root, figures)
            if not args.no_publish:
                from finsight.common.mongo import ensure_indexes, get_client  # noqa: PLC0415

                client = get_client(settings)
                try:
                    db = client[settings.mongo_db]
                    ensure_indexes(db)
                    publish(db, report, run_id)
                finally:
                    client.close()
            (root / params["backtest_report"]).write_text(
                json.dumps({**report, "rebalances": outcome["backtest"]["rebalances"]}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            metrics = {**compact(report), "run_id": run_id}
            (root / params["report"]).write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            run.metrics = compact(report)
    except (FinSightError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"optimize_portfolio failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
