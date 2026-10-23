"""DVC stage ``evaluate_lstm``: score the packaged model on untouched test windows.

The model is loaded from ``models/lstm/model`` with ``mlflow.pyfunc.load_model``
-- exactly how a serving process loads it -- so this stage also proves the
deployable artifact works end to end (signature, code paths, weights).

Outputs:
* ``reports/lstm_metrics.json`` -- compact DVC metrics (``dvc metrics show``)
* ``reports/plots/lstm_horizon_mape.csv`` -- MAPE by horizon, model vs baselines
  (``dvc plots show``)
* ``reports/plots/lstm_test_forecasts.png`` -- last test window per ticker
* ``reports/plots/lstm_input_attribution.csv`` -- occlusion attribution over
  the 100-day window (Day 4, explainability; see ``explain.py``)
* the same metrics + full per-ticker detail logged to the training MLflow run
  (``test_*`` metrics, tag ``beats_naive``), so the run carries its own verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.runs import stage_run
from finsight.models.lstm.data import WindowSet, load_price_series, make_windows, split_masks
from finsight.models.lstm.explain import attribution_figure, occlusion_attribution
from finsight.models.lstm.metrics import compare_to_baselines
from finsight.models.lstm.pyfunc_model import input_columns

log = get_logger(__name__)


def compact(result: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Flatten the evaluation into the small JSON tracked as a DVC metric."""

    def _pick(m: dict[str, Any]) -> dict[str, Any]:
        out = {"rmse": round(m["rmse"], 4), "mape": round(m["mape"], 4)}
        if m["directional_accuracy"] is not None:
            out["directional_accuracy"] = round(m["directional_accuracy"], 4)
        return out

    pooled = result["pooled"]
    return {
        "run_id": run_id,
        "test_windows": result["windows"],
        "model": _pick(pooled["model"]),
        "naive": _pick(pooled["naive"]),
        "drift": _pick(pooled["drift"]),
        "up_share": round(pooled["model"]["up_share"], 4),
        "skill_vs_naive": round(result["skill_vs_naive"], 4),
        "beats_naive": bool(result["beats_naive"]),
    }


def forecast_figure(test: WindowSet, pred: np.ndarray, history_days: int = 40) -> Any:
    """Small multiples: recent history, actual 14 days, LSTM (dashed) and naive."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    tickers = sorted(set(test.tickers.tolist()))
    cols = 5
    rows = int(np.ceil(len(tickers) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    horizon = test.y_raw.shape[1]
    for ax, ticker in zip(axes.flat, tickers, strict=False):
        idx = np.where(test.tickers == ticker)[0][-1]  # last test window
        hist = test.x_raw[idx][-history_days:]
        t_hist = np.arange(-len(hist) + 1, 1)
        t_fut = np.arange(1, horizon + 1)
        ax.plot(t_hist, hist, color="#555", linewidth=1)
        ax.plot(t_fut, test.y_raw[idx], color="#222", linewidth=1.4, label="actual")
        ax.plot(t_fut, pred[idx], "--", color="#2f6fb3", linewidth=1.6, label="LSTM")
        ax.plot(t_fut, np.repeat(hist[-1], horizon), ":", color="#c0504d", label="naive")
        ax.axvline(0.5, color="#bbb", linewidth=0.8)
        ax.set_title(f"{ticker}  (origin {str(test.origins[idx])[:10]})", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(tickers) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle("14-day forecasts on the held-out test set (last window per ticker)")
    fig.tight_layout()
    return fig


def run_evaluation(params: dict[str, Any], prices_path: Path, root: Path) -> dict[str, Any]:
    """Load the packaged model, score the test windows, write reports, log to MLflow."""
    import mlflow  # noqa: PLC0415

    model_dir = root / params["model_dir"]
    run_info = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
    model = mlflow.pyfunc.load_model(str(model_dir / "model"))

    lookback, horizon = int(params["lookback"]), int(params["horizon"])
    windows = make_windows(load_price_series(prices_path), lookback, horizon)
    masks = split_masks(windows, int(params["val_origins"]), int(params["test_origins"]), horizon)
    test = windows.subset(masks["test"])
    pred = model.predict(pd.DataFrame(test.x_raw, columns=input_columns(lookback))).to_numpy()
    result = compare_to_baselines(pred, test.x_raw, test.y_raw, test.tickers)
    metrics = compact(result, run_info["run_id"])

    plots = root / "reports" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["per_horizon"]).to_csv(plots / "lstm_horizon_mape.csv", index=False)
    fig = forecast_figure(test, pred)
    fig.savefig(plots / "lstm_test_forecasts.png", dpi=110)

    columns = input_columns(lookback)
    attribution = occlusion_attribution(
        lambda x: model.predict(pd.DataFrame(x, columns=columns)).to_numpy(),
        test.x_raw,
        test.y_raw,
        int(params.get("attribution_block", 10)),
    )
    attribution.to_csv(plots / "lstm_input_attribution.csv", index=False)
    attr_fig = attribution_figure(attribution)
    latest = attribution.iloc[0]
    older = attribution.iloc[1:]["mean_abs_change_pct"]
    metrics["attribution_latest_block_pct"] = round(float(latest["mean_abs_change_pct"]), 4)
    metrics["attribution_older_blocks_max_pct"] = (
        round(float(older.max()), 4) if len(older) else 0.0
    )

    with mlflow.start_run(run_id=run_info["run_id"]):
        flat = {
            "test_mape": metrics["model"]["mape"],
            "test_rmse": metrics["model"]["rmse"],
            "test_naive_mape": metrics["naive"]["mape"],
            "test_drift_mape": metrics["drift"]["mape"],
            "test_skill_vs_naive": metrics["skill_vs_naive"],
        }
        if "directional_accuracy" in metrics["model"]:
            flat["test_directional_accuracy"] = metrics["model"]["directional_accuracy"]
        mlflow.log_metrics(flat)
        mlflow.set_tags({"evaluated": "true", "beats_naive": str(metrics["beats_naive"]).lower()})
        mlflow.log_dict(result, "evaluation/test_metrics_full.json")
        mlflow.log_figure(fig, "plots/test_forecasts.png")
        mlflow.log_figure(attr_fig, "plots/input_attribution.png")
        mlflow.log_text(attribution.to_csv(index=False), "evaluation/input_attribution.csv")
        mlflow.log_metrics(
            {
                "attribution_latest_block_pct": metrics["attribution_latest_block_pct"],
                "attribution_older_blocks_max_pct": metrics["attribution_older_blocks_max_pct"],
            }
        )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.models.lstm.evaluate``)."""
    argparse.ArgumentParser(description="Evaluate the packaged LSTM").parse_args(argv)
    settings = get_settings()
    configure_logging("evaluate_lstm", settings.log_level, settings.root / "logs")
    params = {
        **settings.section("lstm"),
        "attribution_block": settings.section("lstm_explain")["attribution_block"],
    }
    prices_path = settings.root / settings.section("etl")["processed_prices_path"]
    try:
        from finsight.models.tracking import configure_tracking  # noqa: PLC0415

        with stage_run("evaluate_lstm", settings) as run:
            configure_tracking(params["experiment_name"])
            metrics = run_evaluation(params, prices_path, settings.root)
            path = settings.root / params["metrics_report"]
            path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            run.metrics = {k: metrics[k] for k in ("skill_vs_naive", "beats_naive", "test_windows")}
    except (FinSightError, ValueError, FileNotFoundError) as exc:
        print(f"evaluate_lstm failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
