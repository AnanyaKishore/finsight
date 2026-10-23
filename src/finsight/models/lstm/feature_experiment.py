"""Feature-impact experiment: does rolling volatility help the LSTM?

Logged to MLflow only (experiment ``finsight-lstm-experiments``); it never
changes the production model. Run with ``make feature-experiment`` or
``mlflow run . -e feature_experiment --env-manager local``.

Design (a controlled comparison)
--------------------------------
* **Same windows** for both variants: series start once ``volatility_20`` is
  defined, so the price-only variant does not get extra history.
* **Same configuration**: the one selected by ``train_lstm`` (``models/lstm/run.json``).
* **Variants**: ``price`` (1 channel) vs ``price+volatility_20`` (2 channels,
  each MinMax-scaled per window; the residual anchor stays the price).
* **Several seeds** (``lstm.experiment_seeds``): differences smaller than the
  seed-to-seed spread are not evidence.
* **Decision on validation only**: the extra feature is *recommended* only if
  its mean validation MAPE is lower by more than **two** seed standard
  deviations (the larger of the two variants'; with 3 seeds a looser rule
  "finds" improvements in pure noise). Test MAPE is reported for information,
  not used to decide (using it would leak the test set into model design).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.models.lstm.data import WindowSet, make_windows, split_masks
from finsight.models.lstm.metrics import compare_to_baselines
from finsight.models.lstm.train import fit, predict_prices, set_seeds

log = get_logger(__name__)

FEATURE = "volatility_20"


def load_aligned(path: Path, feature: str) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Price and feature series per ticker, restricted to dates where the feature exists."""
    frame = pd.read_parquet(path, columns=["ticker", "date", "price", feature])
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["price", feature]).sort_values(["ticker", "date"])
    prices, features = {}, {}
    for ticker, group in frame.groupby("ticker"):
        indexed = group.set_index("date")
        prices[ticker] = indexed["price"].astype("float64")
        features[ticker] = indexed[feature].astype("float64")
    return prices, features


def build_windows(
    prices: dict[str, pd.Series], features: dict[str, pd.Series], lookback: int, horizon: int
) -> tuple[WindowSet, WindowSet]:
    """``(price_only, price_plus_feature)`` window sets over identical windows."""
    base = make_windows(prices, lookback, horizon)
    extra = make_windows(features, lookback, horizon)
    with_feature = WindowSet(
        base.x_raw,
        base.y_raw,
        base.tickers,
        base.origins,
        base.position,
        base.n_per_ticker,
        extra.x_raw[:, :, None],
    )
    return base, with_feature


def run_experiment(
    params: dict[str, Any], prices_path: Path, root: Path, seeds: Sequence[int]
) -> dict[str, Any]:
    """Train both variants for every seed inside one MLflow parent run."""
    import mlflow  # noqa: PLC0415

    from finsight.models.tracking import lineage_tags  # noqa: PLC0415

    selected = json.loads((root / params["model_dir"] / "run.json").read_text(encoding="utf-8"))
    base_cfg = dict(selected["selected_config"])
    lookback, horizon = int(params["lookback"]), int(params["horizon"])
    prices, features = load_aligned(prices_path, FEATURE)
    variants = dict(
        zip(
            ("price", f"price+{FEATURE}"),
            build_windows(prices, features, lookback, horizon),
            strict=True,
        )
    )
    masks = split_masks(
        variants["price"], int(params["val_origins"]), int(params["test_origins"]), horizon
    )
    rows = []
    with mlflow.start_run(run_name=f"feature-exp-{datetime.now(UTC):%Y%m%d-%H%M%S}") as parent:
        mlflow.set_tags(
            {
                **lineage_tags(root, str(prices_path.relative_to(root)), int(seeds[0])),
                "stage": "feature_experiment",
                "run_role": "parent",
                "feature": FEATURE,
                "base_config_from_run": selected["run_id"],
            }
        )
        mlflow.log_params(
            {**{f"cfg_{k}": v for k, v in base_cfg.items()}, "seeds": ",".join(map(str, seeds))}
        )
        for name, ws in variants.items():
            cfg = {**base_cfg, "input_size": 1 if ws.x_extra is None else 1 + ws.x_extra.shape[2]}
            train, val, test = (ws.subset(masks[k]) for k in ("train", "val", "test"))
            for seed in seeds:
                with mlflow.start_run(run_name=f"{name}-s{seed}", nested=True):
                    mlflow.set_tags(
                        {"run_role": "variant", "variant": name, "parent_run": parent.info.run_id}
                    )
                    mlflow.log_params(
                        {"variant": name, "seed": seed, "input_size": cfg["input_size"]}
                    )
                    set_seeds(seed, int(params["torch_threads"]))
                    fitted = fit(
                        cfg,
                        train,
                        val,
                        epochs=params["max_epochs"],
                        patience=params["patience"],
                        batch_size=params["batch_size"],
                        seed=seed,
                    )
                    val_res = compare_to_baselines(
                        predict_prices(fitted.net, val), val.x_raw, val.y_raw, val.tickers
                    )
                    test_res = compare_to_baselines(
                        predict_prices(fitted.net, test), test.x_raw, test.y_raw, test.tickers
                    )
                    row = {
                        "variant": name,
                        "seed": seed,
                        "best_epoch": fitted.best_epoch,
                        "val_mape": val_res["pooled"]["model"]["mape"],
                        "val_skill_vs_naive": val_res["skill_vs_naive"],
                        "test_mape": test_res["pooled"]["model"]["mape"],
                        "test_skill_vs_naive": test_res["skill_vs_naive"],
                    }
                    mlflow.log_metrics(
                        {k: float(v) for k, v in row.items() if k not in ("variant",)}
                    )
                    rows.append(row)
                    log.info("variant done", extra=row)

        table = pd.DataFrame(rows)
        summary = (
            table.groupby("variant")[["val_mape", "test_mape", "val_skill_vs_naive"]]
            .agg(["mean", "std"])
            .round(4)
        )
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        base_row = summary.loc["price"]
        feat_row = summary.loc[f"price+{FEATURE}"]
        improvement = float(base_row["val_mape_mean"] - feat_row["val_mape_mean"])
        spread = float(np.nanmax([base_row["val_mape_std"], feat_row["val_mape_std"], 0.0]))
        recommend = bool(improvement > 2 * spread)
        verdict = {
            "feature": FEATURE,
            "seeds": list(seeds),
            "val_mape_improvement": round(improvement, 4),
            "seed_std": round(spread, 4),
            "decision_rule": "improvement > 2 x seed_std (validation only)",
            "recommend_feature": recommend,
            "val_skill_vs_naive": {
                "price": float(base_row["val_skill_vs_naive_mean"]),
                f"price+{FEATURE}": float(feat_row["val_skill_vs_naive_mean"]),
            },
            "summary": summary.reset_index().to_dict("records"),
            "runs": rows,
        }
        mlflow.log_metrics({"val_mape_improvement": improvement, "seed_std": spread})
        mlflow.set_tag("recommend_feature", str(recommend).lower())
        mlflow.log_dict(verdict, "experiment/verdict.json")
        verdict["run_id"] = parent.info.run_id
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point (``python -m finsight.models.lstm.feature_experiment``)."""
    parser = argparse.ArgumentParser(description="Volatility-as-input experiment (logged only)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-epochs", type=int, default=-1)
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("feature_experiment", settings.log_level, settings.root / "logs")
    params = dict(settings.section("lstm"))
    if args.max_epochs > 0:
        params["max_epochs"] = args.max_epochs
    seeds = args.seeds or list(settings.section("lstm_explain")["experiment_seeds"])
    prices_path = settings.root / settings.section("etl")["processed_prices_path"]
    try:
        from finsight.models.tracking import configure_tracking  # noqa: PLC0415

        configure_tracking("finsight-lstm-experiments")
        verdict = run_experiment(params, prices_path, settings.root, seeds)
    except (FinSightError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"feature_experiment failed: {exc}", file=sys.stderr)
        return 1
    out = settings.root / "reports" / "lstm_feature_experiment.json"
    out.write_text(json.dumps({k: v for k, v in verdict.items() if k != "runs"}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verdict.items() if k != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
