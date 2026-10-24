"""DVC stage ``register``: register new model versions and run the promotion gate.

For each model family:

* **LSTM** (``finsight-lstm``): the version trained by ``train_lstm`` is
  registered (once per MLflow run) with its test metrics as tags. If a
  champion exists, it is **re-scored on today's test windows** with the
  registered artifact, so both are compared on identical data.
* **Sentiment** (``finsight-sentiment``): every candidate evaluated by
  ``evaluate_sentiment@<candidate>`` is registered; the best accepted one
  (gate macro-F1 on the Twitter financial-news test half) challenges the
  champion. If the champion was evaluated with a different split or
  dataset version (``eval_fingerprint``), it is re-evaluated first.

The winners get the ``@production`` alias (+ ``Production`` stage), challengers
``@staging``. Versions tagged ``rolled_back`` are never auto-promoted again.

Output: ``reports/promotion.json`` -- the decision, its evidence, and the
production pointer each downstream consumer resolves (``score_sentiment``,
serving, API). Running the stage twice on the same inputs registers nothing new.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.exceptions import MlflowException

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.runs import stage_run
from finsight.registry import client as reg
from finsight.registry.gate import PRODUCTION, decide, lstm_accepted, pick_best

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Re-scoring a champion on current data
# --------------------------------------------------------------------------- #
def lstm_test_mape(model: Any, lstm: dict[str, Any], prices_path: Path) -> float | None:
    """MAPE of any LSTM pyfunc on the current test windows (``None`` if incompatible)."""
    from finsight.models.lstm.data import load_price_series, make_windows, split_masks  # noqa
    from finsight.models.lstm.metrics import compare_to_baselines  # noqa: PLC0415
    from finsight.models.lstm.pyfunc_model import input_columns  # noqa: PLC0415

    lookback, horizon = int(lstm["lookback"]), int(lstm["horizon"])
    windows = make_windows(load_price_series(prices_path), lookback, horizon)
    masks = split_masks(windows, int(lstm["val_origins"]), int(lstm["test_origins"]), horizon)
    test = windows.subset(masks["test"])
    try:
        frame = pd.DataFrame(test.x_raw, columns=input_columns(lookback))
        pred = model.predict(frame).to_numpy()
    except Exception as exc:  # noqa: BLE001 -- e.g. a champion with another lookback
        log.warning("champion not comparable", extra={"error": str(exc)[:200]})
        return None
    if pred.shape != test.y_raw.shape:
        return None
    return float(
        compare_to_baselines(pred, test.x_raw, test.y_raw, test.tickers)["pooled"]["model"]["mape"]
    )


def sentiment_test_macro_f1(model: Any, frame: pd.DataFrame, share: float, seed: int) -> float:
    """3-class macro-F1 of a sentiment pyfunc on the gate dataset's test half."""
    from finsight.sentiment.evaluate import classification_metrics, stratified_split  # noqa
    from finsight.sentiment.scoring import CLASSES  # noqa: PLC0415

    test = frame[~stratified_split(frame["label"], share, seed)]
    pred = model.predict(pd.DataFrame({"text": test["sentence"].tolist()}))["label"]
    return float(classification_metrics(test["label"], pred, CLASSES)["macro_f1"])


def _load(uri: str) -> Any:
    import mlflow  # noqa: PLC0415

    return mlflow.pyfunc.load_model(uri)


def _pointer(name: str, version: Any, **extra: Any) -> dict[str, Any]:
    return {
        "registered_name": name,
        "version": str(version.version),
        "run_id": version.run_id,
        "model_uri": f"models:/{name}/{version.version}",
        **extra,
    }


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def lstm_gate(client: Any, settings_params: dict[str, Any], root: Path, loader=_load) -> dict:
    """Register the newly trained LSTM and decide whether it becomes production."""
    registry = settings_params["registry"]
    lstm = settings_params["lstm"]
    name = registry["lstm_name"]
    run_info = json.loads((root / lstm["model_dir"] / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / lstm["metrics_report"]).read_text(encoding="utf-8"))
    test_mape, naive_mape = float(metrics["model"]["mape"]), float(metrics["naive"]["mape"])
    accepted = lstm_accepted(test_mape, naive_mape, registry["lstm_acceptance_max_ratio_to_naive"])

    reg.ensure_registered_model(client, name, "FinSight 14-day LSTM price forecaster (pyfunc).")
    challenger = reg.ensure_version(
        client,
        name,
        run_info["model_uri"],
        run_info["run_id"],
        {
            "test_mape": test_mape,
            "naive_mape": naive_mape,
            "skill_vs_naive": metrics["skill_vs_naive"],
            "accepted": str(accepted).lower(),
            "selected_config": json.dumps(run_info.get("selected_config", {}), sort_keys=True),
        },
    )
    champion = reg.alias_version(client, name, PRODUCTION)
    champion_metric = None
    if champion is not None and str(champion.version) != str(challenger.version):
        prices = root / settings_params["etl"]["processed_prices_path"]
        champion_metric = lstm_test_mape(loader(f"models:/{name}/{champion.version}"), lstm, prices)
    eligible = not reg.is_rolled_back(challenger)
    decision = decide(
        metric="test_mape",
        challenger_version=str(challenger.version) if eligible else None,
        challenger_metric=test_mape,
        challenger_accepted=accepted,
        champion_version=str(champion.version) if champion else None,
        champion_metric=champion_metric,
        higher_is_better=False,
        min_improvement=float(registry["lstm_min_improvement"]),
    )
    _apply(client, name, decision)
    production = reg.alias_version(client, name, PRODUCTION)
    return {
        "decision": decision.as_dict(),
        "production": _pointer(name, production) if production else None,
    }


def sentiment_gate(
    client: Any, settings_params: dict[str, Any], root: Path, loader=_load
) -> dict[str, Any]:
    """Register every sentiment candidate and promote the best one if it beats production."""
    from finsight.sentiment.evaluate import eval_fingerprint  # noqa: PLC0415

    registry, sent = settings_params["registry"], settings_params["sentiment"]
    name = registry["sentiment_name"]
    reg.ensure_registered_model(client, name, "FinSight news sentiment (pyfunc, text -> score).")
    fingerprint = eval_fingerprint(sent)
    order = list(sent["candidates"])
    candidates: dict[str, dict[str, Any]] = {}
    versions: dict[str, Any] = {}
    for key in order:
        cand = sent["candidates"][key]
        run_info = json.loads((root / cand["model_dir"] / "run.json").read_text(encoding="utf-8"))
        report = json.loads(
            (root / "reports" / f"sentiment_eval_{key}.json").read_text(encoding="utf-8")
        )
        version = reg.ensure_version(
            client,
            name,
            run_info["model_uri"],
            run_info["run_id"],
            {
                "candidate": key,
                "gate_macro_f1": report["gate_macro_f1"],
                "gate_baseline_macro_f1": report["gate_baseline_macro_f1"],
                "accepted": str(report["beats_baseline"]).lower(),
                "eval_fingerprint": run_info["eval_fingerprint"],
                "model_version": run_info["model_version"],
                "neutral_band": run_info["neutral_band"],
                "trained_on": ",".join(cand.get("trained_on", [])) or "-",
            },
        )
        versions[key] = version
        candidates[key] = {
            "version": str(version.version),
            "metric": float(report["gate_macro_f1"]),
            "accepted": bool(report["beats_baseline"]),
            "rolled_back": reg.is_rolled_back(version),
        }
    best = pick_best(candidates, order)
    champion = reg.alias_version(client, name, PRODUCTION)
    champion_metric = None
    if champion is not None:
        tags = champion.tags or {}
        if tags.get("eval_fingerprint") == fingerprint and "gate_macro_f1" in tags:
            champion_metric = float(tags["gate_macro_f1"])
        else:
            gate = sent["datasets"][sent["gate_dataset"]]
            frame = pd.read_csv(root / gate["path"])
            champion_metric = sentiment_test_macro_f1(
                loader(f"models:/{name}/{champion.version}"),
                frame,
                float(sent["calibration_share"]),
                int(sent["seed"]),
            )
    decision = decide(
        metric="gate_macro_f1",
        challenger_version=str(versions[best].version) if best else None,
        challenger_metric=candidates[best]["metric"] if best else None,
        challenger_accepted=candidates[best]["accepted"] if best else False,
        champion_version=str(champion.version) if champion else None,
        champion_metric=champion_metric,
        higher_is_better=True,
        min_improvement=float(registry["sentiment_min_improvement"]),
    )
    _apply(client, name, decision)
    production = reg.alias_version(client, name, PRODUCTION)
    pointer = None
    if production is not None:
        tags = production.tags or {}
        pointer = _pointer(
            name,
            production,
            candidate=tags.get("candidate"),
            model_version=tags.get("model_version"),
            neutral_band=float(tags.get("neutral_band", 0.0)),
        )
    return {
        "decision": {**decision.as_dict(), "best_candidate": best},
        "candidates": candidates,
        "production": pointer,
    }


def _apply(client: Any, name: str, decision: Any) -> None:
    if decision.challenger_version is None:
        return
    if decision.promote:
        reg.set_production(client, name, decision.challenger_version)
    elif decision.challenger_version != decision.champion_version:
        reg.set_staging(client, name, decision.challenger_version)
    log.info(
        "gate decision",
        extra={"model": name, "promote": decision.promote, "reason": decision.reason},
    )


def run_register(params: dict[str, Any], root: Path, client: Any | None = None) -> dict[str, Any]:
    """Both gates; returns the content of ``reports/promotion.json``."""
    client = client or reg.get_client()
    return {
        "lstm": lstm_gate(client, params, root),
        "sentiment": sentiment_gate(client, params, root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.registry.register``)."""
    argparse.ArgumentParser(description="Register models and run the promotion gate").parse_args(
        argv
    )
    settings = get_settings()
    configure_logging("register", settings.log_level, settings.root / "logs")
    params = {k: settings.section(k) for k in ("registry", "lstm", "sentiment", "etl")}
    try:
        from finsight.models.tracking import configure_tracking  # noqa: PLC0415

        with stage_run("register", settings) as run:
            configure_tracking(None)
            result = run_register(params, settings.root)
            path = settings.root / params["registry"]["report"]
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            run.metrics = {
                f"{m}_promoted": result[m]["decision"]["promote"] for m in ("lstm", "sentiment")
            } | {
                f"{m}_production": (result[m]["production"] or {}).get("version")
                for m in ("lstm", "sentiment")
            }
    except (FinSightError, ValueError, FileNotFoundError, KeyError, MlflowException) as exc:
        print(f"register failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
