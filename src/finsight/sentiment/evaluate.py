"""DVC stage ``evaluate_sentiment@<candidate>``: measure, calibrate and package one model.

Why this stage exists
---------------------
The proposal's model is DistilBERT fine-tuned on **SST-2 (movie reviews)**;
Day 5 adds **FinBERT** as a challenger. Before any score reaches the portal,
each candidate is measured on labelled *financial* text, and the promotion
gate (``finsight.registry``) picks the production model from these numbers.

Datasets (``sentiment.datasets``)
---------------------------------
* ``twitter_financial_news`` (the **gate** dataset): finance news headlines,
  bearish / bullish / neutral; no candidate was trained on it.
* ``financial_phrasebank``: reported for comparability with Day 4, flagged
  ``contaminated`` for candidates trained on it (FinBERT).

Procedure (per dataset)
-----------------------
1. Stratified, seeded split into a **calibration** half and a **test** half.
2. Score every sentence once (``p_pos - p_neg``).
3. **Binary** metrics (positive vs negative sentences, test half) and
   **3-class** metrics with a neutral band calibrated on the calibration half,
   each against a majority-class baseline.
4. The band from the *gate* dataset is the one packaged with the model.

Also: latency (single text, p50/p95), throughput, the most confident mistakes
(error analysis), and an MLflow pyfunc model saved to
``models/sentiment/candidates/<candidate>/model`` with ``run.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.runs import stage_run
from finsight.sentiment.scoring import (
    CLASSES,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    SentimentScorer,
    labels_for,
    model_version,
)

log = get_logger(__name__)

BINARY = (NEGATIVE, POSITIVE)


# --------------------------------------------------------------------------- #
# Metrics (small and explicit; no scikit-learn dependency needed)
# --------------------------------------------------------------------------- #
def confusion(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """Confusion matrix, rows = true class, columns = predicted class."""
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        if p in index:
            matrix[index[t], index[p]] += 1
    return matrix


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, macro-F1 and per-class precision / recall / F1 / support.

    A prediction outside ``classes`` (e.g. "neutral" in the binary task) counts
    as wrong for accuracy and recall.
    """
    y_true, y_pred = list(y_true), list(y_pred)
    matrix = confusion(y_true, y_pred, classes)
    per_class, f1s = {}, []
    for i, cls in enumerate(classes):
        tp = int(matrix[i, i])
        predicted = int(sum(1 for p in y_pred if p == cls))
        support = int(sum(1 for t in y_true if t == cls))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1s.append(f1)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return {
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion": matrix.tolist(),
        "classes": list(classes),
        "n": len(y_true),
    }


def stratified_split(labels: pd.Series, share: float, seed: int) -> np.ndarray:
    """Boolean mask selecting ``share`` of each class (seeded) for calibration."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for cls in sorted(labels.unique()):
        idx = np.flatnonzero(labels.to_numpy() == cls)
        rng.shuffle(idx)
        mask[idx[: int(round(len(idx) * share))]] = True
    return mask


def band_candidates(scores: np.ndarray) -> list[float]:
    """Candidate neutral bands: 0 plus quantiles of ``|score|`` (deduplicated)."""
    quantiles = np.quantile(np.abs(scores), np.linspace(0.0, 0.98, 50))
    return sorted({0.0, *(round(float(q), 4) for q in quantiles)})


def calibrate_band(scores: np.ndarray, labels: Sequence[str]) -> tuple[float, pd.DataFrame]:
    """Band maximising 3-class macro-F1 (ties -> the smaller band).

    Returns:
        ``(best_band, curve)``; ``curve`` has one row per candidate.
    """
    rows = []
    for band in band_candidates(scores):
        pred = labels_for(scores, band)
        m = classification_metrics(labels, pred, CLASSES)
        rows.append(
            {
                "band": band,
                "macro_f1": round(m["macro_f1"], 6),
                "accuracy": round(m["accuracy"], 6),
                "neutral_share": round(float(np.mean(pred == NEUTRAL)), 6),
            }
        )
    curve = pd.DataFrame(rows)
    best = curve.sort_values(["macro_f1", "band"], ascending=[False, True]).iloc[0]
    return float(best["band"]), curve


def most_confident_errors(
    frame: pd.DataFrame, predicted: np.ndarray, limit: int = 10
) -> list[dict[str, Any]]:
    """Wrong predictions with the largest ``|score|`` (error analysis)."""
    wrong = frame.assign(predicted=predicted)[lambda f: f["label"] != f["predicted"]]
    wrong = wrong.reindex(wrong["score"].abs().sort_values(ascending=False).index)
    return [
        {
            "sentence": r.sentence,
            "label": r.label,
            "predicted": r.predicted,
            "score": round(float(r.score), 4),
        }
        for r in wrong.head(limit).itertuples()
    ]


def evaluate_scores(
    frame: pd.DataFrame, calibration_share: float, seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Binary + calibrated 3-class evaluation from pre-computed ``frame['score']``.

    Args:
        frame: PhraseBank rows with ``sentence``, ``label`` and ``score``.
        calibration_share: Fraction of each class used to calibrate the band.
        seed: Split seed.

    Returns:
        ``(result, band_curve)``.

    Raises:
        ValueError: If a split has no positive or negative sentences.
    """
    calib_mask = stratified_split(frame["label"], calibration_share, seed)
    calib, test = frame[calib_mask], frame[~calib_mask]
    for name, part in (("calibration", calib), ("test", test)):
        if not set(BINARY) <= set(part["label"]):
            raise ValueError(f"{name} split lacks positive or negative sentences")

    # Binary task (what SST-2 was trained for): sign of the score.
    binary = test[test["label"].isin(BINARY)]
    binary_pred = np.where(binary["score"].to_numpy() >= 0, POSITIVE, NEGATIVE)
    majority_binary = calib[calib["label"].isin(BINARY)]["label"].value_counts().idxmax()
    binary_metrics = classification_metrics(binary["label"], binary_pred, BINARY)
    binary_base = classification_metrics(binary["label"], [majority_binary] * len(binary), BINARY)

    # 3-class task with the band calibrated on the calibration half only.
    band, curve = calibrate_band(calib["score"].to_numpy(), calib["label"].tolist())
    test_pred = labels_for(test["score"].to_numpy(), band)
    three = classification_metrics(test["label"], test_pred, CLASSES)
    majority = calib["label"].value_counts().idxmax()
    three_base = classification_metrics(test["label"], [majority] * len(test), CLASSES)

    result = {
        "n_calibration": int(len(calib)),
        "n_test": int(len(test)),
        "binary": binary_metrics,
        "binary_majority_baseline": {**binary_base, "predicts": majority_binary},
        "neutral_band": band,
        "three_class": three,
        "three_class_majority_baseline": {**three_base, "predicts": majority},
        "label_distribution": frame["label"].value_counts().sort_index().to_dict(),
        "errors_binary": most_confident_errors(binary, binary_pred),
        "errors_three_class": most_confident_errors(test, test_pred),
    }
    return result, curve


def measure_latency(
    score_fn: Callable[[list[str]], np.ndarray], texts: Sequence[str], repeats: int = 30
) -> dict[str, float]:
    """Single-text latency (p50 / p95, ms) on ``repeats`` texts."""
    timings = []
    for text in list(texts)[:repeats]:
        started = time.perf_counter()
        score_fn([text])
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "latency_p50_ms": round(float(np.percentile(timings, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(timings, 95)), 2),
    }


def compact(
    results: dict[str, dict[str, Any]],
    perf: dict[str, float],
    run_id: str,
    candidate: str,
    gate_dataset: str,
    contaminated: Sequence[str],
) -> dict[str, Any]:
    """Small JSON tracked as a DVC metric; ``gate_*`` fields drive promotion."""
    gate = results[gate_dataset]
    b, bb = gate["binary"], gate["binary_majority_baseline"]
    t, tb = gate["three_class"], gate["three_class_majority_baseline"]
    out: dict[str, Any] = {
        "run_id": run_id,
        "candidate": candidate,
        "gate_dataset": gate_dataset,
        "n_test": gate["n_test"],
        "neutral_band": round(gate["neutral_band"], 4),
        "gate_macro_f1": round(t["macro_f1"], 4),
        "gate_accuracy": round(t["accuracy"], 4),
        "gate_baseline_macro_f1": round(tb["macro_f1"], 4),
        "gate_baseline_accuracy": round(tb["accuracy"], 4),
        "gate_binary_accuracy": round(b["accuracy"], 4),
        "gate_binary_macro_f1": round(b["macro_f1"], 4),
        "gate_binary_baseline_accuracy": round(bb["accuracy"], 4),
        "beats_baseline": bool(t["macro_f1"] > tb["macro_f1"]),
    }
    for name, res in results.items():
        if name == gate_dataset:
            continue
        out[f"{name}_macro_f1"] = round(res["three_class"]["macro_f1"], 4)
        out[f"{name}_binary_accuracy"] = round(res["binary"]["accuracy"], 4)
        out[f"{name}_contaminated"] = name in contaminated
    out.update({k: perf[k] for k in ("latency_p50_ms", "latency_p95_ms", "throughput_texts_per_s")})
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def confusion_figure(result: dict[str, Any], title: str = "") -> Any:
    """Binary and 3-class confusion matrices side by side (counts)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in (
        (axes[0], "binary", "Binary (pos/neg sentences)"),
        (axes[1], "three_class", f"3-class, neutral band {result['neutral_band']:.3f}"),
    ):
        matrix = np.array(result[key]["confusion"])
        classes = result[key]["classes"]
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            colour = "white" if value > matrix.max() / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=colour)
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(
            f"{title}\nacc {result[key]['accuracy']:.3f}  F1 {result[key]['macro_f1']:.3f}"
        )
    fig.suptitle(title or "Sentiment model on the test half")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #
def package_model(
    base_dir: Path, config: dict[str, Any], sample: list[str], root: Path, model_dir: Path
) -> dict[str, Any]:
    """Log the pyfunc model to the active run and save a local copy for DVC."""
    import mlflow  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from mlflow.models import infer_signature  # noqa: PLC0415

    from finsight.sentiment.pyfunc_model import OUTPUT_COLUMNS  # noqa: PLC0415

    sample_in = pd.DataFrame({"text": sample})
    sample_out = pd.DataFrame(
        {"label": ["positive"] * len(sample), "confidence": 0.9, "score": 0.8},
        columns=OUTPUT_COLUMNS,
    )
    signature = infer_signature(sample_in, sample_out)
    requirements = [
        f"torch=={torch.__version__.split('+')[0]}",
        f"transformers=={transformers.__version__}",
        f"mlflow=={mlflow.__version__}",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        common = {
            "python_model": str(Path(__file__).with_name("pyfunc_entry.py")),
            "artifacts": {"hf_model": str(base_dir), "config": str(cfg_path)},
            "code_paths": [str(root / "src" / "finsight")],
            "signature": signature,
            "input_example": sample_in,
            "pip_requirements": requirements,
        }
        info = mlflow.pyfunc.log_model(name="model", **common)
        local = model_dir / "model"
        shutil.rmtree(local, ignore_errors=True)
        mlflow.pyfunc.save_model(path=str(local), **common)
    for cache in list(local.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    return {"model_uri": info.model_uri, "model_id": getattr(info, "model_id", None)}


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def eval_fingerprint(params: dict[str, Any]) -> str:
    """Identify the gate evaluation (dataset revision + split).

    Metrics with the same fingerprint are directly comparable between model versions.
    """
    ds = params["datasets"][params["gate_dataset"]]
    return (
        f"{params['gate_dataset']}@{ds['revision'][:12]}"
        f"-share{float(params['calibration_share'])}-seed{int(params['seed'])}"
    )


def evaluate_candidate_scores(
    params: dict[str, Any], datasets: dict[str, pd.DataFrame]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Run :func:`evaluate_scores` on every scored dataset (``score`` column present)."""
    results, curves = {}, {}
    for name, frame in datasets.items():
        results[name], curves[name] = evaluate_scores(
            frame, float(params["calibration_share"]), int(params["seed"])
        )
    return results, curves


def run_evaluation(params: dict[str, Any], candidate: str, root: Path) -> dict[str, Any]:
    """Score every dataset, evaluate, calibrate, package -- all inside one MLflow run."""
    import mlflow  # noqa: PLC0415

    from finsight.models.tracking import lineage_tags  # noqa: PLC0415

    cand = params["candidates"][candidate]
    gate_ds = params["gate_dataset"]
    contaminated = list(cand.get("trained_on", []))
    base_dir, out_dir = root / cand["base_dir"], root / cand["model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = SentimentScorer(
        base_dir,
        max_length=int(params["max_length"]),
        batch_size=int(params["batch_size"]),
        threads=int(params["torch_threads"]),
    )
    datasets, n_texts, started = {}, 0, time.perf_counter()
    for name, spec in params["datasets"].items():
        frame = pd.read_csv(root / spec["path"])
        frame["score"] = scorer.scores(frame["sentence"].tolist())
        datasets[name] = frame
        n_texts += len(frame)
    elapsed = time.perf_counter() - started
    perf = {
        "throughput_texts_per_s": round(n_texts / max(elapsed, 1e-9), 1),
        **measure_latency(scorer.scores, datasets[gate_ds]["sentence"].tolist()),
    }
    results, curves = evaluate_candidate_scores(params, datasets)
    band = results[gate_ds]["neutral_band"]
    config = {
        "candidate": candidate,
        "base_model": cand["repo"],
        "base_revision": cand["revision"],
        "neutral_band": band,
        "max_length": int(params["max_length"]),
        "batch_size": int(params["batch_size"]),
        "serving_threads": 2,
        "model_version": model_version(cand["revision"], band, int(params["max_length"])),
        "eval_fingerprint": eval_fingerprint(params),
    }
    title = f"{candidate} on {gate_ds} (test half)"

    with mlflow.start_run(
        run_name=f"sentiment-{candidate}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    ) as run:
        mlflow.set_tags(
            {
                **lineage_tags(
                    root,
                    None,
                    int(params["seed"]),
                    dvc_inputs={
                        f"dvc_{name}_md5": ("fetch_sentiment_data", spec["path"])
                        for name, spec in params["datasets"].items()
                    }
                    | {"dvc_base_model_md5": (f"fetch_sentiment@{candidate}", cand["base_dir"])},
                ),
                "stage": "evaluate_sentiment",
                "candidate": candidate,
                "model_family": candidate,
                "trained_on": ",".join(contaminated) or "-",
                "model_version": config["model_version"],
                "eval_fingerprint": config["eval_fingerprint"],
            }
        )
        mlflow.log_params(
            {
                "candidate": candidate,
                "base_model": cand["repo"],
                "base_revision": cand["revision"],
                "gate_dataset": gate_ds,
                "calibration_share": params["calibration_share"],
                "seed": params["seed"],
                "max_length": params["max_length"],
                "batch_size": params["batch_size"],
                "torch_threads": params["torch_threads"],
                "neutral_band": band,
                **{f"n_{name}": len(frame) for name, frame in datasets.items()},
            }
        )
        for name, frame in datasets.items():
            spec = params["datasets"][name]
            with warnings.catch_warnings():  # local path matches 2 equivalent source types
                warnings.simplefilter("ignore", UserWarning)
                dataset = mlflow.data.from_pandas(
                    frame[["id", "sentence", "label"]],
                    source=spec["path"],
                    name=name,
                    targets="label",
                )
            mlflow.log_input(dataset, context="evaluation")
        metrics = compact(results, perf, run.info.run_id, candidate, gate_ds, contaminated)
        numeric = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
        mlflow.log_metrics(numeric)
        mlflow.set_tag("beats_baseline", str(metrics["beats_baseline"]).lower())
        for name, res in results.items():
            mlflow.log_dict(res, f"evaluation/{name}_full.json")
            mlflow.log_text(curves[name].to_csv(index=False), f"evaluation/{name}_band_curve.csv")
        mlflow.log_figure(confusion_figure(results[gate_ds], title), "plots/confusion.png")
        packaged = package_model(
            base_dir, config, datasets[gate_ds]["sentence"].head(3).tolist(), root, out_dir
        )
        run_id, experiment_id = run.info.run_id, run.info.experiment_id

    run_info = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_name": params["experiment_name"],
        "tracking_uri": mlflow.get_tracking_uri(),
        "model_uri": packaged["model_uri"],
        "model_id": packaged["model_id"],
        "candidate": candidate,
        "model_version": config["model_version"],
        "neutral_band": band,
        "eval_fingerprint": config["eval_fingerprint"],
        "gate_metrics": {k: v for k, v in metrics.items() if k.startswith("gate_")},
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    return {"metrics": metrics, "results": results, "curves": curves, "title": title}


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.sentiment.evaluate --candidate <key>``)."""
    parser = argparse.ArgumentParser(description="Evaluate + calibrate + package one candidate")
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("evaluate_sentiment", settings.log_level, settings.root / "logs")
    params = settings.section("sentiment")
    if args.candidate not in params["candidates"]:
        print(f"unknown candidate {args.candidate!r}", file=sys.stderr)
        return 2
    try:
        from finsight.models.tracking import configure_tracking  # noqa: PLC0415

        with stage_run(f"evaluate_sentiment@{args.candidate}", settings) as run:
            configure_tracking(params["experiment_name"])
            outcome = run_evaluation(params, args.candidate, settings.root)
            reports = settings.root / "reports"
            plots = reports / "plots"
            plots.mkdir(parents=True, exist_ok=True)
            (reports / f"sentiment_eval_{args.candidate}.json").write_text(
                json.dumps(outcome["metrics"], indent=2) + "\n", encoding="utf-8"
            )
            gate_ds = params["gate_dataset"]
            outcome["curves"][gate_ds].to_csv(
                plots / f"sentiment_band_{args.candidate}.csv", index=False
            )
            confusion_figure(outcome["results"][gate_ds], outcome["title"]).savefig(
                plots / f"sentiment_confusion_{args.candidate}.png", dpi=110
            )
            run.metrics = {
                k: outcome["metrics"][k]
                for k in ("gate_macro_f1", "neutral_band", "beats_baseline")
            }
    except (FinSightError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"evaluate_sentiment failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(outcome["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
