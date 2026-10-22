"""DVC stage ``train_lstm``: grid search, refit, and a fully traced MLflow run.

Procedure
---------
1. Build leak-free windows from ``data/processed/prices`` and split each
   ticker chronologically into train | purge | val | purge | test.
2. **Grid search** -- every configuration in ``lstm.grid`` is trained on
   *train* with early stopping on *val* loss, as an MLflow *child run*
   (per-epoch losses + validation metrics vs the naive baseline).
3. **Selection** -- lowest validation MAPE.
4. **Refit** -- the winner is retrained on train+val for the number of epochs
   that early stopping chose (the test windows are never touched here; they
   belong to ``evaluate_lstm``).
5. **Packaging** -- the refit model is logged to the *parent* run as an MLflow
   pyfunc model with signature + input example, and saved to ``models/lstm/``
   (DVC-tracked) together with ``run.json`` linking it to the MLflow run.

What is logged beyond autolog: git commit + dirty flag, DVC data hash,
export snapshot hash, seed, library versions, the dataset (``mlflow.log_input``),
``params.yaml``, ``dvc.lock``, per-configuration learning curves, validation
metrics *relative to baselines*, and the selection decision.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.runs import stage_run
from finsight.models.lstm.data import (
    WindowSet,
    load_price_series,
    make_windows,
    split_masks,
    unscale_rows,
)
from finsight.models.lstm.metrics import compare_to_baselines
from finsight.models.lstm.pyfunc_model import build_network, input_columns

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Training primitives
# --------------------------------------------------------------------------- #
def set_seeds(seed: int, threads: int) -> None:
    """Make CPU training repeatable (same seed + threads -> same weights)."""
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class FitResult:
    """Outcome of one training run."""

    net: Any
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float | None = None


def fit(
    config: dict[str, Any],
    train: WindowSet,
    val: WindowSet | None,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    on_epoch: Callable[[dict[str, float]], None] | None = None,
) -> FitResult:
    """Train one network; early-stop on ``val`` loss when ``val`` is given."""
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    net = build_network(config)
    optimiser = torch.optim.Adam(net.parameters(), lr=float(config["learning_rate"]))
    loss_fn = torch.nn.MSELoss()
    x_tr, y_tr, _, _ = train.scaled()
    x_tr, y_tr = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    if val is not None:
        x_va, y_va, _, _ = val.scaled()
        x_va, y_va = torch.from_numpy(x_va), torch.from_numpy(y_va)
    generator = torch.Generator().manual_seed(seed)

    result = FitResult(net=net)
    best_state, since_best = None, 0
    for epoch in range(1, int(epochs) + 1):
        net.train()
        order = torch.randperm(len(x_tr), generator=generator)
        total = 0.0
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            optimiser.zero_grad()
            loss = loss_fn(net(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "train_loss": total / len(order)}
        if val is not None:
            net.eval()
            with torch.no_grad():
                row["val_loss"] = float(loss_fn(net(x_va), y_va))
            if result.best_val_loss is None or row["val_loss"] < result.best_val_loss:
                result.best_val_loss, result.best_epoch, since_best = row["val_loss"], epoch, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                since_best += 1
        else:
            result.best_epoch = epoch
        result.history.append(row)
        if on_epoch:
            on_epoch(row)
        if val is not None and since_best >= int(patience):
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return result


def predict_prices(net: Any, ws: WindowSet) -> np.ndarray:
    """Forecast prices (USD) for every window of ``ws``."""
    import torch  # noqa: PLC0415

    x, _, lo, hi = ws.scaled()
    with torch.no_grad():
        out = net(torch.from_numpy(x)).numpy().astype("float64")
    return unscale_rows(out, lo, hi)


def grid_configs(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand ``lstm.grid`` into concrete configurations (deterministic order)."""
    grid = params["grid"]
    keys = sorted(grid)
    base = {
        "lookback": int(params["lookback"]),
        "horizon": int(params["horizon"]),
        "dropout": float(params["dropout"]),
    }
    return [
        {**base, **dict(zip(keys, values, strict=True))}
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def _config_name(cfg: dict[str, Any]) -> str:
    return f"h{cfg['hidden_size']}-l{cfg['num_layers']}-lr{cfg['learning_rate']}"


# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #
def package_model(
    net: Any, config: dict[str, Any], example: WindowSet, root: Path, model_dir: Path
) -> dict[str, Any]:
    """Log the pyfunc model to the active run and save a local copy for DVC."""
    import mlflow  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from mlflow.models import infer_signature  # noqa: PLC0415

    columns = input_columns(int(config["lookback"]))
    sample = pd.DataFrame(example.x_raw[:2], columns=columns)
    sample_out = pd.DataFrame(
        predict_prices(net, example.subset(np.arange(len(example)) < 2)),
        columns=[f"h_{i + 1:02d}" for i in range(int(config["horizon"]))],
    )
    signature = infer_signature(sample, sample_out)
    requirements = [
        f"torch=={torch.__version__.split('+')[0]}",
        f"mlflow=={mlflow.__version__}",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "weights.pt"
        cfg_path = Path(tmp) / "config.json"
        torch.save(net.state_dict(), weights)
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        common = {
            # Models-from-code: no pickled objects inside the model (safer to load).
            "python_model": str(Path(__file__).with_name("pyfunc_entry.py")),
            "artifacts": {"weights": str(weights), "config": str(cfg_path)},
            "code_paths": [str(root / "src" / "finsight")],
            "signature": signature,
            "input_example": sample,
            "pip_requirements": requirements,
        }
        info = mlflow.pyfunc.log_model(name="model", **common)
        local = model_dir / "model"
        shutil.rmtree(local, ignore_errors=True)
        mlflow.pyfunc.save_model(path=str(local), **common)
    # Signature validation imports the model, leaving bytecode caches behind;
    # they are machine-specific noise in a DVC-tracked directory.
    for cache in list(local.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    return {"model_uri": info.model_uri, "model_id": getattr(info, "model_id", None)}


def _learning_curve_figure(histories: dict[str, list[dict[str, float]]], best: str) -> Any:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        width = 2.2 if name == best else 0.9
        ax.plot(
            epochs,
            [h["val_loss"] for h in hist],
            linewidth=width,
            label=f"{name} (val){'  <- selected' if name == best else ''}",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE (window-scaled)")
    ax.set_yscale("log")
    ax.set_title("Grid search: validation loss per configuration")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def run_training(params: dict[str, Any], prices_path: Path, root: Path) -> dict[str, Any]:
    """Execute the whole training procedure inside an MLflow run."""
    import mlflow  # noqa: PLC0415

    from finsight.models.tracking import lineage_tags  # noqa: PLC0415

    seed, threads = int(params["seed"]), int(params["torch_threads"])
    set_seeds(seed, threads)
    lookback, horizon = int(params["lookback"]), int(params["horizon"])
    series = load_price_series(prices_path)
    windows = make_windows(series, lookback, horizon)
    masks = split_masks(windows, int(params["val_origins"]), int(params["test_origins"]), horizon)
    train, val = windows.subset(masks["train"]), windows.subset(masks["val"])
    train_val = windows.subset(masks["train_val"])
    configs = grid_configs(params)
    model_dir = root / params["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    run_name = f"lstm-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.set_tags(
            {
                **lineage_tags(root, str(prices_path.relative_to(root)), seed),
                "stage": "train_lstm",
                "model_family": "lstm",
                "run_role": "parent",
            }
        )
        mlflow.log_params(
            {
                "lookback": lookback,
                "horizon": horizon,
                "purge": horizon,
                "val_origins": params["val_origins"],
                "test_origins": params["test_origins"],
                "batch_size": params["batch_size"],
                "max_epochs": params["max_epochs"],
                "patience": params["patience"],
                "dropout": params["dropout"],
                "seed": seed,
                "torch_threads": threads,
                "selection_metric": params["selection_metric"],
                "grid": json.dumps(params["grid"], sort_keys=True),
                "n_configs": len(configs),
                "tickers": ",".join(sorted(series)),
                "n_windows": len(windows),
                "n_train": len(train),
                "n_val": len(val),
                "n_train_val": len(train_val),
                "n_test": int(masks["test"].sum()),
            }
        )
        frame = pd.read_parquet(prices_path, columns=["ticker", "date", "price"])
        with warnings.catch_warnings():  # local path matches 2 equivalent source types
            warnings.simplefilter("ignore", UserWarning)
            dataset = mlflow.data.from_pandas(
                frame, source=str(prices_path), name="processed_prices"
            )
        mlflow.log_input(dataset, context="training")
        for extra in ("params.yaml", "dvc.lock", "reports/data_quality.json"):
            if (root / extra).exists():
                mlflow.log_artifact(str(root / extra), artifact_path="lineage")

        # ---- grid search: one child run per configuration ------------------
        results, histories = [], {}
        for cfg in configs:
            name = _config_name(cfg)
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({"run_role": "grid_candidate", "parent_run": parent.info.run_id})
                mlflow.log_params(cfg)
                fitted = fit(
                    cfg,
                    train,
                    val,
                    epochs=params["max_epochs"],
                    patience=params["patience"],
                    batch_size=params["batch_size"],
                    seed=seed,
                    on_epoch=lambda row: mlflow.log_metrics(
                        {k: v for k, v in row.items() if k != "epoch"}, step=int(row["epoch"])
                    ),
                )
                evaluation = compare_to_baselines(
                    predict_prices(fitted.net, val), val.x_raw, val.y_raw, val.tickers
                )
                pooled = evaluation["pooled"]
                metrics = {
                    "val_mape": pooled["model"]["mape"],
                    "val_rmse": pooled["model"]["rmse"],
                    "val_naive_mape": pooled["naive"]["mape"],
                    "val_skill_vs_naive": evaluation["skill_vs_naive"],
                    "best_epoch": fitted.best_epoch,
                    "epochs_run": len(fitted.history),
                }
                if pooled["model"]["directional_accuracy"] is not None:
                    metrics["val_directional_accuracy"] = pooled["model"]["directional_accuracy"]
                mlflow.log_metrics(metrics)
                results.append({"name": name, "config": cfg, **metrics})
                histories[name] = fitted.history
                log.info("grid candidate done", extra={"config": name, **metrics})

        best = min(results, key=lambda r: r[params["selection_metric"]])
        mlflow.log_params({f"best_{k}": v for k, v in best["config"].items()})
        mlflow.log_metrics(
            {
                "best_val_mape": best["val_mape"],
                "best_val_rmse": best["val_rmse"],
                "best_val_skill_vs_naive": best["val_skill_vs_naive"],
                "val_naive_mape": best["val_naive_mape"],
            }
        )
        mlflow.set_tag("selected_config", best["name"])
        mlflow.log_figure(
            _learning_curve_figure(histories, best["name"]), "plots/grid_val_loss.png"
        )
        mlflow.log_dict({"candidates": results}, "selection/grid_results.json")

        # ---- refit the winner on train+val (test stays untouched) ----------
        set_seeds(seed, threads)
        refit = fit(
            best["config"],
            train_val,
            None,
            epochs=max(best["best_epoch"], 1),
            patience=0,
            batch_size=params["batch_size"],
            seed=seed,
            on_epoch=lambda row: mlflow.log_metric(
                "refit_train_loss", row["train_loss"], step=int(row["epoch"])
            ),
        )
        packaged = package_model(refit.net, best["config"], val, root, model_dir)
        run_id, experiment_id = parent.info.run_id, parent.info.experiment_id

    duration = round(time.monotonic() - started, 1)
    run_info = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_name": params["experiment_name"],
        "tracking_uri": mlflow.get_tracking_uri(),
        "model_uri": packaged["model_uri"],
        "model_id": packaged["model_id"],
        "selected_config": best["config"],
        "refit_epochs": max(best["best_epoch"], 1),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": duration,
    }
    (model_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    best_history = histories[best["name"]]
    return {
        "run_info": run_info,
        "report": {
            "run_id": run_id,
            "selected": best["name"],
            "val_mape": round(best["val_mape"], 4),
            "val_naive_mape": round(best["val_naive_mape"], 4),
            "val_skill_vs_naive": round(best["val_skill_vs_naive"], 4),
            "best_epoch": best["best_epoch"],
            "n_configs": len(configs),
            "n_train": len(train),
            "n_val": len(val),
        },
        "learning_curve": best_history,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.models.lstm.train``)."""
    parser = argparse.ArgumentParser(description="Train the LSTM forecaster")
    parser.add_argument(
        "--seed", type=int, default=-1, help="override lstm.seed (-1 = params.yaml)"
    )
    parser.add_argument("--max-epochs", type=int, default=-1, help="override lstm.max_epochs")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("train_lstm", settings.log_level, settings.root / "logs")
    params = dict(settings.section("lstm"))
    if args.seed >= 0:
        params["seed"] = args.seed
    if args.max_epochs > 0:
        params["max_epochs"] = args.max_epochs
    prices_path = settings.root / settings.section("etl")["processed_prices_path"]
    try:
        from finsight.models.tracking import configure_tracking  # noqa: PLC0415

        with stage_run("train_lstm", settings) as run:
            configure_tracking(params["experiment_name"])
            outcome = run_training(params, prices_path, settings.root)
            report = settings.root / params["train_report"]
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(outcome["report"], indent=2) + "\n", encoding="utf-8")
            curve = settings.root / "reports" / "plots" / "lstm_learning_curve.csv"
            curve.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(outcome["learning_curve"]).to_csv(curve, index=False)
            run.metrics = outcome["report"]
    except (FinSightError, ValueError) as exc:
        print(f"train_lstm failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(outcome["report"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
