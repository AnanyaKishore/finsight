"""Forecast metrics in price space, with naive and drift baselines.

* **RMSE / MAE** (USD) -- per ticker meaningful; pooled across tickers it is
  dominated by expensive stocks, so model selection uses MAPE.
* **MAPE** (%) -- scale-free, comparable across tickers; the selection metric.
* **Directional accuracy** -- share of (window, horizon) cells where the
  forecast moves in the same direction as reality relative to the last price.
  The naive forecast never moves, so its directional accuracy is undefined;
  the reference is 50% (coin flip) and the share of "up" moves.
* **Skill vs naive** -- ``1 - MAPE_model / MAPE_naive``: > 0 means better than
  "tomorrow equals today", the standard bar for price forecasts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def naive_forecast(x_raw: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed price for every horizon step."""
    return np.repeat(x_raw[:, -1:], horizon, axis=1)


def drift_forecast(x_raw: np.ndarray, horizon: int) -> np.ndarray:
    """Extend the average daily change over the input window."""
    slope = (x_raw[:, -1] - x_raw[:, 0]) / (x_raw.shape[1] - 1)
    steps = np.arange(1, horizon + 1)
    return x_raw[:, -1:] + slope[:, None] * steps[None, :]


def point_metrics(pred: np.ndarray, actual: np.ndarray, last: np.ndarray) -> dict[str, Any]:
    """RMSE, MAE, MAPE, directional accuracy for aligned ``(N, H)`` arrays."""
    err = pred - actual
    true_move = np.sign(actual - last[:, None])
    pred_move = np.sign(pred - last[:, None])
    moving = (true_move != 0) & (pred_move != 0)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.abs(actual)) * 100),
        "directional_accuracy": (
            float(np.mean(pred_move[moving] == true_move[moving])) if moving.any() else None
        ),
        "up_share": float(np.mean(true_move > 0)),
    }


def compare_to_baselines(
    pred: np.ndarray, x_raw: np.ndarray, y_raw: np.ndarray, tickers: np.ndarray
) -> dict[str, Any]:
    """Full evaluation: pooled model/naive/drift, skill, per-ticker and per-horizon."""
    horizon = y_raw.shape[1]
    last = x_raw[:, -1]
    forecasts = {
        "model": pred,
        "naive": naive_forecast(x_raw, horizon),
        "drift": drift_forecast(x_raw, horizon),
    }
    pooled = {name: point_metrics(f, y_raw, last) for name, f in forecasts.items()}
    per_ticker = {}
    for ticker in sorted(set(tickers.tolist())):
        mask = tickers == ticker
        per_ticker[ticker] = {
            name: point_metrics(f[mask], y_raw[mask], last[mask]) for name, f in forecasts.items()
        }
    per_horizon = []
    for h in range(horizon):
        row = {"horizon": h + 1}
        for name, f in forecasts.items():
            row[f"{name}_mape"] = float(np.mean(np.abs(f[:, h] - y_raw[:, h]) / y_raw[:, h]) * 100)
        per_horizon.append(row)
    naive_mape = pooled["naive"]["mape"]
    return {
        "pooled": pooled,
        "skill_vs_naive": 1 - pooled["model"]["mape"] / naive_mape if naive_mape > 0 else None,
        "beats_naive": pooled["model"]["mape"] < naive_mape,
        "per_ticker": per_ticker,
        "per_horizon": per_horizon,
        "windows": int(len(y_raw)),
    }
