"""Input attribution for the LSTM forecaster (explainability).

Question answered: *which part of the 100-day input window does the forecast
actually depend on?*

Method -- **occlusion**, model-agnostic and run through the packaged pyfunc
model (the artifact that is served, not a training-time copy):

1. Forecast every test window as-is.
2. For each block of ``block`` consecutive days, *occlude* it: replace those
   prices with a straight line between the neighbouring observed prices (so
   the level is kept but the block's ups and downs are removed). If the
   block touches the end of the window, it is flattened to the last price
   before it.
3. Re-forecast, and record how much the forecast moved (mean absolute change,
   % of price) and how much test MAPE changed.

Blocks are labelled by *lag*: lag 1 is the most recent input day (the forecast
origin), lag 100 the oldest. A model that is close to the naive forecast
should be most sensitive to the last block (it contains the anchor price) and
almost insensitive to the rest; large sensitivity to old lags would suggest
the model is using stale patterns.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _mape(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual) / np.abs(actual)) * 100)


def occlude(x_raw: np.ndarray, start: int, end: int) -> np.ndarray:
    """Copy of ``x_raw`` with columns ``[start, end)`` replaced by interpolation."""
    x = x_raw.copy()
    n_cols = x.shape[1]
    left = x[:, start - 1] if start > 0 else None
    right = x[:, end] if end < n_cols else None
    if left is None and right is None:
        raise ValueError("cannot occlude the whole window")
    if left is None:
        x[:, start:end] = right[:, None]
    elif right is None:
        x[:, start:end] = left[:, None]
    else:
        steps = np.arange(1, end - start + 1) / (end - start + 1)
        x[:, start:end] = left[:, None] + (right - left)[:, None] * steps[None, :]
    return x


def occlusion_attribution(
    predict: Callable[[np.ndarray], np.ndarray],
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    block: int,
) -> pd.DataFrame:
    """Forecast sensitivity to each ``block``-day slice of the input window.

    Args:
        predict: Maps raw input prices ``(N, L)`` to raw forecasts ``(N, H)``.
        x_raw: Test input windows.
        y_raw: Realised prices for those windows.
        block: Days per occluded block.

    Returns:
        One row per block (most recent first) with ``lag_from``/``lag_to``,
        ``mean_abs_change_pct`` and ``mape_increase`` (percentage points).
    """
    lookback = x_raw.shape[1]
    base = predict(x_raw)
    base_mape = _mape(base, y_raw)
    rows = []
    for start in range(0, lookback, block):
        end = min(start + block, lookback)
        pred = predict(occlude(x_raw, start, end))
        rows.append(
            {
                "lag_from": lookback - end + 1,
                "lag_to": lookback - start,
                "block": f"lag {lookback - end + 1}-{lookback - start}",
                "mean_abs_change_pct": round(float(np.mean(np.abs(pred - base) / base) * 100), 6),
                "mape_increase": round(_mape(pred, y_raw) - base_mape, 6),
            }
        )
    return pd.DataFrame(rows).sort_values("lag_from").reset_index(drop=True)


def attribution_figure(table: pd.DataFrame) -> object:
    """Horizontal bar chart of forecast sensitivity per lag block."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(table["block"], table["mean_abs_change_pct"], color="#2f6fb3")
    ax.invert_yaxis()
    ax.set_xlabel("mean |change in forecast| when the block is occluded (% of price)")
    ax.set_title("LSTM input attribution (occlusion, test windows)")
    fig.tight_layout()
    return fig
