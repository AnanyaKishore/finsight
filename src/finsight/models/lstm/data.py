"""Windowing, scaling and splitting for the LSTM forecaster.

Design (and the baseline bugs it avoids -- see DESIGN_DECISIONS.md, Part B):

* **Per-window MinMax scaling.** Each 100-day input window is scaled with the
  min/max of *that window only*; the 14 targets are scaled with the same
  factors. The baseline fitted one scaler on the whole series, leaking future
  prices into the past. Per-window scaling also puts all 10 tickers on the
  same [0, 1] scale, so one *global* model can learn from all of them
  (10x more windows than a model per ticker).
* **One shared transform.** :func:`scale_rows` is used by training *and* by
  the served pyfunc model, so training/serving skew is impossible by design.
* **Purged chronological split.** Per ticker, windows are ordered by their
  forecast origin. The last ``test_origins`` windows form the test set, the
  ``val_origins`` before them the validation set, with a gap of ``horizon``
  windows between sets so no training target overlaps a validation target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6


def scale_rows(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MinMax-scale each row of a 2-D array of prices to [0, 1].

    Returns:
        ``(scaled, lo, hi)`` where ``lo``/``hi`` are the per-row min/max used,
        so :func:`unscale_rows` can map predictions back to prices.
    """
    lo = raw.min(axis=1)
    hi = raw.max(axis=1)
    span = np.maximum(hi - lo, EPS * np.maximum(np.abs(hi), 1.0))  # flat window guard
    scaled = (raw - lo[:, None]) / span[:, None]
    return scaled, lo, lo + span


def unscale_rows(scaled: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Inverse of :func:`scale_rows` (works for any number of columns)."""
    return lo[:, None] + scaled * (hi - lo)[:, None]


@dataclass
class WindowSet:
    """Aligned arrays describing N forecast windows."""

    x_raw: np.ndarray  # (N, lookback) input prices
    y_raw: np.ndarray  # (N, horizon) target prices
    tickers: np.ndarray  # (N,) symbol
    origins: np.ndarray  # (N,) date of the last input day
    position: np.ndarray  # (N,) origin index within its ticker (0 = oldest)
    n_per_ticker: dict[str, int]
    # Optional extra input channels (N, lookback, k), e.g. rolling volatility in
    # the feature experiment; each is MinMax-scaled per window like the price.
    x_extra: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.x_raw)

    def subset(self, mask: np.ndarray) -> WindowSet:
        """Rows selected by a boolean mask."""
        return WindowSet(
            self.x_raw[mask],
            self.y_raw[mask],
            self.tickers[mask],
            self.origins[mask],
            self.position[mask],
            self.n_per_ticker,
            None if self.x_extra is None else self.x_extra[mask],
        )

    def scaled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Model inputs/targets: ``(x[N, L, C], y[N, H], lo, hi)`` in window units.

        ``C = 1`` (price only) unless :attr:`x_extra` is set; ``lo``/``hi`` are
        always the *price* scaling factors, used to map forecasts back to USD.
        """
        x, lo, hi = scale_rows(self.x_raw)
        y = (self.y_raw - lo[:, None]) / (hi - lo)[:, None]
        channels = [x[:, :, None]]
        if self.x_extra is not None:
            for k in range(self.x_extra.shape[2]):
                channels.append(scale_rows(self.x_extra[:, :, k])[0][:, :, None])
        return np.concatenate(channels, axis=2).astype(np.float32), y.astype(np.float32), lo, hi

    @property
    def last_price(self) -> np.ndarray:
        """Last observed input price per window (the naive forecast)."""
        return self.x_raw[:, -1]


def load_price_series(path: Path) -> dict[str, pd.Series]:
    """Read the Spark ETL output into ``{ticker: price series indexed by date}``."""
    frame = pd.read_parquet(path, columns=["ticker", "date", "price"])
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    series = {}
    for ticker, group in frame.sort_values(["ticker", "date"]).groupby("ticker"):
        series[ticker] = group.set_index("date")["price"].astype("float64")
    return series


def make_windows(series: dict[str, pd.Series], lookback: int, horizon: int) -> WindowSet:
    """All complete (lookback -> horizon) windows for every ticker.

    A series of length ``n`` yields ``n - lookback - horizon + 1`` windows.
    """
    xs, ys, tickers, origins, positions, counts = [], [], [], [], [], {}
    for ticker in sorted(series):
        values = series[ticker].to_numpy()
        dates = series[ticker].index.to_numpy()
        n_windows = len(values) - lookback - horizon + 1
        counts[ticker] = max(n_windows, 0)
        for k in range(max(n_windows, 0)):
            xs.append(values[k : k + lookback])
            ys.append(values[k + lookback : k + lookback + horizon])
            tickers.append(ticker)
            origins.append(dates[k + lookback - 1])
            positions.append(k)
    if not xs:
        raise ValueError(f"no complete windows: every series is shorter than {lookback + horizon}")
    return WindowSet(
        np.asarray(xs, dtype="float64"),
        np.asarray(ys, dtype="float64"),
        np.asarray(tickers),
        np.asarray(origins),
        np.asarray(positions),
        counts,
    )


def split_masks(ws: WindowSet, val_origins: int, test_origins: int, purge: int) -> dict:
    """Boolean masks for ``train``, ``val``, ``test`` and ``train_val`` (refit).

    Per ticker with ``n`` windows (positions ``0..n-1``)::

        [ train ][purge][ val ][purge][ test ]

    ``train_val`` = everything before the second purge gap (used to refit the
    selected configuration before the final, untouched test evaluation).

    Raises:
        ValueError: If a ticker has too few windows for this split.
    """
    n = np.array([ws.n_per_ticker[t] for t in ws.tickers])
    test_start = n - test_origins
    val_end = test_start - purge
    val_start = val_end - val_origins
    train_end = val_start - purge
    short = sorted(
        {t for t, k in ws.n_per_ticker.items() if k - 2 * purge - val_origins - test_origins < 1}
    )
    if short:
        raise ValueError(f"too few windows for the split in: {short}")
    pos = ws.position
    return {
        "train": pos < train_end,
        "val": (pos >= val_start) & (pos < val_end),
        "test": pos >= test_start,
        "train_val": pos < val_end,
    }


def latest_window(series: pd.Series, lookback: int) -> np.ndarray:
    """The most recent ``lookback`` prices (the input for a live forecast)."""
    if len(series) < lookback:
        raise ValueError(f"need {lookback} prices, got {len(series)}")
    return series.to_numpy()[-lookback:].astype("float64")
