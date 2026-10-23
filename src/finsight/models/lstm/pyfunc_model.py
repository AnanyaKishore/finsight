"""MLflow ``pyfunc`` wrapper: the deployable form of the forecaster.

Contract (recorded as the MLflow model *signature*):

* input  -- DataFrame with one row per series and columns ``p_000 .. p_099``:
  the last 100 prices, oldest first, in USD.
* output -- DataFrame with columns ``h_01 .. h_14``: forecast prices in USD.

Scaling happens *inside* the model with the same :func:`scale_rows` used in
training, so callers (``mlflow models serve`` on Day 5, the FastAPI backend)
send raw prices and receive raw prices. Packaging the preprocessing with the
weights removes training/serving skew.
"""

from __future__ import annotations

import json
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from finsight.models.lstm.data import scale_rows, unscale_rows


def input_columns(lookback: int) -> list[str]:
    """Signature column names for the input window."""
    return [f"p_{i:03d}" for i in range(lookback)]


def output_columns(horizon: int) -> list[str]:
    """Signature column names for the forecast."""
    return [f"h_{i + 1:02d}" for i in range(horizon)]


def build_network(config: dict[str, Any]) -> Any:
    """Instantiate the network described by a saved config."""
    from finsight.models.lstm.network import LSTMForecaster  # noqa: PLC0415

    return LSTMForecaster(
        horizon=int(config["horizon"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        input_size=int(config.get("input_size", 1)),
    )


class LSTMPricePyfunc(mlflow.pyfunc.PythonModel):
    """Raw prices in, raw 14-day forecast out."""

    def load_context(self, context: Any) -> None:
        """Load config + weights from the model's artifacts (CPU, eval mode)."""
        import torch  # noqa: PLC0415

        with open(context.artifacts["config"], encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.net = build_network(self.config)
        state = torch.load(context.artifacts["weights"], map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        self.net.eval()
        torch.set_num_threads(1)  # serving: many small requests, not one big batch

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        """Forecast every input row.

        Raises:
            ValueError: Missing columns, or non-finite / non-positive prices.
        """
        import torch  # noqa: PLC0415

        lookback, horizon = int(self.config["lookback"]), int(self.config["horizon"])
        columns = input_columns(lookback)
        missing = [c for c in columns if c not in model_input.columns]
        if missing:
            raise ValueError(f"input is missing {len(missing)} price columns, e.g. {missing[:3]}")
        raw = model_input[columns].to_numpy(dtype="float64")
        if not np.isfinite(raw).all() or (raw <= 0).any():
            raise ValueError("prices must be finite and > 0")
        scaled, lo, hi = scale_rows(raw)
        with torch.no_grad():
            out = self.net(torch.from_numpy(scaled[:, :, None].astype(np.float32))).numpy()
        forecast = unscale_rows(out.astype("float64"), lo, hi)
        return pd.DataFrame(forecast, columns=output_columns(horizon), index=model_input.index)
