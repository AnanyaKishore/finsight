"""Residual LSTM forecaster.

``forecast = last_input_price + correction(LSTM(window))`` (all in window-scaled
units). The correction head is zero-initialised, so an *untrained* model is
exactly the naive "price stays the same" forecast, and training can only move
it away from naive where the data supports it. Compared with the baseline
repo's recursive one-step LSTM (which fed its own predictions back in and
compounded errors over 14 steps), this predicts all 14 steps in one pass.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Direct multi-horizon LSTM with a residual connection to the last input."""

    def __init__(
        self,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        input_size: int = 1,
    ) -> None:
        """Build the network.

        Args:
            horizon: Number of future steps predicted at once.
            hidden_size: LSTM hidden units.
            num_layers: Stacked LSTM layers.
            dropout: Dropout between stacked layers (ignored for one layer).
            input_size: Input channels; channel 0 is always the scaled price
                (the residual anchor). Extra channels are only used by the
                feature experiment (``feature_experiment.py``).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, lookback, channels)`` scaled inputs to ``(batch, horizon)``."""
        out, _ = self.lstm(x)
        correction = self.head(out[:, -1, :])
        return x[:, -1, :1] + correction
