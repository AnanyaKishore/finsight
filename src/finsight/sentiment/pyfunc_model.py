"""MLflow ``pyfunc`` wrapper: the deployable form of the sentiment model.

Contract (recorded as the MLflow model *signature*):

* input  -- DataFrame with one string column ``text`` (headline + summary).
* output -- DataFrame with ``label`` (positive / negative / neutral),
  ``confidence`` (probability of the most likely class) and
  ``score`` (``p_pos - p_neg``, -1..+1).

The calibrated neutral band, truncation length and batch size are stored in
``config.json`` inside the model, so a served model applies exactly the
decision rule that was evaluated (no training/serving skew).
"""

from __future__ import annotations

import json
from typing import Any

import mlflow.pyfunc
import pandas as pd

from finsight.sentiment.scoring import SentimentScorer, labels_for

OUTPUT_COLUMNS = ["label", "confidence", "score"]


class SentimentPyfunc(mlflow.pyfunc.PythonModel):
    """Text in, label + confidence + score out."""

    def load_context(self, context: Any) -> None:
        """Load the decision config and the Hugging Face model (CPU, eval mode)."""
        with open(context.artifacts["config"], encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.scorer = SentimentScorer(
            context.artifacts["hf_model"],
            max_length=int(self.config["max_length"]),
            batch_size=int(self.config["batch_size"]),
            threads=int(self.config.get("serving_threads", 2)),
        )

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        """Score every row of ``model_input['text']``.

        Raises:
            ValueError: If the ``text`` column is missing.
        """
        if "text" not in model_input.columns:
            raise ValueError("input must have a 'text' column")
        scores, confidence = self.scorer.score_and_confidence(model_input["text"].tolist())
        return pd.DataFrame(
            {
                "label": labels_for(scores, float(self.config["neutral_band"])).astype(str),
                "confidence": confidence,
                "score": scores,
            },
            index=model_input.index,
        )
