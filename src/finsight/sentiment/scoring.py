"""Core sentiment scoring shared by evaluation, batch scoring and serving.

Score definition (proposal: "converted to a -1 to +1 sentiment score")
-----------------------------------------------------------------------
DistilBERT-SST-2 outputs two softmax probabilities, ``p_neg + p_pos = 1``.
The score is

    score = p_pos - p_neg            (in [-1, +1])

which equals the *signed confidence* of the predicted class rescaled to the
full range: ``+1`` = certainly positive, ``-1`` = certainly negative, ``0`` =
undecided. Unlike "label sign x max probability" (which can only take values
in [-1, -0.5] U [0.5, 1]), it is continuous through zero, so averaging scores
over a week is meaningful.

Neutral band
------------
SST-2 is a *binary* movie-review dataset, but much financial news is factual
("Apple reports Q3 results on Thursday"). An article is labelled ``neutral``
when ``|score| < neutral_band``; the band is calibrated on labelled financial
sentences (``evaluate_sentiment``), never guessed.

Two- and three-label models
---------------------------
DistilBERT-SST-2 has labels {negative, positive}; FinBERT (Day 5 challenger)
has {positive, negative, neutral}. The score is ``p_pos - p_neg`` for both,
so it stays comparable across models; ``confidence`` is the probability of
the most likely class (0.5-1 for two labels, 1/3-1 for three).

The same :class:`SentimentScorer` and :func:`labels_for` run in evaluation,
batch scoring and the served MLflow model, so there is no training/serving skew.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

POSITIVE, NEGATIVE, NEUTRAL = "positive", "negative", "neutral"
CLASSES = (NEGATIVE, NEUTRAL, POSITIVE)


def scores_from_probabilities(p_neg: np.ndarray, p_pos: np.ndarray) -> np.ndarray:
    """``p_pos - p_neg``, clipped to [-1, 1] against floating-point overshoot."""
    return np.clip(np.asarray(p_pos, dtype="float64") - np.asarray(p_neg, dtype="float64"), -1, 1)


def labels_for(scores: np.ndarray, neutral_band: float) -> np.ndarray:
    """Map scores to ``positive`` / ``negative`` / ``neutral`` with a symmetric band.

    ``neutral_band = 0`` gives the plain binary decision (score 0 -> neutral).
    """
    scores = np.asarray(scores, dtype="float64")
    band = float(neutral_band)
    out = np.full(scores.shape, NEUTRAL, dtype=object)
    out[(scores >= band) & (scores > 0)] = POSITIVE
    out[(scores <= -band) & (scores < 0)] = NEGATIVE
    return out


def model_version(base_revision: str, neutral_band: float, max_length: int) -> str:
    """Fingerprint of everything that changes a score.

    Stored next to every cached score; ``score_sentiment`` re-scores an article
    whenever the fingerprint changes (new base model, new band, new truncation).
    """
    return f"{base_revision[:12]}-band{neutral_band:.4f}-len{int(max_length)}"


def clean_texts(texts: Sequence[Any]) -> list[str]:
    """Replace missing values with empty strings and collapse whitespace."""
    out = []
    for text in texts:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            out.append("")
        else:
            out.append(" ".join(str(text).split()))
    return out


class SentimentScorer:
    """Batched CPU inference with a local Hugging Face sequence classifier.

    Texts are sorted by length before batching, so each batch pads to a similar
    length (roughly 2-3x faster than random batches on CPU); results are
    returned in the caller's order.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        threads: int | None = None,
    ) -> None:
        """Load tokenizer and model from ``model_dir`` (never from the network).

        Raises:
            ValueError: If the model's labels are not a positive/negative pair.
        """
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if threads:
            torch.set_num_threads(int(threads))
        self.max_length, self.batch_size = int(max_length), int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True
        ).eval()
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        if set(labels) not in ({POSITIVE, NEGATIVE}, {POSITIVE, NEGATIVE, NEUTRAL}):
            raise ValueError(
                f"expected POSITIVE/NEGATIVE[/NEUTRAL] labels, model has {sorted(labels)}"
            )
        self.pos_index, self.neg_index = labels[POSITIVE], labels[NEGATIVE]
        self.n_labels = len(labels)

    def probability_matrix(self, texts: Sequence[Any]) -> np.ndarray:
        """Class probabilities ``(N, n_labels)`` in input order (model label order)."""
        import torch  # noqa: PLC0415

        cleaned = clean_texts(texts)
        out = np.zeros((len(cleaned), self.n_labels), dtype="float64")
        order = np.argsort([len(t) for t in cleaned], kind="stable")
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = self.tokenizer(
                [cleaned[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                out[idx] = torch.softmax(self.model(**batch).logits.float(), dim=-1).numpy()
        return out

    def probabilities(self, texts: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        """``(p_neg, p_pos)`` for every text, in input order."""
        probs = self.probability_matrix(texts)
        return probs[:, self.neg_index], probs[:, self.pos_index]

    def score_and_confidence(self, texts: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Scores in [-1, 1] and top-class probabilities, both rounded to 6 decimals."""
        probs = self.probability_matrix(texts)
        scores = scores_from_probabilities(probs[:, self.neg_index], probs[:, self.pos_index])
        return np.round(scores, 6), np.round(probs.max(axis=1), 6)

    def scores(self, texts: Sequence[Any]) -> np.ndarray:
        """Sentiment scores in [-1, 1], rounded to 6 decimals (stable across batchings)."""
        return self.score_and_confidence(texts)[0]
