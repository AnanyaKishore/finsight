"""Synthetic, deterministic datasets shaped exactly like the export snapshot.

Used by the export / validate / ETL tests and by the sandbox pipeline drill.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from finsight.common.timeutils import is_trading_day


def make_prices(
    tickers=("AAPL", "MSFT"),
    start=date(2026, 3, 2),
    end=date(2026, 10, 20),
    seed=7,
    drop: dict[str, list[date]] | None = None,
    source="marketstack",
) -> pd.DataFrame:
    """Geometric-random-walk OHLCV bars on NYSE trading days."""
    rng = np.random.default_rng(seed)
    rows = []
    for t_index, ticker in enumerate(tickers):
        price = 100.0 + 50 * t_index
        current = start
        while current <= end:
            if is_trading_day(current) and current not in (drop or {}).get(ticker, []):
                ret = rng.normal(0.0005, 0.015)
                open_ = price
                price = price * float(np.exp(ret))
                high = max(open_, price) * (1 + abs(rng.normal(0, 0.003)))
                low = min(open_, price) * (1 - abs(rng.normal(0, 0.003)))
                rows.append(
                    {
                        "ticker": ticker,
                        "date": current,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": price,
                        "adj_close": price,
                        "volume": float(rng.integers(1_000_000, 5_000_000)),
                        "source": source,
                    }
                )
            current += timedelta(days=1)
    frame = pd.DataFrame(rows)
    frame["ticker"] = frame["ticker"].astype("string")
    frame["source"] = frame["source"].astype("string")
    return frame


def make_news(
    tickers=("AAPL", "MSFT"), start=date(2026, 6, 1), weeks=10, per_week=8
) -> pd.DataFrame:
    """``per_week`` articles per ticker per week, spread over weekdays."""
    rows = []
    for ticker in tickers:
        for w in range(weeks):
            for i in range(per_week):
                published = datetime(
                    start.year, start.month, start.day, 14, tzinfo=UTC
                ) + timedelta(weeks=w, days=i % 5, minutes=i)
                rows.append(
                    {
                        "ticker": ticker,
                        "news_id": f"{ticker}-{w}-{i}",
                        "published_at": published,
                        "headline": f"{ticker} headline {w}-{i}",
                        "summary": "Shares moved after the report.",
                        "text": f"{ticker} headline {w}-{i}. Shares moved after the report.",
                        "source": ["Reuters", "Yahoo", "MarketWatch"][i % 3],
                        "url": f"https://example.test/{ticker}/{w}/{i}",
                        "provider": "finnhub",
                        "category": "company",
                    }
                )
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column != "published_at":
            frame[column] = frame[column].astype("string")
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    return frame


# --------------------------------------------------------------------------- #
# Day 4: sentiment
# --------------------------------------------------------------------------- #
TINY_VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "company", "shares", "rose",
    "fell", "profit", "loss", "growth", "decline", "reports", "results", "on", "monday",
    "strong", "weak", "sales", "beat", "missed", "expectations", "headline", ".", ",",
    "aapl", "msft", "moved", "after", "report",
]  # fmt: skip


def make_tiny_sentiment_model(path, seed: int = 0, labels=("NEGATIVE", "POSITIVE")):
    """A randomly initialised 1-layer DistilBERT laid out like the Hub repo.

    Same files as ``distilbert-base-uncased-finetuned-sst-2-english``
    (``config.json``, ``model.safetensors``, ``tokenizer_config.json``,
    ``vocab.txt``), so the real loading code is exercised offline in ~1 s.
    Its predictions are meaningless; tests check mechanics, not accuracy.
    """
    import json

    import torch
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(TINY_VOCAB) + "\n", encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"do_lower_case": True, "model_max_length": 64}), encoding="utf-8"
    )
    torch.manual_seed(seed)
    config = DistilBertConfig(
        vocab_size=len(TINY_VOCAB),
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        max_position_embeddings=64,
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    )
    DistilBertForSequenceClassification(config).save_pretrained(path)
    (path / "tokenizer.json").unlink(missing_ok=True)  # the Hub repo ships vocab.txt only
    return path


def make_phrasebank(n_neutral=60, n_positive=25, n_negative=15) -> pd.DataFrame:
    """Labelled sentences shaped like ``data/external/financial_phrasebank.csv``."""
    rows = []
    for label, n, word in (
        ("neutral", n_neutral, "reports results on monday"),
        ("positive", n_positive, "profit rose strong growth"),
        ("negative", n_negative, "loss fell weak decline"),
    ):
        for i in range(n):
            rows.append({"sentence": f"The company {word} {i}.", "label": label})
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", range(len(frame)))
    frame["agreement"] = "AllAgree"
    return frame
