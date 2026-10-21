"""Unit tests for the MarketStack, Finnhub and yfinance adapters (HTTP mocked)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
import responses

from finsight.common.exceptions import ProviderError, QuotaExceededError
from finsight.common.http import HttpClient
from finsight.ingestion.providers.finnhub import (
    FinnhubNewsProvider,
    parse_news_item,
    sample_evenly,
)
from finsight.ingestion.providers.marketstack import (
    MarketStackProvider,
    classify_marketstack,
    parse_eod_row,
)
from finsight.ingestion.providers.yfinance_provider import YFinanceProvider

MS_BASE = "https://api.marketstack.test/v2"
FH_BASE = "https://finnhub.test/api/v1"


def _ms_row(day: str, close: float = 100.0):
    return {
        "date": f"{day}T00:00:00+0000",
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "volume": 12345,
        "adj_close": close - 0.5,
        "symbol": "AAPL",
    }


def _marketstack():
    http = HttpClient(
        "marketstack",
        classify=classify_marketstack,
        max_retries=1,
        backoff_base_s=0,
        sleep=lambda _: None,
    )
    return MarketStackProvider("secret-key", MS_BASE, http)


# ------------------------------------------------------------------ MarketStack
def test_parse_eod_row_valid_and_invalid():
    bar = parse_eod_row(_ms_row("2026-10-20", 250.0), "AAPL")
    assert bar is not None
    assert bar.date == datetime(2026, 10, 20, tzinfo=UTC)
    assert (bar.open, bar.close, bar.adj_close, bar.source) == (249.0, 250.0, 249.5, "marketstack")
    assert parse_eod_row({"date": "2026-10-20", "close": None}, "AAPL") is None
    assert parse_eod_row({"date": "garbage", "close": 1}, "AAPL") is None


@responses.activate
def test_marketstack_paginates_and_sorts():
    url = f"{MS_BASE}/eod"
    responses.get(
        url,
        json={
            "pagination": {"count": 2, "total": 3},
            "data": [_ms_row("2026-10-19"), _ms_row("2026-10-16")],
        },
    )
    responses.get(
        url, json={"pagination": {"count": 1, "total": 3}, "data": [_ms_row("2026-10-20")]}
    )
    bars = _marketstack().fetch_eod("AAPL", date(2026, 10, 16), date(2026, 10, 20))
    assert [b.date.day for b in bars] == [16, 19, 20]
    assert "access_key=secret-key" in responses.calls[0].request.url
    assert "offset=2" in responses.calls[1].request.url


@responses.activate
def test_marketstack_usage_limit_raises_quota_error():
    responses.get(
        f"{MS_BASE}/eod",
        status=429,
        json={"error": {"code": "usage_limit_reached", "message": "limit"}},
    )
    with pytest.raises(QuotaExceededError):
        _marketstack().fetch_eod("AAPL", date(2026, 10, 1), date(2026, 10, 20))
    assert len(responses.calls) == 1  # never retried


@responses.activate
def test_marketstack_invalid_key_is_fatal():
    responses.get(
        f"{MS_BASE}/eod",
        status=401,
        json={"error": {"code": "invalid_access_key", "message": "bad key"}},
    )
    with pytest.raises(ProviderError) as info:
        _marketstack().fetch_eod("AAPL", date(2026, 10, 1), date(2026, 10, 20))
    assert info.value.retryable is False


def test_marketstack_cost_is_one_call_for_a_year():
    assert _marketstack().cost("AAPL", date(2025, 10, 21), date(2026, 10, 21)) == 1


# ---------------------------------------------------------------------- Finnhub
def _fh_item(i: int, ts: int, url: str | None = None):
    return {
        "id": i,
        "datetime": ts,
        "headline": f"News {i}",
        "summary": "s",
        "source": "Reuters",
        "url": url or f"https://n/{i}",
        "related": "AAPL",
    }


def test_parse_news_item_rejects_missing_fields():
    assert parse_news_item({"headline": "", "datetime": 1}, "AAPL") is None
    assert parse_news_item({"headline": "x", "datetime": 0}, "AAPL") is None
    article = parse_news_item(_fh_item(7, 1_760_000_000), "AAPL")
    assert article is not None and article.news_id == "7"
    assert article.model_text.startswith("News 7. s")


def test_parse_news_item_hashes_url_when_id_missing():
    article = parse_news_item({**_fh_item(0, 1_760_000_000), "id": 0}, "AAPL")
    assert article is not None and article.news_id.startswith("h")


def test_sample_evenly_bounds_and_spreads():
    base = parse_news_item(_fh_item(1, 1_760_000_000), "AAPL")
    items = [parse_news_item(_fh_item(i, 1_760_000_000 + i * 3600), "AAPL") for i in range(1, 21)]
    kept = sample_evenly(items, 5)
    assert len(kept) == 5
    assert kept[0].published_at == items[0].published_at
    assert base is not None
    assert sample_evenly(items[:3], 5) == items[:3]


@responses.activate
def test_finnhub_fetch_dedups_and_caps():
    payload = [_fh_item(i, 1_760_000_000 + i) for i in range(1, 9)] + [_fh_item(1, 1_760_000_001)]
    responses.get(f"{FH_BASE}/company-news", json=payload)
    http = HttpClient("finnhub", max_retries=0, sleep=lambda _: None)
    provider = FinnhubNewsProvider("tok", FH_BASE, http, max_articles=4)
    articles = provider.fetch_news("AAPL", date(2026, 10, 1), date(2026, 10, 7))
    assert len(articles) == 4
    assert len({a.news_id for a in articles}) == 4
    assert "symbol=AAPL" in responses.calls[0].request.url


@responses.activate
def test_finnhub_error_payload_raises():
    responses.get(f"{FH_BASE}/company-news", json={"error": "Invalid API key"})
    http = HttpClient("finnhub", max_retries=0, sleep=lambda _: None)
    with pytest.raises(ProviderError):
        FinnhubNewsProvider("tok", FH_BASE, http).fetch_news(
            "AAPL", date(2026, 10, 1), date(2026, 10, 7)
        )


# --------------------------------------------------------------------- yfinance
def test_yfinance_provider_parses_frame_and_skips_nan():
    index = pd.to_datetime(["2026-10-19", "2026-10-20"])
    frame = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.2, float("nan")],
            "Adj Close": [1.1, 2.1],
            "Volume": [10, 20],
        },
        index=index,
    )
    captured = {}

    def fake_download(ticker, **kwargs):
        captured.update(kwargs)
        return frame

    bars = YFinanceProvider(downloader=fake_download).fetch_eod(
        "AAPL", date(2026, 10, 19), date(2026, 10, 20)
    )
    assert len(bars) == 1 and bars[0].source == "yfinance" and bars[0].adj_close == 1.1
    assert captured["end"] == "2026-10-21"  # yfinance end date is exclusive


def test_yfinance_provider_wraps_errors():
    def broken(*_, **__):
        raise RuntimeError("rate limited")

    with pytest.raises(ProviderError):
        YFinanceProvider(downloader=broken).fetch_eod("AAPL", date(2026, 10, 1), date(2026, 10, 2))
