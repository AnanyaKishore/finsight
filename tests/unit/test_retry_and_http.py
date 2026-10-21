"""Unit tests for retry/backoff and the HTTP client error classification."""

from __future__ import annotations

import random

import pytest
import requests
import responses

from finsight.common.exceptions import ProviderError, QuotaExceededError
from finsight.common.http import HttpClient
from finsight.common.retry import backoff_delay, retry_with_backoff

URL = "https://api.example.test/data"


def test_backoff_delay_is_bounded_and_grows():
    rng = random.Random(0)
    for attempt in range(10):
        delay = backoff_delay(attempt, base=1.0, max_delay=8.0, rng=rng)
        assert 0 <= delay <= min(8.0, 2**attempt)


def test_retry_succeeds_after_transient_failures():
    sleeps: list[float] = []
    calls = {"n": 0}

    @retry_with_backoff(max_retries=3, base_delay=1, max_delay=4, sleep=sleeps.append)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_retry_gives_up_after_max_retries():
    calls = {"n": 0}

    @retry_with_backoff(max_retries=2, sleep=lambda _: None)
    def always_fails():
        calls["n"] += 1
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        always_fails()
    assert calls["n"] == 3  # first attempt + 2 retries


def test_non_retryable_error_is_raised_immediately():
    calls = {"n": 0}

    @retry_with_backoff(max_retries=5, is_retryable=lambda e: False, sleep=lambda _: None)
    def bad_key():
        calls["n"] += 1
        raise ValueError("invalid key")

    with pytest.raises(ValueError):
        bad_key()
    assert calls["n"] == 1


def _client(**kwargs):
    return HttpClient("test", max_retries=2, backoff_base_s=0, sleep=lambda _: None, **kwargs)


@responses.activate
def test_http_retries_5xx_then_succeeds():
    responses.get(URL, status=503)
    responses.get(URL, json={"ok": True}, status=200)
    client = _client()
    assert client.get_json(URL) == {"ok": True}
    assert client.calls_made == 2


@responses.activate
def test_http_4xx_is_not_retried():
    responses.get(URL, json={"error": "unauthorized"}, status=401)
    client = _client()
    with pytest.raises(ProviderError) as info:
        client.get_json(URL)
    assert info.value.retryable is False
    assert info.value.status == 401
    assert client.calls_made == 1


@responses.activate
def test_http_network_error_is_retryable():
    responses.get(URL, body=requests.ConnectionError("reset"))
    responses.get(URL, body=requests.ConnectionError("reset"))
    responses.get(URL, body=requests.ConnectionError("reset"))
    client = _client()
    with pytest.raises(ProviderError) as info:
        client.get_json(URL)
    assert info.value.retryable is True
    assert client.calls_made == 3


@responses.activate
def test_http_classifier_can_raise_quota_error_without_retry():
    responses.get(URL, json={"error": {"code": "usage_limit_reached"}}, status=429)

    def classify(status, body):
        if status == 429 and body["error"]["code"] == "usage_limit_reached":
            raise QuotaExceededError("quota", provider="test")

    client = _client(classify=classify)
    with pytest.raises(QuotaExceededError):
        client.get_json(URL)
    assert client.calls_made == 1
