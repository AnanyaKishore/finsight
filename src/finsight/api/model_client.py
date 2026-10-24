"""HTTP client for MLflow model servers (``mlflow models serve``).

Protocol (MLflow scoring server): ``POST /invocations`` with
``{"dataframe_split": {"columns": [...], "data": [[...], ...]}}``; the response
is ``{"predictions": [...]}`` (one record per input row for DataFrame outputs).
``GET /ping`` returns 200 when the model is loaded.

Resilience: per-call timeout, retries with exponential backoff and full jitter
on connection errors / 5xx / 429 (``finsight.common.retry``). Anything else, or
running out of retries, raises :class:`ModelUnavailable`, which the routes turn
into graceful degradation (stale forecast) or a 503.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import requests

from finsight.common.exceptions import FinSightError
from finsight.common.logging import get_logger
from finsight.common.retry import retry_with_backoff

log = get_logger(__name__)


class ModelUnavailable(FinSightError):  # noqa: N818 -- reads naturally at call sites
    """The model server could not produce a prediction."""


class _RetryableError(Exception):
    """Internal: a transient failure worth retrying."""


class ModelClient:
    """Calls one MLflow model server."""

    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        timeout_s: float = 10.0,
        retries: int = 2,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        observe: Callable[[str, str, float], None] | None = None,
    ) -> None:
        """Configure the client.

        Args:
            name: Model family label used in metrics and logs (``lstm``, ``sentiment``).
            base_url: Server root, e.g. ``http://lstm-serve:5001``.
            timeout_s: Per-request timeout.
            retries: Retries after the first attempt (transient errors only).
            session: Injected HTTP session (tests).
            sleep: Injected sleep for the backoff (tests pass a no-op).
            observe: Callback ``(model, outcome, seconds)`` for Prometheus metrics.
        """
        self.name, self.base_url = name, base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.observe = observe or (lambda *_: None)
        self._post = retry_with_backoff(
            max_retries=retries,
            base_delay=0.2,
            max_delay=2.0,
            is_retryable=lambda exc: isinstance(exc, _RetryableError),
            sleep=sleep,
        )(self._post_once)

    def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/invocations", json=payload, timeout=self.timeout_s
            )
        except requests.RequestException as exc:
            raise _RetryableError(f"{exc.__class__.__name__}: {exc}") from exc
        if response.status_code >= 500 or response.status_code == 429:
            raise _RetryableError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise ModelUnavailable(
                f"{self.name} model rejected the request: HTTP {response.status_code} "
                f"{response.text[:200]}"
            )
        return response.json()

    def predict(self, columns: list[str], rows: list[list[Any]]) -> list[dict[str, Any]]:
        """Score ``rows`` and return one record per row.

        Raises:
            ModelUnavailable: Server down, timing out, or rejecting the input.
        """
        started = time.perf_counter()
        try:
            body = self._post({"dataframe_split": {"columns": columns, "data": rows}})
        except _RetryableError as exc:
            self.observe(self.name, "error", time.perf_counter() - started)
            log.warning("model server unavailable", extra={"model": self.name, "error": str(exc)})
            raise ModelUnavailable(f"{self.name} model server unavailable: {exc}") from exc
        except ModelUnavailable:
            self.observe(self.name, "rejected", time.perf_counter() - started)
            raise
        self.observe(self.name, "ok", time.perf_counter() - started)
        predictions = body.get("predictions", body)
        if isinstance(predictions, dict):  # column-oriented output
            keys = list(predictions)
            predictions = [
                dict(zip(keys, vals, strict=True))
                for vals in zip(*predictions.values(), strict=True)
            ]
        if len(predictions) != len(rows):
            raise ModelUnavailable(f"{self.name} returned {len(predictions)} rows for {len(rows)}")
        return predictions

    def ping(self) -> tuple[bool, str]:
        """``(ok, detail)`` from ``GET /ping``."""
        try:
            response = self.session.get(f"{self.base_url}/ping", timeout=min(self.timeout_s, 3))
            return response.status_code == 200, f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            return False, exc.__class__.__name__
