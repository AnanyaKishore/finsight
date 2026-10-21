"""Small JSON-over-HTTP client with timeouts, retries and error classification.

Every external API call in FinSight goes through :class:`HttpClient`, so the
retry / timeout / logging policy is defined once:

* connection errors, timeouts, HTTP 429 and 5xx  -> retryable ``ProviderError``
* other 4xx (bad key, bad symbol)                -> non-retryable ``ProviderError``
* a provider-specific ``classify`` hook can turn a response into a
  ``QuotaExceededError`` (never retried) before the generic rules apply.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import requests

from finsight.common.exceptions import ProviderError
from finsight.common.logging import get_logger
from finsight.common.retry import retry_with_backoff

log = get_logger(__name__)

# Hook signature: (status_code, parsed_json_or_None) -> None, raising to override.
Classifier = Callable[[int, Any], None]


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


class HttpClient:
    """Thin wrapper over a :class:`requests.Session` for one provider."""

    def __init__(
        self,
        provider: str,
        *,
        timeout_s: float = 20.0,
        max_retries: int = 4,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        min_interval_s: float = 0.0,
        classify: Classifier | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        redact_params: tuple[str, ...] = ("access_key", "token", "apikey"),
    ) -> None:
        """Create a client.

        Args:
            provider: Name used in logs and errors.
            timeout_s: Per-request timeout (connect + read).
            max_retries: Retries after the first attempt.
            backoff_base_s: Base delay for exponential backoff.
            backoff_max_s: Maximum single backoff delay.
            min_interval_s: Client-side rate limit; minimum gap between calls.
            classify: Optional provider-specific response classifier.
            session: Injected session (tests).
            sleep: Injected sleep (tests use a no-op).
            redact_params: Query parameters never written to logs (secrets).
        """
        self.provider = provider
        self.timeout_s = timeout_s
        self.min_interval_s = min_interval_s
        self.classify = classify
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "finsight/0.1 (academic project)")
        self._sleep = sleep
        self._last_call = 0.0
        self._redact = set(redact_params)
        self.calls_made = 0
        self._get_with_retry = retry_with_backoff(
            max_retries=max_retries,
            base_delay=backoff_base_s,
            max_delay=backoff_max_s,
            is_retryable=_is_retryable,
            sleep=sleep,
        )(self._get_once)

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        """GET ``url`` and return the decoded JSON body (with retries)."""
        return self._get_with_retry(url, dict(params or {}))

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)

    def _get_once(self, url: str, params: dict[str, Any]) -> Any:
        self._throttle()
        safe_params = {k: ("***" if k in self._redact else v) for k, v in params.items()}
        started = time.monotonic()
        try:
            response = self.session.get(url, params=params, timeout=self.timeout_s)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise ProviderError(
                f"network error: {exc.__class__.__name__}", provider=self.provider, retryable=True
            ) from exc
        finally:
            self._last_call = time.monotonic()
            self.calls_made += 1

        try:
            body: Any = response.json()
        except ValueError:
            body = None
        log.debug(
            "http get",
            extra={
                "provider": self.provider,
                "url": url,
                "params": safe_params,
                "status": response.status_code,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            },
        )

        if self.classify is not None:
            self.classify(response.status_code, body)  # may raise QuotaExceededError etc.

        status = response.status_code
        if status == 429 or status >= 500:
            raise ProviderError(
                f"HTTP {status}", provider=self.provider, retryable=True, status=status
            )
        if status >= 400:
            raise ProviderError(
                f"HTTP {status}: {str(body)[:200]}",
                provider=self.provider,
                retryable=False,
                status=status,
            )
        if body is None:
            raise ProviderError("response was not JSON", provider=self.provider, retryable=True)
        return body
