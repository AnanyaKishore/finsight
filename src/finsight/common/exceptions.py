"""Exception hierarchy for FinSight.

A single base class lets top-level entry points (CLI, scheduler, API) catch
"expected" failures and degrade gracefully, while genuinely unexpected errors
(bugs) still surface with a full traceback.
"""

from __future__ import annotations


class FinSightError(Exception):
    """Base class for all anticipated FinSight failures."""


class ConfigError(FinSightError):
    """A required setting is missing or invalid (e.g. no API key in .env)."""


class ProviderError(FinSightError):
    """An external data provider failed.

    Attributes:
        provider: Short provider name, e.g. ``"marketstack"``.
        retryable: Whether retrying the same request could succeed
            (network blips, HTTP 429/5xx) or not (bad API key, bad symbol).
        status: HTTP status code when the failure came from an HTTP response.
    """

    def __init__(
        self, message: str, *, provider: str, retryable: bool = False, status: int | None = None
    ) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable
        self.status = status


class QuotaExceededError(ProviderError):
    """A provider's request budget is used up (never retried; triggers fallback)."""

    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message, provider=provider, retryable=False, status=429)


class DataValidationError(FinSightError):
    """Data failed a quality check (used from Day 2 by the validate stage)."""
