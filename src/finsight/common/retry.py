"""Retry with exponential backoff and full jitter.

Implemented in-house (instead of pulling in ``tenacity``) because the policy is
small, must be unit-testable without real sleeping, and is a design choice we
want to be able to explain:

    delay_n = uniform(0, min(max_delay, base * 2**n))      ("full jitter")

Full jitter spreads retries from many clients over time, which avoids the
"thundering herd" that plain exponential backoff causes after an outage.
Only errors classified as *retryable* are retried; e.g. an invalid API key
fails immediately instead of burning the retry budget.
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from finsight.common.logging import get_logger

P = ParamSpec("P")
T = TypeVar("T")
log = get_logger(__name__)


def backoff_delay(attempt: int, base: float, max_delay: float, rng: random.Random) -> float:
    """Return the full-jitter delay (seconds) before retry number ``attempt`` (0-based)."""
    cap = min(max_delay, base * (2**attempt))
    return rng.uniform(0, cap)


def retry_with_backoff(
    *,
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    is_retryable: Callable[[BaseException], bool] = lambda exc: True,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate a function so retryable exceptions are retried with backoff.

    Args:
        max_retries: Retries *after* the first attempt (total calls = 1 + max_retries).
        base_delay: Base of the exponential schedule, in seconds.
        max_delay: Upper bound for a single sleep, in seconds.
        is_retryable: Predicate deciding whether an exception is worth retrying.
            Non-retryable exceptions are re-raised immediately.
        sleep: Injected sleep function (tests pass a no-op).
        rng: Injected random generator (tests pass a seeded one).

    Returns:
        A decorator. The wrapped function re-raises the last exception when all
        attempts are exhausted.
    """
    generator = rng or random.Random()  # noqa: S311 -- jitter, not cryptography

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt >= max_retries or not is_retryable(exc):
                        raise
                    delay = backoff_delay(attempt, base_delay, max_delay, generator)
                    log.warning(
                        "retrying after error",
                        extra={
                            "func": func.__qualname__,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "delay_s": round(delay, 2),
                            "error": str(exc),
                        },
                    )
                    sleep(delay)
                    attempt += 1

        return wrapper

    return decorator
