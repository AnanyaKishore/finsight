"""Backend configuration, from environment variables (12-factor style).

=============================  ==========================  =====================================
variable                       default                     meaning
=============================  ==========================  =====================================
``LSTM_SERVE_URL``             ``http://localhost:5001``   MLflow server for ``finsight-lstm``
``SENTIMENT_SERVE_URL``        ``http://localhost:5002``   MLflow server for ``finsight-sentiment``
``FRONTEND_ORIGINS``           localhost + 127.0.0.1:8080  comma-separated CORS allow-list
``MODEL_TIMEOUT_S``            ``10``                      per model call
``MODEL_RETRIES``              ``2``                       retries after the first attempt
``REGISTRY_CACHE_S``           ``60``                      cache for ``/api/model-info``
``SCHEDULER_URL``              ``http://localhost:8001``   scheduler service (Day 7)
``SCHEDULER_TOKEN``            (unset)                     shared token; unset = no console triggers
``ALERTMANAGER_URL``           ``http://localhost:9093``   active alerts for ``/api/monitoring``
=============================  ==========================  =====================================

MongoDB and MLflow settings come from the shared ``finsight.common.config``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from finsight.common.config import usable_secret

# The portal (Day 6) at http://localhost:8080; browsers treat 127.0.0.1 as a
# different origin, so both spellings are allowed by default.
DEFAULT_ORIGINS = ("http://localhost:8080", "http://127.0.0.1:8080")


def _list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class ApiConfig:
    """Resolved backend settings."""

    lstm_url: str = "http://localhost:5001"
    sentiment_url: str = "http://localhost:5002"
    frontend_origins: tuple[str, ...] = field(default=DEFAULT_ORIGINS)
    model_timeout_s: float = 10.0
    model_retries: int = 2
    registry_cache_s: float = 60.0
    scheduler_url: str = "http://localhost:8001"
    alertmanager_url: str = "http://localhost:9093"
    scheduler_token: str | None = field(default=None, repr=False)
    lstm_name: str = "finsight-lstm"
    sentiment_name: str = "finsight-sentiment"
    history_days_in_forecast: int = 60
    max_analyze_texts: int = 32
    max_text_chars: int = 2000

    @classmethod
    def from_env(cls) -> ApiConfig:
        """Read overrides from the environment."""
        env = os.environ
        return cls(
            lstm_url=env.get("LSTM_SERVE_URL", cls.lstm_url).rstrip("/"),
            sentiment_url=env.get("SENTIMENT_SERVE_URL", cls.sentiment_url).rstrip("/"),
            frontend_origins=_list(env.get("FRONTEND_ORIGINS", ",".join(DEFAULT_ORIGINS))),
            model_timeout_s=float(env.get("MODEL_TIMEOUT_S", cls.model_timeout_s)),
            model_retries=int(env.get("MODEL_RETRIES", cls.model_retries)),
            registry_cache_s=float(env.get("REGISTRY_CACHE_S", cls.registry_cache_s)),
            scheduler_url=env.get("SCHEDULER_URL", cls.scheduler_url).rstrip("/"),
            alertmanager_url=env.get("ALERTMANAGER_URL", cls.alertmanager_url).rstrip("/"),
            scheduler_token=usable_secret(env.get("SCHEDULER_TOKEN")),
        )
