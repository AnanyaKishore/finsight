"""Prometheus instrumentation for the backend (scraped at ``GET /metrics``).

Metric names follow Prometheus conventions (``_total`` counters, ``_seconds``
histograms). Labels use the *route template* (``/api/stock/{ticker}``), never
the raw path, to keep cardinality bounded. Day 8 builds the Grafana panels and
alert rules (p95 latency, error rate, model failures, stale forecasts) on these.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class ApiMetrics:
    """All backend metrics, on a private registry (one per app; test-friendly)."""

    def __init__(self) -> None:
        """Create the metric families."""
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "finsight_http_requests_total",
            "HTTP requests by route and status code",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "finsight_http_request_duration_seconds",
            "HTTP request latency by route",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.model_calls = Counter(
            "finsight_model_requests_total",
            "Calls to model servers by outcome (ok, error, rejected)",
            ["model", "outcome"],
            registry=self.registry,
        )
        self.model_latency = Histogram(
            "finsight_model_request_duration_seconds",
            "Model server call latency (including retries)",
            ["model"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.forecasts = Counter(
            "finsight_forecasts_served_total",
            "Forecasts served by source (model = fresh call, cache = stored today, "
            "stale = model down, older forecast)",
            ["source"],
            registry=self.registry,
        )
        self.dependency_up = Gauge(
            "finsight_dependency_up",
            "1 if the dependency answered the last health check",
            ["dependency"],
            registry=self.registry,
        )

    def observe_model(self, model: str, outcome: str, seconds: float) -> None:
        """Callback for :class:`~finsight.api.model_client.ModelClient`."""
        self.model_calls.labels(model, outcome).inc()
        self.model_latency.labels(model).observe(seconds)
