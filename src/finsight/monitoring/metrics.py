"""Prometheus gauges for the monitoring report (served by the scheduler's /metrics).

Alert rules (``monitoring/prometheus/rules/finsight.yml``) use these names; the
``*_detected`` gauges carry the decision so the thresholds live in one place
(``params.yaml`` -> ``monitoring``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from prometheus_client import CollectorRegistry, Gauge


class MonitorMetrics:
    """Gauges updated from each monitoring report."""

    def __init__(self, registry: CollectorRegistry) -> None:
        """Register the gauges on the scheduler's registry."""

        def gauge(name: str, doc: str, labels: tuple[str, ...] = ()) -> Gauge:
            return Gauge(name, doc, list(labels), registry=registry)

        self.psi = gauge("finsight_drift_psi", "PSI, recent vs reference", ("feature", "ticker"))
        self.ks = gauge(
            "finsight_drift_ks", "KS statistic, recent vs reference", ("feature", "ticker")
        )
        self.drifted_share = gauge(
            "finsight_drift_drifted_ticker_share", "Share of tickers whose returns drifted"
        )
        self.detected = gauge(
            "finsight_drift_detected", "1 when drift crosses the configured rule", ("feature",)
        )
        self.sentiment_mean = gauge(
            "finsight_sentiment_mean_score", "Mean article sentiment", ("window",)
        )
        self.mape = gauge(
            "finsight_forecast_live_mape_percent",
            "Live MAPE of stored forecasts on realised prices",
            ("model",),
        )
        self.ticker_mape = gauge(
            "finsight_forecast_live_mape_by_ticker_percent", "Live MAPE per ticker", ("ticker",)
        )
        self.error_points = gauge(
            "finsight_forecast_realised_points", "Realised forecast points in the window"
        )
        self.error_ratio = gauge(
            "finsight_forecast_error_ratio_to_naive", "Live MAPE divided by naive MAPE"
        )
        self.error_detected = gauge(
            "finsight_forecast_degraded", "1 when live error crosses the retrain rule"
        )
        self.lag = gauge(
            "finsight_data_lag_trading_days",
            "Trading days behind the expected session",
            ("ticker",),
        )
        self.quality = gauge(
            "finsight_data_quality_failed_checks",
            "Failed checks in the last validation",
            ("severity",),
        )
        self.model_version = gauge(
            "finsight_model_production_version", "Production model version", ("model",)
        )
        self.drill = gauge("finsight_drift_drill_active", "1 while a drift drill is running")
        self.updated = gauge(
            "finsight_monitoring_last_update_timestamp_seconds", "When the report was computed"
        )

    def update(self, report: Mapping[str, Any]) -> None:
        """Copy one report into the gauges (sections that failed are skipped)."""
        returns = report.get("returns") or {}
        for ticker, row in (returns.get("per_ticker") or {}).items():
            if "psi" in row:
                self.psi.labels("returns", ticker).set(row["psi"])
                self.ks.labels("returns", ticker).set(row["ks"])
        if "detected" in returns:
            self.drifted_share.set(returns.get("drifted_share", 0))
            self.detected.labels("returns").set(int(returns["detected"]))
        sentiment = report.get("sentiment") or {}
        if "psi" in sentiment:
            self.psi.labels("sentiment", "all").set(sentiment["psi"])
            self.ks.labels("sentiment", "all").set(sentiment["ks"])
            self.detected.labels("sentiment").set(int(sentiment["detected"]))
            self.sentiment_mean.labels("current").set(sentiment["mean_current"])
            self.sentiment_mean.labels("reference").set(sentiment["mean_reference"])
        forecast = report.get("forecast") or {}
        if "points" in forecast:
            self.error_points.set(forecast["points"])
            self.error_detected.set(int(forecast.get("detected", False)))
        if "mape" in forecast:
            self.mape.labels("lstm").set(forecast["mape"])
            self.mape.labels("naive").set(forecast["naive_mape"])
            if forecast.get("ratio_to_naive") is not None:
                self.error_ratio.set(forecast["ratio_to_naive"])
            for ticker, value in forecast.get("per_ticker_mape", {}).items():
                self.ticker_mape.labels(ticker).set(value)
        for ticker, lag in ((report.get("freshness") or {}).get("lag_trading_days") or {}).items():
            self.lag.labels(ticker).set(lag)
        quality = report.get("quality") or {}
        if "errors" in quality:
            self.quality.labels("error").set(quality["errors"])
            self.quality.labels("warn").set(quality["warnings"])
        for model, version in (report.get("models") or {}).items():
            if version and str(version).isdigit():
                self.model_version.labels(model).set(int(version))
        self.drill.set(1 if report.get("drill") else 0)
        generated = report.get("generated_at")
        if generated is not None:
            self.updated.set(generated.timestamp())
