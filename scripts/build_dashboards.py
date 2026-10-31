"""Generate the Grafana dashboards (dashboards as code, Day 8).

    python scripts/build_dashboards.py      # writes monitoring/grafana/dashboards/*.json

Grafana provisions these files at start-up (read-only in the UI), so the JSON
in Git is the single source of truth. Editing this script and re-running it is
easier to review than hand-edited dashboard JSON.

Panel conventions: one unit per panel (never two y-axes), legends below the
chart, thresholds drawn where an alert fires, status values mapped to words.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "monitoring" / "grafana" / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}
GREEN, AMBER, RED, BLUE, GREY = "#1baf7a", "#eda100", "#d03b3b", "#2a78d6", "#a8a69f"

STATUS_MAPPING = [
    {
        "type": "value",
        "options": {
            "1": {"text": "Succeeded", "color": GREEN},
            "0.5": {"text": "Partial", "color": AMBER},
            "0": {"text": "Failed", "color": RED},
        },
    }
]
YES_NO = [
    {
        "type": "value",
        "options": {"0": {"text": "No", "color": GREEN}, "1": {"text": "Yes", "color": RED}},
    }
]
UP_DOWN = [
    {
        "type": "value",
        "options": {"0": {"text": "Down", "color": RED}, "1": {"text": "Up", "color": GREEN}},
    }
]


class Board:
    """Collects panels and lays them out on Grafana's 24-column grid."""

    def __init__(self, uid: str, title: str, description: str, tags: list[str]) -> None:
        self.uid, self.title, self.description, self.tags = uid, title, description, tags
        self.panels: list[dict[str, Any]] = []
        self.x = self.y = self.row_h = 0

    def _place(self, w: int, h: int) -> dict[str, int]:
        if self.x + w > 24:
            self.x, self.y = 0, self.y + self.row_h
            self.row_h = 0
        pos = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return pos

    def row(self, title: str) -> None:
        """Start a new titled row."""
        if self.x:
            self.x, self.y = 0, self.y + self.row_h
        self.panels.append(
            {
                "type": "row",
                "title": title,
                "collapsed": False,
                "id": len(self.panels) + 1,
                "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1},
                "panels": [],
            }
        )
        self.y += 1
        self.row_h = 0

    def _panel(self, kind: str, title: str, targets: list[tuple[str, str]], w: int, h: int,
               description: str = "", **field: Any) -> dict[str, Any]:  # fmt: skip
        panel = {
            "id": len(self.panels) + 1,
            "type": kind,
            "title": title,
            "description": description,
            "datasource": DS,
            "gridPos": self._place(w, h),
            "targets": [
                {"refId": chr(65 + i), "datasource": DS, "expr": expr, "legendFormat": legend}
                for i, (expr, legend) in enumerate(targets)
            ],
            "fieldConfig": {"defaults": {}, "overrides": []},
            "options": {},
        }
        defaults = panel["fieldConfig"]["defaults"]
        if "unit" in field:
            defaults["unit"] = field["unit"]
        if "decimals" in field:
            defaults["decimals"] = field["decimals"]
        if "min" in field:
            defaults["min"] = field["min"]
        if "max" in field:
            defaults["max"] = field["max"]
        if "mappings" in field:
            defaults["mappings"] = field["mappings"]
        steps = field.get("thresholds") or [(None, GREEN)]
        defaults["thresholds"] = {
            "mode": "absolute",
            "steps": [{"value": v, "color": c} for v, c in steps],
        }
        self.panels.append(panel)
        return panel

    def stat(self, title: str, expr: str, w: int = 4, h: int = 4, legend: str = "",
             description: str = "", **field: Any) -> dict[str, Any]:  # fmt: skip
        """A single big number (or mapped word)."""
        panel = self._panel("stat", title, [(expr, legend)], w, h, description, **field)
        panel["options"] = {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "background" if "mappings" in field or "thresholds" in field else "none",
            "graphMode": "none",
            "textMode": "auto",
            "justifyMode": "center",
        }
        return panel

    def series(self, title: str, targets: list[tuple[str, str]], w: int = 12, h: int = 8,
               description: str = "", line: float | None = None, stack: bool = False,
               **field: Any) -> dict[str, Any]:  # fmt: skip
        """A time series; ``line`` draws a dashed threshold (where an alert fires)."""
        if line is not None:
            field.setdefault("thresholds", [(None, GREEN), (line, RED)])
        panel = self._panel("timeseries", title, targets, w, h, description, **field)
        panel["fieldConfig"]["defaults"]["custom"] = {
            "lineWidth": 2,
            "fillOpacity": 10 if stack else 0,
            "showPoints": "never",
            "spanNulls": True,
            "stacking": {"mode": "normal" if stack else "none"},
            "thresholdsStyle": {"mode": "dashed" if line is not None else "off"},
        }
        panel["options"] = {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        }
        return panel

    def bars(self, title: str, expr: str, legend: str, w: int = 12, h: int = 8,
             description: str = "", **field: Any) -> dict[str, Any]:  # fmt: skip
        """Horizontal bar gauge, one bar per series (latest value)."""
        panel = self._panel("bargauge", title, [(expr, legend)], w, h, description, **field)
        panel["options"] = {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "orientation": "horizontal",
            "displayMode": "basic",
            "showUnfilled": True,
        }
        return panel

    def text(self, content: str, w: int = 24, h: int = 3) -> None:
        """A markdown note."""
        self.panels.append(
            {
                "id": len(self.panels) + 1,
                "type": "text",
                "title": "",
                "gridPos": self._place(w, h),
                "options": {"mode": "markdown", "content": content},
            }
        )

    def json(self) -> dict[str, Any]:
        """The dashboard document Grafana expects."""
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": ["finsight", *self.tags],
            "timezone": "browser",
            "editable": False,
            "graphTooltip": 1,
            "refresh": "30s",
            "time": {"from": "now-6h", "to": "now"},
            "schemaVersion": 39,
            "version": 1,
            "links": [
                {"title": t, "type": "link", "url": u, "targetBlank": True}
                for t, u in (
                    ("Pipeline Console", "http://localhost:8080/#/pipeline"),
                    ("Prometheus alerts", "http://localhost:9090/alerts"),
                    ("Alertmanager", "http://localhost:9093"),
                    ("MLflow", "http://localhost:5000"),
                )
            ]
            + [
                {"title": "FinSight dashboards", "type": "dashboards", "tags": ["finsight"],
                 "asDropdown": True}  # fmt: skip
            ],
            "panels": self.panels,
        }


# ---------------------------------------------------------------- overview
def overview() -> Board:
    """Home dashboard: is everything healthy right now?"""
    b = Board("finsight-overview", "FinSight · Overview",
              "Health of the whole system at a glance.", ["overview"])  # fmt: skip
    b.row("Right now")
    b.stat("API", 'max(up{job="backend"})', mappings=UP_DOWN)
    b.stat("Error rate (5 min)", "finsight:http_error_ratio:rate5m or vector(0)", unit="percentunit",
           decimals=2, thresholds=[(None, GREEN), (0.01, RED)])  # fmt: skip
    b.stat("p95 latency (cached routes)",
           'max(finsight:http_latency_p95:rate5m{route!~".*(prediction|analyze).*"})',
           unit="s", decimals=3, thresholds=[(None, GREEN), (0.3, RED)])  # fmt: skip
    b.stat("Last daily run", 'finsight_pipeline_last_run_status{kind="daily"}',
           mappings=STATUS_MAPPING)  # fmt: skip
    b.stat("Returns drift", 'finsight_drift_detected{feature="returns"}', mappings=YES_NO,
           description="Yes = enough tickers drifted to trigger a retrain")  # fmt: skip
    b.stat("Firing alerts", 'count(ALERTS{alertstate="firing"}) or vector(0)',
           thresholds=[(None, GREEN), (1, AMBER)])  # fmt: skip
    b.stat("LSTM live MAPE", 'finsight_forecast_live_mape_percent{model="lstm"}', unit="percent",
           decimals=2, description="Stored forecasts vs the prices that followed")  # fmt: skip
    b.stat("Naive live MAPE", 'finsight_forecast_live_mape_percent{model="naive"}',
           unit="percent", decimals=2)  # fmt: skip
    b.stat("Prices behind (trading days)", "max(finsight_data_lag_trading_days)",
           thresholds=[(None, GREEN), (2, AMBER), (3, RED)])  # fmt: skip
    b.stat("LSTM in production", 'finsight_model_production_version{model="lstm"}',
           legend="version")  # fmt: skip
    b.stat("Sentiment model", 'finsight_model_production_version{model="sentiment"}')
    b.stat("Host memory available", "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes",
           unit="percentunit", decimals=0, thresholds=[(None, RED), (0.1, AMBER), (0.2, GREEN)])  # fmt: skip
    b.row("Trends")
    b.series("API requests per second", [("sum by (route) (rate(finsight_http_requests_total[5m]))",
             "{{route}}")], unit="reqps")  # fmt: skip
    b.series("Returns drift (PSI) per ticker",
             [('finsight_drift_psi{feature="returns"}', "{{ticker}}")], line=0.25,
             description="Dashed line: significant shift (monitoring.psi_threshold)")  # fmt: skip
    b.series("Memory per container",
             [('container_memory_working_set_bytes{name=~"finsight-.*"}', "{{name}}")],
             unit="bytes", stack=True)  # fmt: skip
    b.series("Firing alerts", [('count by (alertname) (ALERTS{alertstate="firing"})',
             "{{alertname}}")], decimals=0)  # fmt: skip
    return b


# --------------------------------------------------------------------- API
def api() -> Board:
    """Backend latency, errors and model-server calls."""
    b = Board("finsight-api", "FinSight · API & model serving",
              "FastAPI backend and the two MLflow model servers (via the backend).", ["api"])  # fmt: skip
    b.row("Traffic and errors")
    b.series("Requests per second by status",
             [("sum by (status) (rate(finsight_http_requests_total[5m]))", "{{status}}")],
             unit="reqps", w=12)  # fmt: skip
    b.series("Error ratio (5xx / all)", [("finsight:http_error_ratio:rate5m", "error ratio")],
             unit="percentunit", line=0.01, w=12)  # fmt: skip
    b.row("Latency")
    b.series("p95 latency by route", [("finsight:http_latency_p95:rate5m", "{{route}}")],
             unit="s", line=0.3, w=12,
             description="Target: p95 < 300 ms for cached routes (dashed)")  # fmt: skip
    b.series("p50 / p95 / p99, all routes", [
        (f"histogram_quantile({q}, sum by (le) (rate(finsight_http_request_duration_seconds_bucket[5m])))",
         f"p{int(q * 100)}") for q in (0.5, 0.95, 0.99)], unit="s", w=12)  # fmt: skip
    b.row("Model servers")
    b.stat("lstm-serve", 'finsight_dependency_up{dependency="lstm-serve"}', mappings=UP_DOWN, w=3)
    b.stat("sentiment-serve", 'finsight_dependency_up{dependency="sentiment-serve"}',
           mappings=UP_DOWN, w=3)  # fmt: skip
    b.stat("MongoDB", 'finsight_dependency_up{dependency="mongodb"}', mappings=UP_DOWN, w=3)
    b.stat("MLflow registry", 'finsight_dependency_up{dependency="mlflow-registry"}',
           mappings=UP_DOWN, w=3)  # fmt: skip
    b.series("Model call p95 latency",
             [("histogram_quantile(0.95, sum by (le, model) (rate(finsight_model_request_duration_seconds_bucket[5m])))",
               "{{model}}")], unit="s", w=12, h=8)  # fmt: skip
    b.series("Model calls by outcome",
             [("sum by (model, outcome) (rate(finsight_model_requests_total[5m]))",
               "{{model}} {{outcome}}")], unit="reqps", w=12)  # fmt: skip
    b.series("Forecasts served by source",
             [("sum by (source) (increase(finsight_forecasts_served_total[1h]))", "{{source}}")],
             decimals=0, w=12,
             description="model = fresh call, cache = stored today, stale = model down")  # fmt: skip
    return b


# ---------------------------------------------------------------- pipeline
def pipeline() -> Board:
    """Scheduler runs, step durations, throughput, data freshness and quality."""
    b = Board("finsight-pipeline", "FinSight · Pipeline & data",
              "Daily and retrain runs from the scheduler; data freshness and quality.",
              ["pipeline"])  # fmt: skip
    b.row("Runs")
    b.stat("Last daily run", 'finsight_pipeline_last_run_status{kind="daily"}',
           mappings=STATUS_MAPPING)  # fmt: skip
    b.stat("Last retrain", 'finsight_pipeline_last_run_status{kind="retrain"}',
           mappings=STATUS_MAPPING)  # fmt: skip
    b.stat("Running now", "finsight_pipeline_running",
           mappings=[{"type": "value", "options": {"0": {"text": "Idle", "color": GREY},
                                                  "1": {"text": "Running", "color": BLUE}}}])  # fmt: skip
    b.stat("Last daily run took", 'finsight_pipeline_last_run_duration_seconds{kind="daily"}',
           unit="s")  # fmt: skip
    b.stat("Since last daily run", 'time() - finsight_pipeline_last_run_timestamp_seconds{kind="daily"}',
           unit="s", thresholds=[(None, GREEN), (80 * 3600, RED)])  # fmt: skip
    b.stat("Retrains (7 days)", "sum(increase(finsight_retrains_triggered_total[7d])) or vector(0)",
           decimals=0)  # fmt: skip
    b.series("Finished runs by status (per day)",
             [("sum by (kind, status) (increase(finsight_pipeline_runs_total[1d]))",
               "{{kind}} {{status}}")], decimals=0, w=12)  # fmt: skip
    b.bars("Step durations, last run", "finsight_pipeline_step_duration_seconds", "{{step}}",
           unit="s", w=12)  # fmt: skip
    b.row("Throughput (one unit per chart)")
    b.series("Spark ETL", [('finsight_pipeline_throughput{measure="etl_rows_per_s"}', "rows/s")],
             w=8)  # fmt: skip
    b.series("Sentiment scoring", [('finsight_pipeline_throughput{measure="scoring_articles_per_s"}',
             "articles/s")], w=8)  # fmt: skip
    b.series("Ingestion", [('finsight_pipeline_throughput{measure="ingest_rows_per_s"}', "rows/s")],
             w=8)  # fmt: skip
    b.row("Data")
    b.bars("Trading days behind, per ticker", "finsight_data_lag_trading_days", "{{ticker}}",
           thresholds=[(None, GREEN), (2, AMBER), (3, RED)], decimals=0, w=12)  # fmt: skip
    b.stat("Blocking quality checks failed", 'finsight_data_quality_failed_checks{severity="error"}',
           thresholds=[(None, GREEN), (1, RED)], w=6, h=8)  # fmt: skip
    b.stat("Quality warnings", 'finsight_data_quality_failed_checks{severity="warn"}',
           thresholds=[(None, GREEN), (1, AMBER)], w=6, h=8)  # fmt: skip
    return b


# ------------------------------------------------------------------ models
def models() -> Board:
    """Data drift, live forecast error and the retraining loop."""
    b = Board("finsight-models", "FinSight · Drift & model performance",
              "Drift of returns and news sentiment, live forecast error, automated retrains.",
              ["models", "drift"])  # fmt: skip
    b.text("**How to read this.** PSI compares the last 60 trading days of each stock's daily returns "
           "with the year before (above 0.25 = significant). When at least 30% of stocks drift, or the "
           "live forecast error is 1.25x worse than the naive forecast, an alert calls the scheduler, "
           "which retrains the LSTM; the promotion gate then keeps or replaces the production model. "
           "Thresholds: `params.yaml` → `monitoring`.", h=3)  # fmt: skip
    b.row("Returns drift")
    b.stat("Drift detected", 'finsight_drift_detected{feature="returns"}', mappings=YES_NO, w=4)
    b.stat("Tickers drifted", "finsight_drift_drifted_ticker_share", unit="percentunit",
           decimals=0, thresholds=[(None, GREEN), (0.3, RED)], w=4)  # fmt: skip
    b.stat("Drill running", "finsight_drift_drill_active",
           mappings=[{"type": "value", "options": {"0": {"text": "No", "color": GREY},
                                                  "1": {"text": "DRILL", "color": AMBER}}}], w=4)  # fmt: skip
    b.bars("PSI per ticker (now)", 'finsight_drift_psi{feature="returns"}', "{{ticker}}",
           decimals=3, thresholds=[(None, GREEN), (0.1, AMBER), (0.25, RED)], w=12, h=8)  # fmt: skip
    b.series("PSI per ticker over time", [('finsight_drift_psi{feature="returns"}', "{{ticker}}")],
             line=0.25, decimals=3, w=12)  # fmt: skip
    b.series("KS statistic per ticker", [('finsight_drift_ks{feature="returns"}', "{{ticker}}")],
             decimals=3, min=0, max=1, w=12)  # fmt: skip
    b.row("News sentiment")
    b.series("Sentiment PSI (all tickers)", [('finsight_drift_psi{feature="sentiment"}', "PSI")],
             line=0.25, decimals=3, w=12)  # fmt: skip
    b.series("Mean article score: recent vs reference",
             [("finsight_sentiment_mean_score", "{{window}}")], decimals=3, min=-1, max=1, w=12)  # fmt: skip
    b.row("Live forecast error")
    b.series("Live MAPE: LSTM vs naive (same points)",
             [("finsight_forecast_live_mape_percent", "{{model}}")], unit="percent", w=12,
             description="Forecasts stored in the last 30 days whose target days have traded")  # fmt: skip
    b.series("LSTM MAPE / naive MAPE", [("finsight_forecast_error_ratio_to_naive", "ratio")],
             line=1.25, decimals=2, w=12)  # fmt: skip
    b.bars("Live MAPE per ticker", "finsight_forecast_live_mape_by_ticker_percent", "{{ticker}}",
           unit="percent", w=12)  # fmt: skip
    b.stat("Realised points", "finsight_forecast_realised_points", w=4, h=8,
           description="No verdict below monitoring.min_error_points (20)")  # fmt: skip
    b.stat("Retrains triggered (7 d)", "sum(increase(finsight_retrains_triggered_total[7d])) or vector(0)",
           decimals=0, w=4, h=8)  # fmt: skip
    b.stat("Skipped (cooldown, 7 d)", "increase(finsight_retrains_skipped_total[7d]) or vector(0)",
           decimals=0, w=4, h=8)  # fmt: skip
    b.series("Production model versions", [("finsight_model_production_version", "{{model}}")],
             decimals=0, w=12, description="A step up = a new version was promoted")  # fmt: skip
    return b


# ---------------------------------------------------------- infrastructure
def infrastructure() -> Board:
    """Host, containers, MongoDB and nginx."""
    b = Board("finsight-infra", "FinSight · Infrastructure",
              "Laptop resources, every container, MongoDB and nginx.", ["infrastructure"])  # fmt: skip
    b.row("Host")
    b.series("CPU busy", [('1 - avg(rate(node_cpu_seconds_total{mode="idle"}[2m]))', "CPU")],
             unit="percentunit", min=0, max=1, w=8)  # fmt: skip
    b.series("Memory available", [("node_memory_MemAvailable_bytes", "available"),
                                  ("node_memory_MemTotal_bytes", "total")], unit="bytes", w=8)  # fmt: skip
    b.series("Disk free (/)",
             [('node_filesystem_avail_bytes{mountpoint="/",fstype!~"tmpfs|overlay"} / node_filesystem_size_bytes{mountpoint="/",fstype!~"tmpfs|overlay"}',
               "free")], unit="percentunit", min=0, max=1, line=0.1, w=8)  # fmt: skip
    b.row("Containers")
    b.bars("Memory used / limit", 'container_memory_working_set_bytes{name=~"finsight-.*"} / (container_spec_memory_limit_bytes{name=~"finsight-.*"} > 0)',
           "{{name}}", unit="percentunit", min=0, max=1,
           thresholds=[(None, GREEN), (0.75, AMBER), (0.9, RED)], w=12, h=10)  # fmt: skip
    b.series("Memory per container", [('container_memory_working_set_bytes{name=~"finsight-.*"}',
             "{{name}}")], unit="bytes", w=12, h=10)  # fmt: skip
    b.series("CPU per container (cores)",
             [('sum by (name) (rate(container_cpu_usage_seconds_total{name=~"finsight-.*"}[2m]))',
               "{{name}}")], decimals=2, w=12)  # fmt: skip
    b.series("Container restarts (start time changes)",
             [('changes(container_start_time_seconds{name=~"finsight-.*"}[1h])', "{{name}}")],
             decimals=0, w=12,
             description="Includes the model servers restarting after a promotion")  # fmt: skip
    b.row("MongoDB and nginx")
    b.stat("MongoDB", "max(mongodb_up)", mappings=UP_DOWN, w=4)
    b.series("MongoDB operations per second",
             [("sum by (type) (rate(mongodb_op_counters_total[5m]))", "{{type}}")], unit="ops", w=10)  # fmt: skip
    b.series("MongoDB connections", [('mongodb_connections{state="current"}', "current")],
             decimals=0, w=10)  # fmt: skip
    b.stat("nginx", "max(nginx_up)", mappings=UP_DOWN, w=4)
    b.series("Portal requests per second", [("rate(nginx_http_requests_total[5m])", "requests")],
             unit="reqps", w=10)  # fmt: skip
    b.series(
        "Portal active connections", [("nginx_connections_active", "active")], decimals=0, w=10
    )
    return b


def main() -> None:
    """Write every dashboard."""
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (overview, api, pipeline, models, infrastructure):
        board = build()
        path = OUT / f"{board.uid}.json"
        path.write_text(json.dumps(board.json(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(board.panels)} panels)")


if __name__ == "__main__":
    main()
