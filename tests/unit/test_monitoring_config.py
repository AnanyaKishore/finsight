"""Static checks of the Day 8 monitoring stack: compose, Prometheus, Alertmanager, Grafana.

When ``promtool`` / ``amtool`` are installed (they ship in the Prometheus and
Alertmanager images), the configs and rule unit tests are also run through them.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / "monitoring"
MONITORING_SERVICES = (
    "prometheus",
    "alertmanager",
    "grafana",
    "node-exporter",
    "cadvisor",
    "mongodb-exporter",
    "nginx-exporter",
)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compose() -> dict:
    return load(ROOT / "docker-compose.yml")["services"]


@pytest.fixture(scope="module")
def rules() -> list[dict]:
    groups = load(MON / "prometheus/rules/finsight.yml")["groups"]
    return [rule for group in groups for rule in group["rules"]]


def test_monitoring_services_are_pinned_limited_and_local_only(compose):
    for name in MONITORING_SERVICES:
        service = compose[name]
        assert re.search(r":v?\d+\.\d+\.\d+$", service["image"]), name  # exact version tag
        assert service.get("mem_limit"), name
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:"), (name, port)  # nothing exposed to the network


def test_secrets_stay_out_of_configs(compose):
    grafana = compose["grafana"]["environment"]
    assert grafana["GF_SECURITY_ADMIN_PASSWORD"].startswith("${GRAFANA_ADMIN_PASSWORD:?")
    assert grafana["GF_ANALYTICS_REPORTING_ENABLED"] == "false"
    am = compose["alertmanager"]
    script = am["command"][0]
    assert "umask 077" in script and "/tmp/scheduler_token" in script  # noqa: S108
    config = (MON / "alertmanager/alertmanager.yml").read_text(encoding="utf-8")
    assert "credentials_file: /tmp/scheduler_token" in config
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GRAFANA_ADMIN_PASSWORD=replace-me" in env_example


def test_prometheus_scrapes_every_component(compose):
    config = load(MON / "prometheus/prometheus.yml")
    targets = {
        target.split(":")[0]
        for job in config["scrape_configs"]
        for sc in job["static_configs"]
        for target in sc["targets"]
    }
    # Services with their own /metrics, plus exporters for the rest (cAdvisor
    # covers mlflow, the model servers and every other container).
    for name in (
        "backend",
        "scheduler",
        "alertmanager",
        "grafana",
        "node-exporter",
        "cadvisor",
        "mongodb-exporter",
        "nginx-exporter",
    ):
        assert name in targets and name in compose, name
    assert config["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"] == [
        "alertmanager:9093"
    ]
    assert config["rule_files"] == ["/etc/prometheus/rules/*.yml"]


def test_every_alert_is_explained_and_retrain_alerts_are_labelled(rules):
    alerts = [r for r in rules if "alert" in r]
    assert len(alerts) >= 15
    for rule in alerts:
        assert rule["labels"]["severity"] in {"critical", "warning", "info"}, rule["alert"]
        assert rule["annotations"]["summary"], rule["alert"]
    retrain = {r["alert"] for r in alerts if r["labels"].get("action") == "retrain"}
    assert retrain == {"ReturnsDriftDetected", "ForecastErrorDegraded"}
    names = {r["alert"] for r in alerts}
    for required in (
        "TargetDown",
        "ApiHighErrorRate",
        "PipelineRunFailed",
        "DataStale",
        "SentimentShift",
        "HostMemoryLow",
    ):
        assert required in names


def test_alertmanager_sends_retrain_alerts_to_the_scheduler_once_a_day():
    config = load(MON / "alertmanager/alertmanager.yml")
    route = next(r for r in config["route"]["routes"] if r["receiver"] == "scheduler-retrain")
    assert 'action="retrain"' in route["matchers"][0]
    assert route["repeat_interval"] == "24h"
    receiver = next(r for r in config["receivers"] if r["name"] == "scheduler-retrain")
    hook = receiver["webhook_configs"][0]
    assert hook["url"] == "http://scheduler:8001/retrain" and hook["send_resolved"] is False


def test_mongodb_exporter_collects_what_the_dashboard_queries(compose):
    # mongodb_op_counters_total and mongodb_connections come from serverStatus,
    # which the 0.4x exporter only collects with the diagnosticdata collector.
    assert "--collector.diagnosticdata" in compose["mongodb-exporter"]["command"]


def test_rule_unit_tests_cover_the_retrain_alerts():
    tests = (MON / "prometheus/tests/finsight_test.yml").read_text(encoding="utf-8")
    assert "ReturnsDriftDetected" in tests and "ForecastErrorDegraded" in tests


def test_dashboards_are_generated_valid_and_use_the_provisioned_datasource():
    datasource = load(MON / "grafana/provisioning/datasources/prometheus.yml")["datasources"][0]
    assert datasource["uid"] == "prometheus" and datasource["url"] == "http://prometheus:9090"
    boards = sorted((MON / "grafana/dashboards").glob("*.json"))
    assert {b.stem for b in boards} == {
        "finsight-overview",
        "finsight-api",
        "finsight-pipeline",
        "finsight-models",
        "finsight-infra",
    }
    for path in boards:
        board = json.loads(path.read_text(encoding="utf-8"))
        assert board["uid"] == path.stem and board["refresh"]
        panels = [p for p in board["panels"] if p["type"] not in {"row", "text"}]
        assert panels, path.name
        for panel in panels:
            assert panel["datasource"]["uid"] == "prometheus", (path.name, panel["title"])
            assert panel["targets"] and all(t["expr"] for t in panel["targets"])


def test_nginx_exposes_stub_status_on_an_internal_port_only():
    conf = (ROOT / "frontend/nginx/default.conf.template").read_text(encoding="utf-8")
    assert "listen 8081;" in conf and "stub_status;" in conf
    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert ":8081" not in compose_text.split("nginx-exporter:")[0]  # never published


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool not installed")
def test_promtool_accepts_config_and_rule_tests():
    rules_dir = MON / "prometheus/rules"
    check = subprocess.run(  # noqa: S603
        ["promtool", "check", "rules", *map(str, rules_dir.glob("*.yml"))],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    test = subprocess.run(  # noqa: S603
        ["promtool", "test", "rules", "finsight_test.yml"],  # noqa: S607
        cwd=MON / "prometheus/tests",
        capture_output=True,
        text=True,
        check=False,
    )
    assert test.returncode == 0, test.stdout + test.stderr


@pytest.mark.skipif(shutil.which("amtool") is None, reason="amtool not installed")
def test_amtool_accepts_the_alertmanager_config():
    result = subprocess.run(  # noqa: S603
        ["amtool", "check-config", str(MON / "alertmanager/alertmanager.yml")],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
