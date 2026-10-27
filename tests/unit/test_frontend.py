"""Static checks for the web portal (Day 6): files, wiring and container config.

The portal is plain HTML/CSS/JS, so these tests need no browser. The
end-to-end browser test (compose up -> API -> UI) is part of Day 9.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "frontend"
PAGES = ["index.html", "help.html"]


def _refs(page: str) -> list[str]:
    text = (WEB / page).read_text(encoding="utf-8")
    return re.findall(r'(?:src|href)="([^"#][^"]*)"', text)


@pytest.mark.parametrize("page", PAGES)
def test_every_local_asset_exists_and_nothing_loads_from_the_internet(page):
    for ref in _refs(page):
        if ref.startswith(("data:", "mailto:")):
            continue
        assert not ref.startswith(("http://", "https://", "//")), f"{page} loads {ref} from outside"
        path = ref.split("#")[0].split("?")[0]
        assert (WEB / path).exists(), f"{page} references missing file {ref}"


def test_js_modules_import_only_existing_local_files():
    for module in (WEB / "assets/js").glob("*.js"):
        for target in re.findall(r'from "(\./[^"]+)"', module.read_text(encoding="utf-8")):
            assert (module.parent / target).resolve().exists(), f"{module.name} imports {target}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_modules_parse():
    for module in (WEB / "assets/js").glob("*.js"):
        result = subprocess.run(
            ["node", "--check", "--input-type=module"],
            input=module.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{module.name}: {result.stderr}"


def test_nginx_template_generates_config_and_restricts_connections():
    conf = (WEB / "nginx/default.conf.template").read_text(encoding="utf-8")
    assert "location = /config.js" in conf and 'backendUrl: "${BACKEND_URL}"' in conf
    assert "connect-src 'self' ${BACKEND_URL}" in conf
    assert "location = /healthz" in conf and "listen       8080;" in conf


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("http://localhost:8000", True),
        ("https://finsight.example.org/api", True),
        ("http://192.168.1.20:8000", True),
        ('http://x"; alert(1)', False),
        ("localhost:8000", False),
        ("", False),
    ],
)
def test_backend_url_is_validated_before_nginx_starts(url, ok):
    script = WEB / "nginx/15-finsight-check-env.sh"
    env = {
        "BACKEND_URL": url,
        "MLFLOW_UI_URL": "http://localhost:5000",
        "PROMETHEUS_UI_URL": "http://localhost:9090",
        "GRAFANA_UI_URL": "http://localhost:3000",
        "PATH": "/usr/bin:/bin",
    }
    result = subprocess.run(
        ["sh", str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert (result.returncode == 0) is ok, result.stderr


def test_compose_wires_frontend_and_shares_the_serving_image():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    frontend = services["frontend"]
    assert frontend["ports"] == ["127.0.0.1:8080:8080"]
    assert "BACKEND_URL" in frontend["environment"]
    assert "depends_on" not in frontend  # the portal starts (and degrades) on its own
    assert "8080" in services["backend"]["environment"]["FRONTEND_ORIGINS"]
    assert "build" in services["lstm-serve"] and "build" not in services["sentiment-serve"]
    assert services["sentiment-serve"]["image"] == services["lstm-serve"]["image"]


def test_frontend_dockerfile_copies_existing_files():
    dockerfile = (ROOT / "docker/Dockerfile.frontend").read_text(encoding="utf-8")
    for source in re.findall(r"^COPY (?:--chmod=\d+ )?(.+) \S+$", dockerfile, flags=re.M):
        for path in source.split():
            assert (ROOT / path).exists(), f"Dockerfile copies missing {path}"
    assert "nginx-unprivileged" in dockerfile  # never runs as root


# ---------------------------------------------------------------- Day 7
def test_pipeline_console_is_routed_linked_and_documented():
    app = (WEB / "assets/js/app.js").read_text(encoding="utf-8")
    assert '"#/pipeline"' in app and "renderPipeline(" in app
    assert 'href="#/pipeline"' in (WEB / "index.html").read_text(encoding="utf-8")
    assert "Pipeline page" in (WEB / "help.html").read_text(encoding="utf-8")
    page = (WEB / "assets/js/pipeline.js").read_text(encoding="utf-8")
    for endpoint in (
        "pipelineStatus",
        "pipelineRuns",
        "pipelineRun",
        "pipelineDag",
        "pipelineStats",
        "pipelineTrigger",
    ):
        assert f"endpoints.{endpoint}(" in page
    api = (WEB / "assets/js/api.js").read_text(encoding="utf-8")
    assert '"/api/pipeline/trigger", { method: "POST"' in api
    # The browser never handles the scheduler token: it only talks to the backend.
    assert "SCHEDULER_TOKEN" not in api and "8001" not in page + api


def test_console_links_come_from_runtime_config():
    conf = (WEB / "nginx/default.conf.template").read_text(encoding="utf-8")
    assert (
        'prometheusUrl: "${PROMETHEUS_UI_URL}"' in conf
        and 'grafanaUrl: "${GRAFANA_UI_URL}"' in conf
    )
    dockerfile = (ROOT / "docker/Dockerfile.frontend").read_text(encoding="utf-8")
    assert "PROMETHEUS_UI_URL=" in dockerfile and "GRAFANA_UI_URL=" in dockerfile
    check = (WEB / "nginx/15-finsight-check-env.sh").read_text(encoding="utf-8")
    assert "check_url GRAFANA_UI_URL" in check


def test_compose_scheduler_service():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    sched = services["scheduler"]
    assert sched["image"] == services["ingestion"]["image"]  # one image for CI and production
    assert sched["entrypoint"] == ["python", "-m", "finsight.scheduler"]
    assert sched["ports"] == ["127.0.0.1:8001:8001"] and sched["mem_limit"] == "3g"
    assert ".:/app" in sched["volumes"]
    assert any(v.startswith("${DVC_REMOTE_DIR") for v in sched["volumes"])
    env = sched["environment"]
    assert env["GIT_CONFIG_VALUE_0"] == "/app" and env["HOME"] == "/tmp"  # noqa: S108
    assert sched["user"].startswith("${FINSIGHT_UID")
    assert "GITHUB" not in str(env).upper()  # local commits only, no push credentials
    backend = services["backend"]["environment"]
    assert backend["SCHEDULER_URL"] == "http://scheduler:8001" and "SCHEDULER_TOKEN" in backend
    assert "SCHEDULER_TOKEN" not in services["frontend"]["environment"]


def test_pipeline_image_has_what_the_scheduler_runs():
    dockerfile = (ROOT / "docker/Dockerfile.pipeline").read_text(encoding="utf-8")
    assert "requirements/scheduler.txt" in dockerfile and " git" in dockerfile
    reqs = (ROOT / "requirements/scheduler.txt").read_text(encoding="utf-8")
    for package in ("-r pipeline.txt", "-r api.txt", "dvc==", "pytest==", "mongomock==", "httpx=="):
        assert package in reqs


def test_serving_image_watches_the_production_alias():
    dockerfile = (ROOT / "docker/Dockerfile.serve").read_text(encoding="utf-8")
    assert "serve_watch.py" in dockerfile
    watcher = (ROOT / "docker/serve_watch.py").read_text(encoding="utf-8")
    assert "get_model_version_by_alias" in watcher and "return 0" in watcher
