"""Day-1 preflight: verify the local environment before the first backfill.

    python scripts/check_env.py

Checks (no MarketStack quota is spent):
  1. Python version is 3.11.x
  2. Key packages import
  3. .env exists and has no placeholder values
  4. MongoDB answers a ping with the configured credentials
  5. Finnhub key works (one free company-news call; Finnhub is rate-limited, not metered)
  6. git, git-lfs, dvc and docker are on PATH; a DVC remote is configured
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OK, BAD, WARN = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"
failures = 0


def report(ok: bool | None, label: str, detail: str = "") -> None:
    global failures
    status = OK if ok else (WARN if ok is None else BAD)
    failures += ok is False
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    report(sys.version_info[:2] == (3, 11), "Python 3.11", sys.version.split()[0])

    for module in (
        "requests",
        "pymongo",
        "yaml",
        "dotenv",
        "pandas",
        "yfinance",
        "pyspark",
        "torch",
        "mlflow",
    ):
        try:
            __import__(module)
            report(True, f"import {module}")
        except ImportError as exc:
            report(False, f"import {module}", str(exc))

    env_file = ROOT / ".env"
    if not env_file.exists():
        report(False, ".env present", "run: cp .env.example .env  and fill it in")
    else:
        placeholders = [
            line.split("=", 1)[0]
            for line in env_file.read_text().splitlines()
            if line.strip().endswith("=replace-me")
        ]
        report(not placeholders, ".env filled in", ", ".join(placeholders) or "all set")

    from finsight.common.config import load_settings
    from finsight.common.exceptions import ConfigError, FinSightError
    from finsight.common.mongo import get_client

    settings = load_settings(ROOT)
    try:
        client = get_client(settings, timeout_ms=3000)
        report(True, "MongoDB ping", f"{settings.mongo_host}:{settings.mongo_port}")
        client.close()
    except FinSightError as exc:
        report(False, "MongoDB ping", str(exc))

    try:
        from finsight.common.http import HttpClient

        token = settings.require("finnhub_api_key")
        today = date.today()
        payload = HttpClient("finnhub", max_retries=1).get_json(
            f"{settings.finnhub_base_url}/company-news",
            {
                "symbol": "AAPL",
                "from": str(today - timedelta(days=3)),
                "to": str(today),
                "token": token,
            },
        )
        report(isinstance(payload, list), "Finnhub key", f"{len(payload)} recent AAPL articles")
    except (ConfigError, FinSightError) as exc:
        report(False, "Finnhub key", str(exc))

    try:
        import os

        import requests

        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        requests.get(f"{uri}/health", timeout=3).raise_for_status()
        report(True, "MLflow tracking server", uri)
    except Exception as exc:  # noqa: BLE001
        report(
            False,
            "MLflow tracking server",
            f"{exc.__class__.__name__}: docker compose up -d mlflow",
        )
    try:
        import torch

        cuda = torch.version.cuda
        report(
            cuda is None,
            "PyTorch CPU build",
            f"{torch.__version__}"
            + ("" if cuda is None else " (CUDA build: works, but ~2 GB larger than needed)"),
        )
    except ImportError:
        pass

    try:
        settings.require("marketstack_api_key")
        report(True, "MarketStack key present", "not called here (every call costs quota)")
    except ConfigError as exc:
        report(None, "MarketStack key", f"{exc}; prices will use the yfinance fallback")

    java = shutil.which("java")
    if java:
        version = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, check=False
        ).stderr
        line = next((ln for ln in version.splitlines() if "version" in ln), version.strip()[:60])
        major = line.split('"')[1].split(".")[0] if '"' in line else "?"
        report(major.isdigit() and int(major) >= 17, "Java 17+ (for Spark)", line.strip())
    else:
        report(False, "Java 17+ (for Spark)", "not on PATH: mamba install -n finsight openjdk=17")

    for tool in ("git", "git-lfs", "dvc", "docker"):
        report(shutil.which(tool) is not None, f"{tool} on PATH")
    if shutil.which("dvc"):
        remotes = subprocess.run(
            ["dvc", "remote", "list"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        report(bool(remotes), "DVC remote configured", remotes or "run: dvc remote add -d ...")

    print(f"\n{failures} check(s) failed." if failures else "\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
