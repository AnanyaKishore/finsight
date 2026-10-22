"""MLflow configuration and lineage capture (the "beyond autolog" part).

Every training run is tagged with everything needed to reproduce or audit it
(Module 1, Handout 3: "audit trail cheat sheet"):

=========================  ==================================================
tag                        answers
=========================  ==================================================
``git_commit``             which code produced this model?
``git_dirty_code``         was that code uncommitted? (DVC outputs excluded)
``dvc_prices_md5``         which exact processed dataset? (from ``dvc.lock``)
``export_prices_sha256``   which MongoDB snapshot was it built from?
``seed``, versions         can the run be repeated bit-for-bit?
=========================  ==================================================

Reproduce a run:  ``git checkout <git_commit> && dvc checkout && dvc repro``
(or ``mlflow run . -e train_lstm``) -- the guidelines' "Git commit hash +
MLflow run ID" requirement.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests
import yaml

from finsight.common.exceptions import FinSightError
from finsight.common.logging import get_logger

log = get_logger(__name__)

# Paths whose modification does NOT mean the *code* is dirty: they are outputs
# that DVC itself rewrites while `dvc repro` is running.
_GENERATED_PREFIXES = ("data/", "models/", "reports/", "dvc.lock", "notebooks/", "logs/")


def configure_tracking(experiment: str | None) -> str:
    """Point MLflow at the tracking server and select the experiment.

    Raises:
        FinSightError: If an HTTP tracking server is configured but unreachable.
    """
    import mlflow  # noqa: PLC0415

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if uri.startswith(("http://", "https://")):
        try:
            requests.get(f"{uri.rstrip('/')}/health", timeout=5).raise_for_status()
        except requests.RequestException as exc:
            raise FinSightError(
                f"MLflow tracking server unreachable at {uri} ({exc.__class__.__name__}). "
                "Start it with: docker compose up -d mlflow"
            ) from exc
    mlflow.set_tracking_uri(uri)
    # Under `mlflow run`, MLflow has already created the run in the experiment
    # given by --experiment-name; switching experiments here would conflict.
    if experiment and not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(experiment)
    return uri


def _git(*args: str, root: Path, strip: bool = True) -> str:
    out = subprocess.run(  # noqa: S603 -- fixed executable; arguments come from code only
        ["git", *args],  # noqa: S607
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout
    # Porcelain status lines start with a significant space (" M file"), so
    # its output must not be stripped.
    return out.strip() if strip else out


def git_state(root: Path) -> dict[str, Any]:
    """Commit, branch and whether any *code* (not DVC output) is uncommitted."""
    try:
        commit = _git("rev-parse", "HEAD", root=root)
        branch = _git("rev-parse", "--abbrev-ref", "HEAD", root=root)
        status = _git("status", "--porcelain", "--untracked-files=no", root=root, strip=False)
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "git_branch": "unknown", "git_dirty_code": "unknown"}
    dirty = [
        line[3:]
        for line in status.splitlines()
        if line[3:] and not line[3:].startswith(_GENERATED_PREFIXES)
    ]
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty_code": str(bool(dirty)).lower(),
        "git_dirty_files": ",".join(dirty)[:500],
    }


def dvc_output_md5(root: Path, stage: str, path: str) -> str | None:
    """Content hash DVC recorded for one stage output (``None`` if unknown)."""
    lock = root / "dvc.lock"
    if not lock.exists():
        return None
    try:
        data = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    for out in data.get("stages", {}).get(stage, {}).get("outs", []):
        if out.get("path") == path:
            return out.get("md5")
    return None


def lineage_tags(root: Path, processed_prices: str, seed: int) -> dict[str, str]:
    """All reproducibility tags for a training run."""
    import mlflow  # noqa: PLC0415
    import torch  # noqa: PLC0415

    tags: dict[str, str] = {k: str(v) for k, v in git_state(root).items()}
    tags["dvc_prices_md5"] = str(dvc_output_md5(root, "process", processed_prices))
    manifest = root / "reports" / "export_manifest.json"
    if manifest.exists():
        tags["export_prices_sha256"] = json.loads(manifest.read_text())["prices"]["sha256"]
    tags.update(
        {
            "seed": str(seed),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__,
            "platform": platform.platform(),
            "cpu_count": str(os.cpu_count()),
            "entry_point": " ".join(sys.argv)[:250],
        }
    )
    return tags
