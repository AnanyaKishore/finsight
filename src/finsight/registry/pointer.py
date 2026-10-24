"""Resolve the production model from ``reports/promotion.json``.

``score_sentiment`` (and anything else in the pipeline) asks this module for
the production model instead of hard-coding a folder. The local packaged copy
is used when it *is* the production version (same MLflow run); otherwise --
after a rollback, for instance -- the exact registered version is downloaded
from the registry into ``models/registry_cache/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finsight.common.exceptions import FinSightError
from finsight.common.logging import get_logger

log = get_logger(__name__)


def load_promotion(path: Path) -> dict[str, Any]:
    """Read the promotion report.

    Raises:
        FinSightError: If the file is missing (run ``dvc repro register`` first).
    """
    if not path.exists():
        raise FinSightError(f"{path} not found: run `dvc repro register` first")
    return json.loads(path.read_text(encoding="utf-8"))


def production_entry(promotion: dict[str, Any], family: str) -> dict[str, Any]:
    """Production pointer for ``lstm`` or ``sentiment``.

    Raises:
        FinSightError: If no version holds ``@production``.
    """
    entry = (promotion.get(family) or {}).get("production")
    if not entry:
        raise FinSightError(f"no production {family} model in the promotion report")
    return entry


def local_model_dir(
    entry: dict[str, Any], local_dir: Path, cache_root: Path, downloader: Any = None
) -> Path:
    """Directory holding the production pyfunc model.

    Args:
        entry: Production pointer (``run_id``, ``registered_name``, ``version``).
        local_dir: Packaged candidate directory (``model/`` + ``run.json``).
        cache_root: Where registry downloads are kept.
        downloader: Injected ``download_artifacts`` (tests).
    """
    run_json = local_dir / "run.json"
    if run_json.exists() and json.loads(run_json.read_text())["run_id"] == entry["run_id"]:
        return local_dir / "model"
    target = cache_root / entry["registered_name"] / f"v{entry['version']}"
    if not (target / "MLmodel").exists():
        if downloader is None:
            from mlflow.artifacts import download_artifacts as downloader  # noqa: PLC0415,N813
        log.info("downloading production model", extra={"uri": entry["model_uri"]})
        target.mkdir(parents=True, exist_ok=True)
        downloader(artifact_uri=entry["model_uri"], dst_path=str(target))
    return target
