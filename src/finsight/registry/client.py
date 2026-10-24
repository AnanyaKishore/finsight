"""Thin, idempotent helpers over ``MlflowClient`` for the registry.

Both naming schemes are maintained (DESIGN_DECISIONS D-08):

* **aliases** ``@production`` / ``@staging`` -- what serving and the API load
  (``models:/finsight-lstm@production``);
* classic **stages** ``Production`` / ``Staging`` / ``Archived`` -- the UI view
  described in the proposal. MLflow 3 deprecates stages, so any failure there is
  logged and ignored (aliases are the source of truth).

The registered model keeps a ``production_history`` tag (JSON list of
versions) so a rollback knows what "previous" means.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

from finsight.common.logging import get_logger
from finsight.registry.gate import PRODUCTION, STAGING

log = get_logger(__name__)

ROLLED_BACK = "rolled_back"


def get_client() -> Any:
    """``MlflowClient`` for the configured tracking URI (already set by the caller)."""
    from mlflow import MlflowClient  # noqa: PLC0415

    return MlflowClient()


def ensure_registered_model(client: Any, name: str, description: str) -> None:
    """Create the registered model if it does not exist yet."""
    from mlflow.exceptions import MlflowException  # noqa: PLC0415

    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name, description=description)
        log.info("registered model created", extra={"model": name})


def ensure_version(
    client: Any, name: str, model_uri: str, run_id: str, tags: dict[str, Any]
) -> Any:
    """Register ``model_uri`` once per run (re-running the stage reuses the version)."""
    existing = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    if existing:
        version = max(existing, key=lambda v: int(v.version))
    else:
        import mlflow  # noqa: PLC0415

        # register_model resolves MLflow 3 logged-model URIs (models:/m-...) to
        # their model_id, which the REST registry requires; create_model_version
        # with a bare source only works against a local store.
        version = mlflow.register_model(model_uri, name)
        log.info("model version registered", extra={"model": name, "version": version.version})
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, str(value))
    return client.get_model_version(name, version.version)


def alias_version(client: Any, name: str, alias: str) -> Any | None:
    """Model version holding ``alias``, or ``None``."""
    from mlflow.exceptions import MlflowException  # noqa: PLC0415

    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def is_rolled_back(version: Any) -> bool:
    """Whether a version was rolled back (never auto-promoted again)."""
    return (version.tags or {}).get(ROLLED_BACK) == "true"


def history(client: Any, name: str) -> list[str]:
    """Versions that have held ``@production``, oldest first."""
    tags = client.get_registered_model(name).tags or {}
    try:
        return [str(v) for v in json.loads(tags.get("production_history", "[]"))]
    except json.JSONDecodeError:
        return []


def _transition(client: Any, name: str, version: str, stage: str) -> None:
    try:
        with warnings.catch_warnings():  # stages are deprecated in MLflow 3
            warnings.simplefilter("ignore")
            client.transition_model_version_stage(
                name, version, stage, archive_existing_versions=stage == "Production"
            )
    except Exception as exc:  # noqa: BLE001 -- stages are cosmetic; aliases rule
        log.warning("stage transition skipped", extra={"model": name, "error": str(exc)})


def set_staging(client: Any, name: str, version: str) -> None:
    """Mark ``version`` as the latest challenger."""
    client.set_registered_model_alias(name, STAGING, version)
    _transition(client, name, version, "Staging")


def set_production(client: Any, name: str, version: str) -> None:
    """Point ``@production`` at ``version`` and append it to the history."""
    client.set_registered_model_alias(name, PRODUCTION, version)
    _transition(client, name, version, "Production")
    past = history(client, name)
    if not past or past[-1] != str(version):
        past.append(str(version))
    client.set_registered_model_tag(name, "production_history", json.dumps(past))
    log.info("production alias moved", extra={"model": name, "version": version})
