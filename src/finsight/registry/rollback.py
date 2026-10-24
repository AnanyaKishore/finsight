"""Roll ``@production`` back to an earlier model version.

    python -m finsight.registry.rollback --model lstm --reason "forecasts drifting"
    python -m finsight.registry.rollback --model sentiment --to-version 1 --reason "..."

What happens:

1. The current production version is tagged ``rolled_back=true`` (with the
   reason), so the ``register`` gate never promotes it again automatically
   (``--keep-eligible`` skips the mark, for drills and demos).
2. ``@production`` (and the ``Production`` stage) move to ``--to-version`` or,
   by default, the most recent earlier version in ``production_history`` that
   was not itself rolled back.
3. ``reports/promotion.json`` is updated, so the next ``dvc repro`` re-runs
   whatever depends on the production model (e.g. ``score_sentiment``).
4. A ``pipeline_runs`` record (stage ``rollback``) is written for the Pipeline Console.

Serving containers load ``@production`` at start-up, so restart them afterwards
(the command prints the exact ``docker compose restart`` line).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mlflow.exceptions import MlflowException

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.runs import stage_run
from finsight.registry import client as reg
from finsight.registry.gate import PRODUCTION

log = get_logger(__name__)

FAMILIES = {"lstm": ("lstm_name", "lstm-serve"), "sentiment": ("sentiment_name", "sentiment-serve")}


def choose_target(client: Any, name: str, current: str, to_version: str | None) -> str:
    """Version to roll back to.

    Raises:
        FinSightError: If no earlier eligible version exists.
    """
    if to_version is not None:
        version = client.get_model_version(name, str(to_version))
        if str(version.version) == str(current):
            raise FinSightError(f"version {current} is already production")
        return str(version.version)
    for candidate in reversed(reg.history(client, name)):
        if candidate == str(current):
            continue
        if not reg.is_rolled_back(client.get_model_version(name, candidate)):
            return candidate
    raise FinSightError(f"no earlier production version of {name} to roll back to")


def rollback(
    client: Any,
    name: str,
    reason: str,
    to_version: str | None = None,
    mark: bool = True,
) -> dict[str, Any]:
    """Perform the rollback in the registry; returns what changed.

    Raises:
        FinSightError: If there is no production version or no target.
    """
    current = reg.alias_version(client, name, PRODUCTION)
    if current is None:
        raise FinSightError(f"{name} has no production version")
    target = choose_target(client, name, str(current.version), to_version)
    if reg.is_rolled_back(client.get_model_version(name, target)):
        # An operator explicitly restoring a version clears its earlier rollback mark.
        client.delete_model_version_tag(name, target, reg.ROLLED_BACK)
    if mark:
        client.set_model_version_tag(name, str(current.version), reg.ROLLED_BACK, "true")
    client.set_model_version_tag(name, str(current.version), "rollback_reason", reason[:250])
    reg.set_production(client, name, target)
    log.warning(
        "production rolled back",
        extra={"model": name, "from": current.version, "to": target, "reason": reason},
    )
    return {"model": name, "from_version": str(current.version), "to_version": target}


def update_report(path: Path, family: str, client: Any, name: str, change: dict) -> None:
    """Rewrite the production pointer for ``family`` in ``reports/promotion.json``."""
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    version = client.get_model_version(name, change["to_version"])
    tags = version.tags or {}
    pointer = {
        "registered_name": name,
        "version": str(version.version),
        "run_id": version.run_id,
        "model_uri": f"models:/{name}/{version.version}",
    }
    if family == "sentiment":
        pointer |= {
            "candidate": tags.get("candidate"),
            "model_version": tags.get("model_version"),
            "neutral_band": float(tags.get("neutral_band", 0.0)),
        }
    entry = report.setdefault(family, {})
    entry["production"] = pointer
    entry["decision"] = {"action": "rollback", **change}
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Roll a production model back")
    parser.add_argument("--model", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--to-version", default=None, help="default: previous production version")
    parser.add_argument("--reason", required=True, help="recorded on the rolled-back version")
    parser.add_argument(
        "--keep-eligible",
        action="store_true",
        help="drills/demos: do not mark the current version rolled_back (the gate may "
        "promote it again)",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("rollback", settings.log_level, settings.root / "logs")
    registry = settings.section("registry")
    name_key, service = FAMILIES[args.model]
    name = registry[name_key]
    try:
        from finsight.models.tracking import configure_tracking  # noqa: PLC0415

        with stage_run("rollback", settings) as run:
            configure_tracking(None)
            client = reg.get_client()
            change = rollback(
                client, name, args.reason, args.to_version, mark=not args.keep_eligible
            )
            update_report(settings.root / registry["report"], args.model, client, name, change)
            run.metrics = change | {"reason": args.reason}
    except (FinSightError, MlflowException) as exc:
        print(f"rollback failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(change, indent=2))
    print(f"\nNext: docker compose restart {service}", end="")
    print("  &&  dvc repro score_sentiment" if args.model == "sentiment" else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
