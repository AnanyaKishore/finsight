"""The DVC pipeline as data, for the console's DAG diagram.

``dvc dag --dot`` gives the stages and edges; ``dvc status --json`` gives the
stages whose inputs changed since they last ran ("stale"). The scheduler
stores the result in MongoDB after every run, so the backend (which has no
DVC) can serve it.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_NODE = re.compile(r'^\s*"([^"]+)";\s*$')
_EDGE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)";\s*$')


def parse_dot(dot: str) -> tuple[list[str], list[list[str]]]:
    """Nodes (in DVC's order) and edges from ``dvc dag --dot`` output."""
    nodes: list[str] = []
    edges: list[list[str]] = []
    for line in dot.splitlines():
        if m := _EDGE.match(line):
            edges.append([m.group(1), m.group(2)])
            for name in m.groups():
                if name not in nodes:
                    nodes.append(name)
        elif m := _NODE.match(line):
            if m.group(1) not in nodes:
                nodes.append(m.group(1))
    return nodes, edges


def stale_stages(status: Mapping[str, Any]) -> list[str]:
    """Stages with changed dependencies or outputs (``always changed`` alone is not stale)."""
    stale = []
    for stage, changes in status.items():
        real = [c for c in changes if c != "always changed"]
        if real:
            stale.append(stage)
    return sorted(stale)


def build_dag(
    cwd: Path, env: Mapping[str, str] | None = None, timeout_s: float = 300
) -> dict[str, Any]:
    """Run DVC and return ``{nodes, edges, stale, generated_at}``."""

    def dvc(*args: str) -> str:
        return subprocess.run(  # noqa: S603 -- fixed dvc commands
            ["dvc", *args],  # noqa: S607
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_s,
        ).stdout

    nodes, edges = parse_dot(dvc("dag", "--dot"))
    try:
        stale = stale_stages(json.loads(dvc("status", "--json") or "{}"))
    except (subprocess.SubprocessError, json.JSONDecodeError):
        stale = []  # the diagram is still useful without the stale markers
    return {"nodes": nodes, "edges": edges, "stale": stale, "generated_at": datetime.now(UTC)}
