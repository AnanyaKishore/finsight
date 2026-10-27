"""After a run: push DVC outputs to the local remote and commit ``dvc.lock``.

The commit is **local only** -- the scheduler never pushes to GitHub, so no
Git credentials ever enter a container. It commits only ``dvc.lock`` and
``reports/`` (with a pathspec, so anything else you have staged is left
alone), and it skips the commit when the repository is mid-merge or mid-rebase,
or HEAD is detached.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

COMMIT_PATHS = ("dvc.lock", "reports")


def _git(
    cwd: Path, env: Mapping[str, str], *args: str, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 -- fixed git commands
        ["git", *args],  # noqa: S607
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        check=check,
        timeout=120,
    )


def repo_busy(cwd: Path) -> str | None:
    """Why a commit would be unsafe right now, or ``None``."""
    git_dir = cwd / ".git"
    for marker, reason in (
        ("MERGE_HEAD", "a merge is in progress"),
        ("rebase-merge", "a rebase is in progress"),
        ("rebase-apply", "a rebase is in progress"),
        ("CHERRY_PICK_HEAD", "a cherry-pick is in progress"),
    ):
        if (git_dir / marker).exists():
            return reason
    return None


def commit_pipeline_outputs(cwd: Path, env: Mapping[str, str], message: str) -> str:
    """Commit ``dvc.lock`` + ``reports/`` if they changed; return what happened."""
    if not (cwd / ".git").exists():
        return "skipped: not a Git repository"
    if reason := repo_busy(cwd):
        return f"skipped: {reason}"
    head = _git(cwd, env, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if head == "HEAD":
        return "skipped: HEAD is detached"
    paths = [p for p in COMMIT_PATHS if (cwd / p).exists()]
    if not paths:
        return "skipped: nothing to commit"
    _git(cwd, env, "add", "-A", "--", *paths)
    if _git(cwd, env, "diff", "--cached", "--quiet", "--", *paths, check=False).returncode == 0:
        return "no changes to commit"
    # --no-verify: this commit only records pipeline outputs, and the container
    # does not have the developer's pre-commit hook environment.
    _git(cwd, env, "commit", "--no-verify", "-q", "-m", message, "--", *paths)
    sha = _git(cwd, env, "rev-parse", "--short", "HEAD").stdout.strip()
    return f"committed {sha} on {head}"


def dvc_push(cwd: Path, env: Mapping[str, str], timeout_s: float = 900) -> str:
    """``dvc push`` to the default remote; returns DVC's last output line."""
    result = subprocess.run(  # noqa: S603
        ["dvc", "push", "-q"],  # noqa: S607
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_s,
    )
    lines = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    return lines[-1] if lines else "pushed"
