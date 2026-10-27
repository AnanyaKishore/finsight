"""Running one command step: output to a log file, a tail kept, a hard timeout.

Each step runs as a child process in its own process group, so a timeout
stops the whole tree (``dvc repro`` starts Python stages, which start Spark's
JVM). The full output goes to ``logs/scheduler/<run_id>.log``; the last lines
are kept in the run record so a failure can be read in the console without
opening a terminal.
"""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepResult:
    """Outcome of one step."""

    status: str  # success | partial | failed | skipped
    duration_s: float = 0.0
    exit_code: int | None = None
    detail: str = ""
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
    log_path: Path,
    tail_lines: int = 40,
    partial_exit_codes: frozenset[int] = frozenset(),
) -> StepResult:
    """Run ``argv``; success on exit 0, partial on ``partial_exit_codes``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: collections.deque[str] = collections.deque(maxlen=tail_lines)
    started = time.monotonic()
    timed_out = threading.Event()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(argv)}\n")
        log.flush()
        try:
            proc = subprocess.Popen(  # noqa: S603 -- argv comes from the fixed plan
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,  # own process group: a timeout kills the tree
            )
        except OSError as exc:
            return StepResult("failed", error=f"could not start {argv[0]}: {exc}")

        def _kill() -> None:
            timed_out.set()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(10)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        watchdog = threading.Timer(timeout_s, _kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            for line in proc.stdout or ():
                log.write(line)
                tail.append(line.rstrip("\n")[:500])
            code = proc.wait()
        finally:
            watchdog.cancel()
    duration = round(time.monotonic() - started, 2)
    if timed_out.is_set():
        return StepResult(
            "failed",
            duration,
            code,
            error=f"stopped after the {timeout_s:.0f} s time limit",
            log_tail=list(tail),
        )
    if code == 0:
        return StepResult("success", duration, code, log_tail=list(tail))
    if code in partial_exit_codes:
        return StepResult(
            "partial", duration, code, detail="finished with some errors", log_tail=list(tail)
        )
    last = next((line for line in reversed(tail) if line.strip()), "")
    return StepResult(
        "failed", duration, code, error=f"exit code {code}: {last}"[:500], log_tail=list(tail)
    )
