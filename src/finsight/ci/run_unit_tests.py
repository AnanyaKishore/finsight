"""DVC stage ``unit_tests``: the CI gate at the head of the pipeline.

Runs the unit-test suite in-process and writes ``reports/unit_tests.json``::

    {"exit_code": 0, "passed": 70, "failed": 0, "errors": 0, "skipped": 0, "total": 70}

* A failing suite exits non-zero, so ``dvc repro`` stops **before** any data
  is exported or processed ("tests as gating stages", proposal step 2).
* The summary is deterministic (no timings, no timestamps). ``export`` lists it
  as a dependency, so a passing suite with unchanged counts does not invalidate
  the data stages, while a code change that alters the suite re-runs them.

Per-test details (JUnit XML) go to ``reports/tmp/``, which is git-ignored.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

SUMMARY_PATH = Path("reports/unit_tests.json")
JUNIT_PATH = Path("reports/tmp/unit_tests.xml")


class _OutcomeCounter:
    """Minimal pytest plugin that tallies test outcomes."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Count one outcome per test (setup errors and call results)."""
        if report.when == "call":
            self.counts[report.outcome] += 1
        elif report.when == "setup" and report.outcome != "passed":
            self.counts["skipped" if report.skipped else "errors"] += 1


def main() -> int:
    """Run ``tests/unit`` and write the summary; return pytest's exit code."""
    os.environ.setdefault("FINSIGHT_RUN_RECORDING", "0")
    counter = _OutcomeCounter()
    JUNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    exit_code = int(
        pytest.main(
            ["tests/unit", "-q", "-p", "no:cacheprovider", f"--junitxml={JUNIT_PATH}"],
            plugins=[counter],
        )
    )
    summary = {
        "exit_code": exit_code,
        "passed": counter.counts["passed"],
        "failed": counter.counts["failed"],
        "errors": counter.counts["errors"],
        "skipped": counter.counts["skipped"],
    }
    summary["total"] = sum(v for k, v in summary.items() if k != "exit_code")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
