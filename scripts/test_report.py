"""Run every test suite and write the test report and the test-case catalogue (Day 9).

    python scripts/test_report.py                 # unit + integration + e2e + rule tests
    python scripts/test_report.py --skip-e2e      # when the stack is not running

Run ``python scripts/load_test.py`` first if you want fresh latency figures:
this script reads ``reports/load_test.json`` rather than loading the API itself.

Outputs:

* ``TEST_REPORT.md``  totals (passed / failed / skipped per suite),
  coverage, the acceptance criteria of ``TEST_PLAN.md`` with the measured
  value and PASS / FAIL, and every failed test.
* ``TEST_CASES.md``   every test case with an ID, what it checks and its
  result in this run (the "enlistment of test cases").

Test descriptions come from each test's docstring or, failing that, its name
(``test_wrong_key_is_refused`` -> "Wrong key is refused"); files are parsed,
not imported. Exit code 0 only when every acceptance criterion passes.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET  # noqa: N817,S405 -- parses our own pytest output
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "reports" / "tmp"
SUITES = {
    "unit": {"prefix": "UT", "args": ["tests/unit"]},
    "integration": {"prefix": "IT", "args": ["tests/integration", "-m", "integration"]},
    "e2e": {"prefix": "E2E", "args": ["tests/e2e", "-m", "e2e"]},
}
SCHEDULED_TRIGGERS = {"schedule", "catch-up"}


# --------------------------------------------------------------------- tests
def run_suite(name: str, coverage: bool) -> Path:
    """Run one pytest suite in a subprocess; return its JUnit XML path."""
    junit = TMP / f"junit_{name}.xml"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *SUITES[name]["args"],
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junitxml={junit}",
    ]
    if coverage:
        cmd += ["--cov=finsight", f"--cov-report=json:{TMP / 'coverage.json'}", "--cov-report="]
    env = {**os.environ, "FINSIGHT_RUN_RECORDING": "0"}
    print(f"\n=== {name} tests ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=False)  # noqa: S603
    return junit


def describe(path: Path) -> tuple[str, dict[str, str]]:
    """Module summary and ``{test name: one-line description}`` from the source."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module = (ast.get_docstring(tree) or path.stem).splitlines()[0]
    tests: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            doc = ast.get_docstring(node)
            text = doc.splitlines()[0] if doc else node.name[5:].replace("_", " ").capitalize()
            tests[node.name] = text.rstrip(".")
    return module, tests


def parse_junit(junit: Path) -> list[dict[str, str]]:
    """One row per test case: file, name, outcome, message."""
    if not junit.exists():
        return []
    rows = []
    for case in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = (node.get("message") or "").splitlines()[0][:160]
                if message == "collection skipped":  # whole module skipped: reason is in the text
                    found = re.search(r"Skipped: (.+?)[\"']?\)?$", (node.text or "").strip())
                    message = f"whole file skipped: {found.group(1) if found else 'see log'}"
                break
        name, classname = case.get("name", ""), case.get("classname", "")
        if not classname:  # a module that could not be collected or was skipped as a whole
            classname, name = name, "(whole file)"
        parts = classname.split(".")
        module = next(
            (i for i, part in enumerate(parts) if part.startswith("test_")), len(parts) - 1
        )
        file = "/".join(parts[: module + 1]) + ".py"
        rows.append({"file": file, "name": name, "outcome": outcome, "message": message})
    return rows


def promtool_rule_tests() -> dict[str, Any]:
    """Alert-rule unit tests: a local ``promtool``, else the one in the Prometheus container."""
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "-w",
        "/etc/prometheus/tests",
        "prometheus",
        "promtool",
        "test",
        "rules",
        "finsight_test.yml",
    ]
    cwd = ROOT
    if shutil.which("promtool"):  # a local install is used when there is one
        cmd, cwd = [
            "promtool",
            "test",
            "rules",
            "finsight_test.yml",
        ], ROOT / "monitoring/prometheus/tests"
    try:
        done = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=60, check=False
        )  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ran": False, "detail": str(exc)}
    if done.returncode != 0 and "SUCCESS" not in done.stdout and "FAILED" not in done.stdout:
        noise = ("level=warning", "variable is not set")
        lines = [
            ln
            for ln in (done.stderr or done.stdout).splitlines()
            if not any(n in ln for n in noise)
        ]
        return {"ran": False, "detail": " ".join(lines).strip()[:200] or "prometheus not running"}
    return {"ran": True, "passed": done.returncode == 0, "detail": done.stdout.strip()[-200:]}


# ------------------------------------------------------------------- metrics
def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pipeline_success(api: str, days: int = 30) -> dict[str, Any] | None:
    """Success rate of *scheduled* runs (console drills excluded) from the backend."""
    try:
        runs = requests.get(f"{api}/api/pipeline/runs", params={"limit": 200}, timeout=10).json()[
            "runs"
        ]
    except (requests.RequestException, ValueError, KeyError):
        return None
    since = datetime.now(UTC) - timedelta(days=days)
    scheduled = [
        r
        for r in runs
        if r.get("trigger") in SCHEDULED_TRIGGERS
        and r.get("status") != "running"
        and datetime.fromisoformat(r["started_at"].replace("Z", "+00:00")) >= since
    ]
    ok = sum(r["status"] in {"success", "partial"} for r in scheduled)
    return {
        "runs": len(scheduled),
        "ok": ok,
        "rate": ok / len(scheduled) if scheduled else None,
        "by_status": dict(Counter(r["status"] for r in scheduled)),
    }


def criterion(cid: str, what: str, target: str, value: Any, ok: bool | None) -> dict[str, Any]:
    return {
        "id": cid,
        "what": what,
        "target": target,
        "value": value,
        "result": "PASS" if ok else ("FAIL" if ok is False else "NOT RUN"),
    }


def acceptance(counts: dict[str, Counter], api: str, rules: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate AC-01 ... AC-12 of TEST_PLAN.md."""
    out = []
    for cid, suite, what in (
        ("AC-01", "unit", "Unit tests"),
        ("AC-02", "integration", "Integration tests"),
        ("AC-03", "e2e", "End-to-end smoke tests"),
    ):
        c = counts.get(suite)
        if not c or not (c["passed"] + c["failed"] + c["error"]):
            out.append(criterion(cid, what, "all pass", "not run", None))
        else:
            bad = c["failed"] + c["error"]
            out.append(
                criterion(cid, what, "all pass", f"{c['passed']} passed, {bad} failed", bad == 0)
            )
    out.append(
        criterion(
            "AC-04",
            "Alert-rule unit tests (promtool)",
            "all pass",
            rules.get("detail", "")[-60:] or "not run",
            rules.get("passed") if rules.get("ran") else None,
        )
    )

    lstm = read_json(ROOT / "reports" / "lstm_metrics.json")
    if lstm:
        model, naive = lstm["model"]["mape"], lstm["naive"]["mape"]
        out.append(
            criterion(
                "AC-05",
                "LSTM test MAPE vs naive (held-out windows)",
                "≤ 1.02 × naive",
                f"{model:.3f}% vs {naive:.3f}% ({model / naive:.3f}×)",
                model <= 1.02 * naive,
            )
        )
    else:
        out.append(
            criterion("AC-05", "LSTM test MAPE vs naive", "≤ 1.02 × naive", "no report", None)
        )

    promotion = read_json(ROOT / "reports" / "promotion.json") or {}
    candidate = ((promotion.get("sentiment") or {}).get("production") or {}).get("candidate")
    sent = read_json(ROOT / "reports" / f"sentiment_eval_{candidate}.json") if candidate else None
    if sent:
        f1, base = sent["gate_macro_f1"], sent["gate_baseline_macro_f1"]
        out.append(
            criterion(
                "AC-06",
                f"Sentiment macro-F1 in production ({candidate}) vs majority baseline",
                "> baseline",
                f"{f1:.3f} vs {base:.3f}",
                f1 > base,
            )
        )
    else:
        out.append(
            criterion(
                "AC-06", "Sentiment macro-F1 vs majority baseline", "> baseline", "no report", None
            )
        )

    portfolio = read_json(ROOT / "reports" / "portfolio.json")
    if portfolio and portfolio.get("grid_sharpe_gap") is not None:
        gap = portfolio["grid_sharpe_gap"]
        out.append(
            criterion(
                "AC-07",
                "Portfolio grid optimum vs continuous (SLSQP) optimum",
                "Sharpe gap ≤ 0.05",
                f"{gap:.4f} (out-of-sample Sharpe {portfolio['oos_sharpe_max_sharpe']:.2f} vs "
                f"equal weight {portfolio['oos_sharpe_equal_weight']:.2f}, reported)",
                gap <= 0.05,
            )
        )
    else:
        out.append(
            criterion(
                "AC-07",
                "Portfolio grid vs continuous optimum",
                "Sharpe gap ≤ 0.05",
                "no report",
                None,
            )
        )

    load = read_json(ROOT / "reports" / "load_test.json")
    if load:
        a, label = load["acceptance"], f"{load['users']} users, {load['duration_s']} s"
        out.append(
            criterion(
                "AC-08",
                f"API p95 latency, cached endpoints ({label})",
                a["cached_p95_ms"]["target"] + " ms",
                f"{a['cached_p95_ms']['value']} ms",
                a["cached_p95_ms"]["pass"],
            )
        )
        out.append(
            criterion(
                "AC-09",
                "API error rate under load",
                a["error_rate"]["target"],
                f"{a['error_rate']['value']:.2%}",
                a["error_rate"]["pass"],
            )
        )
        out.append(
            criterion(
                "AC-10",
                "Live sentiment model endpoint p95",
                a["model_p95_ms"]["target"] + " ms",
                f"{a['model_p95_ms']['value']} ms",
                a["model_p95_ms"]["pass"],
            )
        )
    else:
        for cid, what in (
            ("AC-08", "API p95 latency"),
            ("AC-09", "API error rate"),
            ("AC-10", "Model endpoint p95"),
        ):
            out.append(criterion(cid, what, "see plan", "run scripts/load_test.py", None))

    runs = pipeline_success(api)
    if runs and runs["rate"] is not None:
        out.append(
            criterion(
                "AC-11",
                "Scheduled pipeline runs succeeded (last 30 days)",
                "≥ 90 %",
                f"{runs['ok']}/{runs['runs']} ({runs['rate']:.0%})",
                runs["rate"] >= 0.9,
            )
        )
    else:
        out.append(
            criterion(
                "AC-11",
                "Scheduled pipeline runs succeeded",
                "≥ 90 %",
                "backend not reachable",
                None,
            )
        )

    quality = read_json(ROOT / "reports" / "data_quality.json")
    if quality:
        errors = [
            c["name"]
            for c in quality.get("checks", [])
            if not c.get("success", True) and c.get("severity") == "error"
        ]
        out.append(
            criterion(
                "AC-12",
                "Data-quality checks of the latest validation",
                "no blocking failure",
                f"{len(quality.get('checks', []))} checks, {len(errors)} blocking failures",
                not errors,
            )
        )
    else:
        out.append(
            criterion("AC-12", "Data-quality checks", "no blocking failure", "no report", None)
        )
    return out


# -------------------------------------------------------------------- output
def git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                capture_output=True,  # noqa: S607
                text=True,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        return "unknown"


def write_reports(
    cases: list[dict[str, Any]],
    counts: dict[str, Counter],
    criteria: list[dict[str, Any]],
    coverage: float | None,
) -> None:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    head = f"Generated by `python scripts/test_report.py` on {now}, commit `{git_commit()}`, Python {platform.python_version()}."
    total = Counter()
    for c in counts.values():
        total.update(c)
    lines = [
        "# FinSight — Test report",
        "",
        head,
        "",
        "Plan, scope and acceptance criteria: [TEST_PLAN.md](TEST_PLAN.md). Every test case with its result: "
        "[TEST_CASES.md](TEST_CASES.md).",
        "",
        "## 1. Summary",
        "",
        "| Suite | Test cases | Passed | Failed | Errors | Skipped |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for suite in SUITES:
        c = counts.get(suite, Counter())
        n = sum(c.values())
        lines.append(
            f"| {suite} | {n} | {c['passed']} | {c['failed']} | {c['error']} | {c['skipped']} |"
        )
    lines.append(
        f"| **total** | **{sum(total.values())}** | **{total['passed']}** | **{total['failed']}** | "
        f"**{total['error']}** | **{total['skipped']}** |"
    )
    if coverage is not None:
        lines += ["", f"Line coverage of `src/finsight` by the unit tests: **{coverage:.1f} %**."]
    skipped = [c for c in cases if c["outcome"] == "skipped"]
    if skipped:
        reasons = Counter(c["message"] or "no reason given" for c in skipped)
        lines += ["", "Skipped tests and why:", ""] + [
            f"- {n} × {r}" for r, n in reasons.most_common()
        ]
    met = sum(c["result"] == "PASS" for c in criteria)
    lines += [
        "",
        "## 2. Acceptance criteria",
        "",
        f"**{met} of {len(criteria)} criteria met.**",
        "",
        "| ID | Criterion | Target | Measured | Result |",
        "|---|---|---|---|---|",
    ]
    for c in criteria:
        lines.append(
            f"| {c['id']} | {c['what']} | {c['target']} | {c['value']} | **{c['result']}** |"
        )
    failed = [c for c in cases if c["outcome"] in {"failed", "error"}]
    lines += ["", "## 3. Failed tests", ""]
    lines += [f"- `{c['id']}` {c['file']}::{c['name']} — {c['message']}" for c in failed] or [
        "None."
    ]
    lines += [
        "",
        "## 4. Coverage by area",
        "",
        "| Area (test file) | Cases | Passed | Failed | Skipped |",
        "|---|---:|---:|---:|---:|",
    ]
    by_file: dict[str, Counter] = {}
    for c in cases:
        by_file.setdefault(c["file"], Counter())[c["outcome"]] += 1
    for file, c in by_file.items():
        lines.append(
            f"| `{file}` | {sum(c.values())} | {c['passed']} | {c['failed'] + c['error']} | {c['skipped']} |"
        )
    notes = ROOT / "TEST_NOTES.md"
    if notes.exists():  # hand-written context for a result (kept across regenerations)
        body = [
            ln for ln in notes.read_text(encoding="utf-8").splitlines() if not ln.startswith("# ")
        ]
        lines += [
            "",
            "## 5. Notes",
            "",
            "Written by hand.",
            *body,
        ]
    (ROOT / "TEST_REPORT.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )

    cat = [
        "# FinSight — Test cases",
        "",
        head,
        "",
        "Every automated test case, grouped by suite and area. IDs are assigned in file order; the "
        "description is the test's docstring or its name. Result is from the run above.",
        "",
    ]
    current_suite, current_file = None, None
    for i, c in enumerate(cases):
        if c["suite"] != current_suite:
            current_suite, current_file = c["suite"], None
            cat += [f"## {current_suite.capitalize()} tests", ""]
        if c["file"] != current_file:
            current_file = c["file"]
            cat += [
                f"### `{current_file}` — {c['area']}",
                "",
                "| ID | Test case | Result |",
                "|---|---|---|",
            ]
        params = f" `{c['name'][len(c['func']):]}`" if c["name"] != c["func"] else ""
        cat.append(f"| {c['id']} | {c['what']}{params} | {c['outcome']} |")
        if i + 1 == len(cases) or cases[i + 1]["file"] != current_file:
            cat.append("")  # a blank line closes the table
    (ROOT / "TEST_CASES.md").write_text("\n".join(cat).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the suites, evaluate the criteria and write both documents."""
    parser = argparse.ArgumentParser(description="FinSight test report")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-integration", action="store_true")
    parser.add_argument(
        "--api", default=os.environ.get("FINSIGHT_API_URL", "http://localhost:8000")
    )
    args = parser.parse_args(argv)
    TMP.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    counts: dict[str, Counter] = {}
    for suite in SUITES:
        if (suite == "e2e" and args.skip_e2e) or (suite == "integration" and args.skip_integration):
            continue
        rows = parse_junit(run_suite(suite, coverage=suite == "unit"))
        counts[suite] = Counter(r["outcome"] for r in rows)
        docs: dict[str, tuple[str, dict[str, str]]] = {}
        for i, row in enumerate(rows, start=1):
            path = ROOT / row["file"]
            if row["file"] not in docs:
                docs[row["file"]] = describe(path) if path.exists() else (row["file"], {})
            area, tests = docs[row["file"]]
            func = row["name"].split("[")[0]
            cases.append(
                {
                    **row,
                    "suite": suite,
                    "id": f"{SUITES[suite]['prefix']}-{i:03d}",
                    "func": func,
                    "area": area,
                    "what": tests.get(func, row["message"] or func),
                }
            )

    cov = read_json(TMP / "coverage.json")
    coverage = cov["totals"]["percent_covered"] if cov else None
    print("\n=== alert-rule tests (promtool) ===")
    rules = promtool_rule_tests()
    print(rules.get("detail", ""))
    criteria = acceptance(counts, args.api.rstrip("/"), rules)
    write_reports(cases, counts, criteria, coverage)
    print("\n=== acceptance criteria ===")
    for c in criteria:
        print(f"{c['result']:8} {c['id']}  {c['what']}: {c['value']}")
    print("\nwrote TEST_REPORT.md and TEST_CASES.md")
    return 0 if all(c["result"] == "PASS" for c in criteria) else 1


if __name__ == "__main__":
    sys.exit(main())
