"""Reproducibility drill (Day 10): rebuild the LSTM from a Git commit and compare metrics.

    python scripts/repro_drill.py                 # the commit you are on (HEAD)
    python scripts/repro_drill.py --commit day9   # any tag or commit hash

What it does, in a fresh clone so your working copy is never touched:

1. ``git clone`` this repository into ``--workdir`` and check out the commit where
   the model of ``--commit`` was trained (the last commit that changed
   ``reports/lstm_train.json``; later daily runs refresh the data but not the model).
2. Restore the secrets: ``secrets/.env.enc`` is decrypted with your key (Day 9);
   without a key, ``.env`` is copied instead.
3. ``dvc pull --with-deps evaluate_lstm``: exactly the data and model of that
   commit, from the local DVC remote.
4. ``mlflow run . -e train_lstm`` (the MLproject entry point) retrains the
   model from scratch in a new MLflow run, experiment ``finsight-repro``.
5. ``python -m finsight.models.lstm.evaluate`` scores it on the same held-out
   test windows.
6. Compares the new validation and test metrics with the ones committed at that
   commit and writes ``docs/REPRODUCIBILITY.md`` (in your working copy) with
   the commit hash, both MLflow run IDs, the table and the verdict.

The clone runs its *own* code (``PYTHONPATH=<clone>/src``), not the copy
installed in your environment. Needs the MLflow server (``docker compose up -d
mlflow``); MongoDB is not needed. Takes about 5 minutes (the training grid).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# (file, key path, label, digits stored in the report)
METRICS = [
    ("reports/lstm_train.json", ("selected",), "selected configuration (validation)", None),
    ("reports/lstm_train.json", ("best_epoch",), "best epoch", None),
    ("reports/lstm_train.json", ("val_mape",), "validation MAPE (%)", 4),
    ("reports/lstm_metrics.json", ("model", "mape"), "test MAPE, LSTM (%)", 4),
    ("reports/lstm_metrics.json", ("naive", "mape"), "test MAPE, naive (%)", 4),
    ("reports/lstm_metrics.json", ("skill_vs_naive",), "skill vs naive", 4),
    ("reports/lstm_metrics.json", ("test_windows",), "test windows", None),
]
TOLERANCE = 0.01  # absolute, in the metric's own unit (MAPE percentage points)


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    """Run one step, echoing it; stop the drill if it fails."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    done = subprocess.run(cmd, cwd=cwd, env=env, check=False)  # noqa: S603
    if done.returncode != 0:
        sys.exit(f"step failed (exit {done.returncode}): {' '.join(cmd)}")


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,  # noqa: S607
                          check=True).stdout.strip()  # fmt: skip


def pick(doc: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if not isinstance(doc, dict):
            return None
        doc = doc.get(key)
    return doc


def read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def check_lock(workdir: Path) -> None:
    """Warn if dvc.lock pairs the model with different data than it was trained on."""
    import yaml  # noqa: PLC0415

    lock = yaml.safe_load((workdir / "dvc.lock").read_text(encoding="utf-8"))["stages"]
    path = "data/processed/prices"
    produced = {o["path"]: o.get("md5") for o in lock["process"]["outs"]}.get(path)
    trained_on = {d["path"]: d.get("md5") for d in lock["train_lstm"]["deps"]}.get(path)
    if produced != trained_on:
        print(f"warning: dvc.lock: {path} is {produced} but train_lstm used {trained_on};"
              " metrics may differ because the data differs")  # fmt: skip


def compare(before: dict[str, dict], after: dict[str, dict]) -> tuple[list[dict], str]:
    """Row per metric and the overall verdict: identical / within tolerance / differs."""
    rows, worst = [], "identical"
    for file, keys, label, digits in METRICS:
        old, new = pick(before.get(file), keys), pick(after.get(file), keys)
        if old is None or new is None:
            status = "missing"
        elif digits is None or not isinstance(old, int | float):
            status = "identical" if old == new else "differs"
        elif round(old, digits) == round(new, digits):
            status = "identical"
        elif abs(old - new) <= TOLERANCE:
            status = "within tolerance"
        else:
            status = "differs"
        rows.append({"metric": label, "committed": old, "reproduced": new, "status": status})
        order = ["identical", "within tolerance", "differs", "missing"]
        worst = max(worst, status, key=order.index)
    return rows, worst


def write_markdown(info: dict[str, Any], rows: list[dict], verdict: str) -> Path:
    lines = [
        "# FinSight — Reproducibility drill",
        "",
        "Written by `python scripts/repro_drill.py`. The guidelines ask that every experiment be "
        "reproducible from a Git commit hash and an MLflow run ID; this drill proves it for the "
        "production forecasting model.",
        "",
        "| | |",
        "|---|---|",
        f"| Date | {info['date']} |",
        f"| Git commit (where the model was trained) | `{info['commit']}` (requested: {info['ref']}) |",
        f"| Original MLflow run (committed model) | `{info['old_run']}` |",
        f"| Reproduced MLflow run (experiment `finsight-repro`) | `{info['new_run']}` |",
        f"| Fresh clone | `{info['workdir']}` |",
        f"| Environment | Python {platform.python_version()}, {platform.system()} {platform.machine()}, "
        "`mlflow run --env-manager local` |",
        "",
        "## Steps",
        "",
        "```bash",
        f"git clone {ROOT} {info['workdir']} && cd {info['workdir']}",
        f"git checkout {info['commit']}",
        "python -m finsight.security.secrets decrypt          # .env from secrets/.env.enc",
        "dvc pull --with-deps evaluate_lstm                    # data + model of that commit",
        "mlflow run . -e train_lstm --env-manager local --experiment-name finsight-repro",
        "python -m finsight.models.lstm.evaluate               # same held-out test windows",
        "```",
        "",
        "## Result",
        "",
        f"**Verdict: {verdict}.**",
        "",
        "| Metric | Committed | Reproduced | Status |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r['metric']} | {r['committed']} | {r['reproduced']} | {r['status']} |" for r in rows
    ]
    lines += [
        "",
        "*identical* = equal to the precision stored in the report; *within tolerance* = "
        f"differs by at most {TOLERANCE} (floating-point differences between machines or thread "
        "counts, for example a model trained inside the scheduler container and reproduced on the "
        "host); *differs* = a real difference to investigate.",
        "",
        "Same seed (`lstm.seed` in `params.yaml`), same data (DVC hashes in `dvc.lock`), same code "
        "(the commit), same package versions (`environment.yml` → `requirements/*.txt`, also used by "
        "the Docker images).",
    ]
    out = ROOT / "docs" / "REPRODUCIBILITY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the drill."""
    parser = argparse.ArgumentParser(description="FinSight reproducibility drill")
    parser.add_argument("--commit", default="HEAD", help="tag, branch or hash (default HEAD)")
    parser.add_argument(
        "--workdir", type=Path, default=None, help="default /tmp/finsight-repro-<hash>"
    )
    parser.add_argument("--keep", action="store_true", help="keep the clone afterwards")
    args = parser.parse_args(argv)

    requested = git("rev-parse", "--short", f"{args.commit}^{{commit}}")
    # The model at a commit was trained in the run that last wrote lstm_train.json. Later
    # daily runs refresh data/processed without retraining, so their dvc.lock pairs new data
    # with the older model; reproducing must use the commit where the two belong together.
    commit = git("log", "-1", "--format=%h", requested, "--", "reports/lstm_train.json")
    if not commit:
        sys.exit(f"no commit at or before {args.commit} contains reports/lstm_train.json")
    if commit != requested:
        print(f"{args.commit} = {requested}; its model was trained at commit {commit}, using that")
    workdir = (args.workdir or Path(tempfile.gettempdir()) / f"finsight-repro-{commit}").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    if shutil.which("mlflow") is None or shutil.which("dvc") is None:
        sys.exit("mlflow and dvc must be on PATH (activate the finsight environment)")

    no_lfs = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}  # images are not needed here
    run(["git", "clone", "--quiet", str(ROOT), str(workdir)], cwd=ROOT, env=no_lfs)
    run(["git", "checkout", "--quiet", commit], cwd=workdir, env=no_lfs)
    env = {**os.environ, "PYTHONPATH": str(workdir / "src"), "FINSIGHT_ROOT": str(workdir),
           "FINSIGHT_RUN_RECORDING": "0"}  # fmt: skip
    env.pop("MLFLOW_RUN_ID", None)
    if (workdir / "secrets" / ".env.enc").exists():
        try:
            run(
                [sys.executable, "-m", "finsight.security.secrets", "decrypt"], cwd=workdir, env=env
            )
        except SystemExit:
            print("could not decrypt secrets/.env.enc; copying .env instead")
            shutil.copy2(ROOT / ".env", workdir / ".env")
    elif (ROOT / ".env").exists():
        shutil.copy2(ROOT / ".env", workdir / ".env")

    check_lock(workdir)
    before = {file: read(workdir / file) for file in {m[0] for m in METRICS}}
    old_run = pick(before.get("reports/lstm_train.json"), ("run_id",))
    run(["dvc", "pull", "--quiet", "--with-deps", "evaluate_lstm"], cwd=workdir, env=env)
    run(["mlflow", "run", ".", "-e", "train_lstm", "--env-manager", "local",
         "--experiment-name", "finsight-repro"], cwd=workdir, env=env)  # fmt: skip
    # evaluate logs into the training run; pointing MLflow at that run (as `mlflow run` does)
    # stops it from switching to the params.yaml experiment, which would not match.
    trained = read(workdir / "models" / "lstm" / "run.json") or {}
    eval_env = {**env, "MLFLOW_RUN_ID": trained.get("run_id", "")}
    run([sys.executable, "-m", "finsight.models.lstm.evaluate"], cwd=workdir, env=eval_env)
    after = {file: read(workdir / file) for file in {m[0] for m in METRICS}}
    new_run = pick(after.get("reports/lstm_train.json"), ("run_id",))

    rows, verdict = compare(before, after)
    info = {"date": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"), "commit": commit,
            "ref": args.commit, "old_run": old_run, "new_run": new_run, "workdir": workdir}  # fmt: skip
    out = write_markdown(info, rows, verdict)
    print(f"\n{'metric':40} {'committed':>14} {'reproduced':>14}  status")
    for r in rows:
        print(f"{r['metric']:40} {r['committed']!s:>14} {r['reproduced']!s:>14}  {r['status']}")
    print(f"\nverdict: {verdict}\nwrote {out.relative_to(ROOT)}")
    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if verdict in {"identical", "within tolerance"} else 1


if __name__ == "__main__":
    sys.exit(main())
