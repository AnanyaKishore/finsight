"""The reproducibility drill's comparison (scripts/repro_drill.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("repro_drill", ROOT / "scripts" / "repro_drill.py")
drill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drill)


def _reports(val_mape=2.91234, test_mape=3.21001, epoch=17):
    return {
        "reports/lstm_train.json": {
            "selected": "h64_l1",
            "best_epoch": epoch,
            "val_mape": val_mape,
        },
        "reports/lstm_metrics.json": {
            "model": {"mape": test_mape},
            "naive": {"mape": 3.339},
            "skill_vs_naive": 0.0386,
            "test_windows": 640,
        },
    }


def test_identical_metrics_are_identical():
    rows, verdict = drill.compare(_reports(), _reports())
    assert verdict == "identical" and all(r["status"] == "identical" for r in rows)


def test_float_noise_is_within_tolerance_but_a_real_change_differs():
    assert drill.compare(_reports(), _reports(test_mape=3.2150))[1] == "within tolerance"
    assert drill.compare(_reports(), _reports(test_mape=3.40))[1] == "differs"
    assert drill.compare(_reports(), _reports(epoch=18))[1] == "differs"


def test_a_missing_report_is_reported_not_passed():
    after = _reports()
    del after["reports/lstm_metrics.json"]
    assert drill.compare(_reports(), after)[1] == "missing"
