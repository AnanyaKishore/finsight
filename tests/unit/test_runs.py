"""Tests for stage run recording (best effort, never breaks a stage)."""

from __future__ import annotations

import mongomock
import pytest

from finsight.common import runs


def test_stage_run_success_and_failure_are_recorded(monkeypatch):
    client = mongomock.MongoClient(tz_aware=True)
    monkeypatch.setenv("FINSIGHT_RUN_RECORDING", "1")
    import finsight.common.mongo as mongo_module

    monkeypatch.setattr(mongo_module, "get_client", lambda settings, timeout_ms=0: client)

    class _S:  # minimal settings stand-in
        mongo_db = "finsight"

    with runs.stage_run("validate", _S()) as run:
        run.metrics["checks"] = 3
    with pytest.raises(ValueError), runs.stage_run("process", _S()):
        raise ValueError("boom")

    docs = {d["stage"]: d for d in client.finsight.pipeline_runs.find()}
    assert docs["validate"]["status"] == "success" and docs["validate"]["metrics"] == {"checks": 3}
    assert docs["process"]["status"] == "failed" and "boom" in docs["process"]["error"]
    assert docs["process"]["pipeline"] == "dvc"


def test_stage_run_survives_unreachable_mongo(monkeypatch):
    monkeypatch.setenv("FINSIGHT_RUN_RECORDING", "1")
    import finsight.common.mongo as mongo_module
    from finsight.common.exceptions import FinSightError

    def _down(*_, **__):
        raise FinSightError("mongo down")

    monkeypatch.setattr(mongo_module, "get_client", _down)

    class _S:
        mongo_db = "finsight"

    with runs.stage_run("export", _S()) as run:  # must not raise
        run.metrics["ok"] = True
    assert run.status == "success"


def test_recording_disabled_flag(monkeypatch):
    monkeypatch.setenv("FINSIGHT_RUN_RECORDING", "0")
    assert runs.recording_enabled() is False
