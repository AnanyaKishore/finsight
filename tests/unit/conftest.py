"""Unit-test isolation from the developer's machine.

Unit tests must behave identically on every machine, so they never read the
real ``.env`` (which holds your API keys). Without this, ``load_settings``
would re-load keys that a test deliberately removed with ``monkeypatch``.
Integration tests (tests/integration) still read ``.env`` because they need
the real MongoDB credentials.

Run recording (``finsight.common.runs``) is also switched off, so unit tests
never write pipeline-run documents into your real MongoDB.
"""

from __future__ import annotations

import pytest

from finsight.common import config as config_module


@pytest.fixture(autouse=True)
def _ignore_real_dotenv(monkeypatch):
    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("FINSIGHT_RUN_RECORDING", "0")
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Day 5: a throw-away sentiment project (two tiny candidates, two datasets)
# --------------------------------------------------------------------------- #
def sentiment_params() -> dict:
    """``sentiment`` params shaped like params.yaml, pointing at tiny local assets."""
    return {
        "candidates": {
            "tiny_a": {
                "repo": "tiny/a",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "base_dir": "models/sentiment/base/tiny_a",
                "model_dir": "models/sentiment/candidates/tiny_a",
                "trained_on": [],
            },
            "tiny_b": {
                "repo": "tiny/b",
                "revision": "fedcba9876543210fedcba9876543210fedcba98",
                "base_dir": "models/sentiment/base/tiny_b",
                "model_dir": "models/sentiment/candidates/tiny_b",
                "trained_on": ["pb"],
            },
        },
        "datasets": {
            "gate": {"revision": "aaaabbbbccccdddd", "path": "data/external/gate.csv"},
            "pb": {"revision": "1111222233334444", "path": "data/external/pb.csv"},
        },
        "gate_dataset": "gate",
        "calibration_share": 0.5,
        "seed": 42,
        "max_length": 32,
        "batch_size": 8,
        "torch_threads": 1,
        "top_headlines": 2,
        "experiment_name": "test-sentiment",
        "scores_path": "data/sentiment/article_scores.parquet",
        "weekly_path": "data/sentiment/weekly.parquet",
        "promotion_report": "reports/promotion.json",
    }


@pytest.fixture
def sentiment_project(tmp_path, monkeypatch):
    """Tiny candidates + datasets on disk and a local SQLite MLflow store."""
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("transformers")
    from tests.conftest import PROJECT_ROOT
    from tests.factories import make_phrasebank, make_tiny_sentiment_model

    params = sentiment_params()
    make_tiny_sentiment_model(tmp_path / params["candidates"]["tiny_a"]["base_dir"])
    make_tiny_sentiment_model(
        tmp_path / params["candidates"]["tiny_b"]["base_dir"],
        seed=1,
        labels=("positive", "negative", "neutral"),
    )
    for name, spec in params["datasets"].items():
        path = tmp_path / spec["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        make_phrasebank(n_neutral=60 if name == "gate" else 30).to_csv(path, index=False)
    (tmp_path / "src").symlink_to(PROJECT_ROOT / "src")
    monkeypatch.chdir(tmp_path)  # before MLflow resolves any relative artifact path
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test-sentiment", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment("test-sentiment")
    yield tmp_path, mlflow, params
    mlflow.set_tracking_uri("")
