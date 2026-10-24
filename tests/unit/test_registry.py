"""Tests for the model registry: gate rules, registration, promotion, rollback, pointer."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from finsight.common.exceptions import FinSightError
from finsight.registry import gate
from finsight.registry.pointer import local_model_dir, production_entry

REGISTRY = {
    "lstm_name": "t-lstm",
    "sentiment_name": "t-sentiment",
    "lstm_acceptance_max_ratio_to_naive": 1.02,
    "lstm_min_improvement": 0.0,
    "sentiment_min_improvement": 0.0,
    "report": "reports/promotion.json",
}


# --------------------------------------------------------------- pure rules
def _decide(**kw):
    base = {
        "metric": "m",
        "challenger_version": "2",
        "challenger_metric": 1.0,
        "challenger_accepted": True,
        "champion_version": "1",
        "champion_metric": 2.0,
        "higher_is_better": False,
        "min_improvement": 0.0,
    }
    return gate.decide(**(base | kw))


def test_gate_promotes_only_better_accepted_challengers():
    assert _decide().promote  # lower MAPE wins
    assert not _decide(challenger_metric=2.0).promote  # a tie keeps the champion
    assert not _decide(challenger_accepted=False).promote
    assert _decide(higher_is_better=True, challenger_metric=3.0).promote
    assert not _decide(min_improvement=1.5).promote  # gain 1.0 <= 1.5
    assert _decide(champion_metric=None).promote  # champion not comparable
    assert not _decide(challenger_version="1").promote  # already production
    assert not _decide(challenger_version=None).promote


def test_gate_bootstraps_first_model_and_records_acceptance():
    first = _decide(champion_version=None, champion_metric=None, challenger_accepted=False)
    assert first.promote and "FAILS acceptance" in first.reason
    assert json.dumps(first.as_dict())


def test_acceptance_and_best_candidate():
    assert gate.lstm_accepted(2.9, 2.9, 1.02) and not gate.lstm_accepted(3.0, 2.9, 1.02)
    cands = {
        "a": {"metric": 0.5, "accepted": True},
        "b": {"metric": 0.6, "accepted": True},
        "c": {"metric": 0.9, "accepted": False},
    }
    assert gate.pick_best(cands, ["a", "b", "c"]) == "b"
    assert (
        gate.pick_best(cands | {"b": {**cands["b"], "rolled_back": True}}, ["a", "b", "c"]) == "a"
    )
    tie = {"a": {"metric": 0.5, "accepted": True}, "b": {"metric": 0.5, "accepted": True}}
    assert gate.pick_best(tie, ["a", "b"]) == "a"  # ties keep params order
    assert gate.pick_best({}, ["a"]) is None


# ------------------------------------------------------------------ pointer
def test_pointer_prefers_local_copy_and_downloads_otherwise(tmp_path):
    local = tmp_path / "cand"
    (local / "model").mkdir(parents=True)
    (local / "run.json").write_text(json.dumps({"run_id": "r1"}))
    entry = {"run_id": "r1", "registered_name": "n", "version": "3", "model_uri": "models:/n/3"}
    assert local_model_dir(entry, local, tmp_path / "cache") == local / "model"

    calls = []

    def fake_download(artifact_uri, dst_path):
        calls.append(artifact_uri)
        (tmp_path / "cache/n/v3/MLmodel").write_text("x")

    other = entry | {"run_id": "r0"}
    path = local_model_dir(other, local, tmp_path / "cache", downloader=fake_download)
    assert path == tmp_path / "cache/n/v3" and calls == ["models:/n/3"]
    local_model_dir(other, local, tmp_path / "cache", downloader=fake_download)
    assert len(calls) == 1  # cached after the first download
    with pytest.raises(FinSightError):
        production_entry({"lstm": {"production": None}}, "lstm")


# ------------------------------------------------------- registry integration
class _Constant:
    """A trivially loggable pyfunc used to create registrable LSTM versions."""


def _log_lstm(mlflow, root, mape, naive=3.0):
    import mlflow.pyfunc

    class Naive(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    with mlflow.start_run() as run:
        info = mlflow.pyfunc.log_model(name="model", python_model=Naive())
    model_dir = root / "models/lstm"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "run.json").write_text(
        json.dumps({"run_id": run.info.run_id, "model_uri": info.model_uri, "selected_config": {}})
    )
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/lstm_metrics.json").write_text(
        json.dumps({"model": {"mape": mape}, "naive": {"mape": naive}, "skill_vs_naive": 0.0})
    )
    return run.info.run_id


def _settings(params):
    return {
        "registry": REGISTRY,
        "lstm": {"model_dir": "models/lstm", "metrics_report": "reports/lstm_metrics.json"},
        "etl": {"processed_prices_path": "data/processed/prices"},
        "sentiment": params,
    }


def test_lstm_gate_registers_idempotently_and_compares_on_same_data(sentiment_project, monkeypatch):
    from finsight.registry import register
    from finsight.registry.client import get_client

    root, mlflow, params = sentiment_project
    client = get_client()
    _log_lstm(mlflow, root, mape=2.95)
    first = register.lstm_gate(client, _settings(params), root)
    assert first["decision"]["promote"] and first["production"]["version"] == "1"
    again = register.lstm_gate(client, _settings(params), root)
    assert not again["decision"]["promote"]
    assert len(client.search_model_versions("name='t-lstm'")) == 1  # nothing re-registered

    # A new model: the champion is re-scored on today's data (stubbed here).
    _log_lstm(mlflow, root, mape=2.90)
    monkeypatch.setattr(register, "lstm_test_mape", lambda *a, **k: 2.80)
    kept = register.lstm_gate(client, _settings(params), root, loader=lambda uri: None)
    assert not kept["decision"]["promote"] and kept["production"]["version"] == "1"
    assert str(client.get_model_version_by_alias("t-lstm", "staging").version) == "2"

    _log_lstm(mlflow, root, mape=2.70)
    won = register.lstm_gate(client, _settings(params), root, loader=lambda uri: None)
    assert won["decision"]["promote"] and won["production"]["version"] == "3"
    assert won["decision"]["champion_metric"] == 2.8


def test_sentiment_gate_rollback_and_no_auto_repromotion(sentiment_project):
    from finsight.registry import register, rollback
    from finsight.registry.client import ROLLED_BACK, get_client, history
    from finsight.sentiment.evaluate import run_evaluation

    root, mlflow, params = sentiment_project
    for key in params["candidates"]:
        outcome = run_evaluation(params, key, root)
        (root / "reports").mkdir(exist_ok=True)
        (root / f"reports/sentiment_eval_{key}.json").write_text(json.dumps(outcome["metrics"]))
    client = get_client()
    result = register.sentiment_gate(client, _settings(params), root)
    best = result["decision"]["best_candidate"]
    assert result["decision"]["promote"] and result["production"]["candidate"] == best
    assert set(result["candidates"]) == {"tiny_a", "tiny_b"}
    assert result["production"]["model_version"] and "neutral_band" in result["production"]
    assert len(client.search_model_versions("name='t-sentiment'")) == 2
    again = register.sentiment_gate(client, _settings(params), root)
    assert not again["decision"]["promote"]
    assert len(client.search_model_versions("name='t-sentiment'")) == 2

    # Roll back: first force the other candidate into production so history has 2 entries.
    other = next(k for k in result["candidates"] if k != best)
    from finsight.registry.client import set_production

    set_production(client, "t-sentiment", result["candidates"][other]["version"])
    set_production(client, "t-sentiment", result["candidates"][best]["version"])
    change = rollback.rollback(client, "t-sentiment", "test rollback")
    assert change["to_version"] == result["candidates"][other]["version"]
    bad = client.get_model_version("t-sentiment", change["from_version"])
    assert bad.tags[ROLLED_BACK] == "true" and bad.tags["rollback_reason"] == "test rollback"
    assert history(client, "t-sentiment")[-1] == change["to_version"]

    rollback.update_report(
        root / "reports/promotion.json", "sentiment", client, "t-sentiment", change
    )
    report = json.loads((root / "reports/promotion.json").read_text())
    assert report["sentiment"]["production"]["candidate"] == other

    # The gate must not promote the rolled-back version again on the next run.
    rerun = register.sentiment_gate(client, _settings(params), root)
    assert rerun["production"]["version"] == change["to_version"]
    assert rerun["decision"]["best_candidate"] == other

    with pytest.raises(FinSightError):
        rollback.choose_target(client, "t-sentiment", change["to_version"], change["to_version"])

    # Explicitly restoring the rolled-back version clears its mark (operator decision).
    undo = rollback.rollback(client, "t-sentiment", "undo", to_version=change["from_version"])
    assert undo["to_version"] == change["from_version"]
    restored = client.get_model_version("t-sentiment", change["from_version"])
    assert ROLLED_BACK not in (restored.tags or {})
    demo = rollback.rollback(
        client, "t-sentiment", "demo", to_version=change["to_version"], mark=False
    )
    demo_from = client.get_model_version("t-sentiment", demo["from_version"])
    assert ROLLED_BACK not in (demo_from.tags or {})


def test_rollback_without_history_fails(sentiment_project):
    from finsight.registry import rollback
    from finsight.registry.client import ensure_registered_model, get_client

    client = get_client()
    ensure_registered_model(client, "empty", "x")
    with pytest.raises(FinSightError, match="no production"):
        rollback.rollback(client, "empty", "why")


def test_champion_reevaluation_for_sentiment_uses_test_half():
    from finsight.registry.register import sentiment_test_macro_f1
    from tests.factories import make_phrasebank

    frame = make_phrasebank()

    class Oracle:
        def predict(self, df):
            lookup = dict(zip(frame["sentence"], frame["label"], strict=True))
            return pd.DataFrame({"label": [lookup[t] for t in df["text"]]})

    assert sentiment_test_macro_f1(Oracle(), frame, 0.5, 42) == pytest.approx(1.0)
