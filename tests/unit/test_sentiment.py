"""Tests for news sentiment: scoring rules, data fetch/parse, evaluation, incremental scoring."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from finsight.common.exceptions import FinSightError
from finsight.sentiment import evaluate as ev
from finsight.sentiment import fetch
from finsight.sentiment import score as sc
from finsight.sentiment.scoring import (
    clean_texts,
    labels_for,
    model_version,
    scores_from_probabilities,
)
from tests.factories import make_news, make_phrasebank, make_tiny_sentiment_model


# ------------------------------------------------------------------ scoring
def test_score_is_p_pos_minus_p_neg_and_bounded():
    s = scores_from_probabilities(np.array([0.9, 0.5, 0.0]), np.array([0.1, 0.5, 1.0000001]))
    assert np.allclose(s, [-0.8, 0.0, 1.0]) and s.max() <= 1.0


def test_labels_with_and_without_neutral_band():
    s = np.array([-0.9, -0.3, 0.0, 0.3, 0.9])
    assert list(labels_for(s, 0.0)) == ["negative", "negative", "neutral", "positive", "positive"]
    assert list(labels_for(s, 0.5)) == ["negative", "neutral", "neutral", "neutral", "positive"]
    assert list(labels_for(np.array([0.5, -0.5]), 0.5)) == ["positive", "negative"]  # inclusive


def test_model_version_changes_with_band_and_length():
    a = model_version("714eb0fa89d2f80546fda750413ed43d93601a13", 0.9, 256)
    assert a == "714eb0fa89d2-band0.9000-len256"
    assert a != model_version("714eb0fa89d2f80546fda750413ed43d93601a13", 0.95, 256)
    assert a != model_version("714eb0fa89d2f80546fda750413ed43d93601a13", 0.9, 128)


def test_clean_texts_handles_missing_and_whitespace():
    assert clean_texts([None, float("nan"), "  a \n b  "]) == ["", "", "a b"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    pytest.importorskip("transformers")
    return make_tiny_sentiment_model(tmp_path_factory.mktemp("tiny") / "base")


def test_scorer_is_order_preserving_and_batch_invariant(tiny_model):
    from finsight.sentiment.scoring import SentimentScorer

    texts = ["the company shares rose", None, "loss fell", "strong growth " * 30, "profit"]
    scorer = SentimentScorer(tiny_model, max_length=32, batch_size=2, threads=1)
    a = scorer.scores(texts)
    scorer.batch_size = 1
    b = scorer.scores(texts)
    assert a.shape == (5,) and np.all(np.abs(a) <= 1)
    assert np.allclose(a, b, atol=1e-6)  # padding/batching must not change a score
    reordered = scorer.scores(texts[::-1])
    assert np.allclose(reordered[::-1], a, atol=1e-6)


def test_scorer_rejects_models_without_pos_neg_labels(tmp_path):
    pytest.importorskip("transformers")
    from finsight.sentiment.scoring import SentimentScorer

    path = make_tiny_sentiment_model(tmp_path / "m", labels=("A", "B", "C"))
    with pytest.raises(ValueError, match="POSITIVE/NEGATIVE"):
        SentimentScorer(path)


# -------------------------------------------------------------------- fetch
def test_parse_phrasebank_keeps_at_signs_and_latin1():
    raw = "Revenue rose 5 % @ Q3 .@positive\r\nNokia's caf\xe9 sales fell .@negative\n\n".encode(
        "iso-8859-1"
    )
    frame = fetch.parse_phrasebank(raw, "AllAgree")
    assert frame["label"].tolist() == ["positive", "negative"]
    assert frame["sentence"].iloc[0] == "Revenue rose 5 % @ Q3 ."
    assert "café" in frame["sentence"].iloc[1]
    with pytest.raises(FinSightError):
        fetch.parse_phrasebank(b"no label here", "AllAgree")
    with pytest.raises(FinSightError):
        fetch.parse_phrasebank(b"x@bullish", "AllAgree")


def test_phrasebank_from_zip_selects_agreement_level():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", "Up .@positive\n")
        zf.writestr("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "a@neutral\nb@neutral\n")
    assert len(fetch.phrasebank_from_zip(buf.getvalue(), "AllAgree")) == 1
    assert len(fetch.phrasebank_from_zip(buf.getvalue(), "50Agree")) == 2
    with pytest.raises(FinSightError):
        fetch.phrasebank_from_zip(buf.getvalue(), "66Agree")


def test_fetch_model_cleans_hub_metadata_and_checks_files(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    def fake_snapshot(repo_id, revision, local_dir, allow_patterns):
        for name in allow_patterns:
            (tmp_path / "dest" / name).write_text("x")
        (tmp_path / "dest" / ".cache").mkdir()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    info = fetch.fetch_model("org/model", "abc123", tmp_path / "dest")
    assert not (tmp_path / "dest" / ".cache").exists()
    assert json.loads((tmp_path / "dest" / "SOURCE.json").read_text())["revision"] == "abc123"
    assert info["files"] == fetch.MODEL_FILES

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: None)
    with pytest.raises(FinSightError, match="incomplete"):
        fetch.fetch_model("org/model", "abc123", tmp_path / "dest")


def test_download_retry_policy():
    class Resp:
        def __init__(self, status):
            self.status_code = status

    class FakeHTTPError(Exception):
        def __init__(self, status):
            self.response = Resp(status)

    assert fetch._network_error(FakeHTTPError(503)) and fetch._network_error(FakeHTTPError(429))
    assert not fetch._network_error(FakeHTTPError(404))
    assert fetch._network_error(ConnectionError()) and not fetch._network_error(ValueError())


# --------------------------------------------------------------- evaluation
def test_classification_metrics_known_values():
    m = ev.classification_metrics(
        ["positive", "positive", "negative", "negative"],
        ["positive", "negative", "negative", "neutral"],  # neutral = wrong in binary
        ("negative", "positive"),
    )
    assert m["accuracy"] == 0.5
    assert m["confusion"] == [[1, 0], [1, 1]]
    assert m["per_class"]["positive"]["precision"] == 1.0
    assert m["per_class"]["positive"]["recall"] == 0.5
    assert m["per_class"]["negative"]["recall"] == 0.5


def test_stratified_split_is_seeded_and_proportional():
    labels = make_phrasebank()["label"]
    a = ev.stratified_split(labels, 0.5, seed=1)
    b = ev.stratified_split(labels, 0.5, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, ev.stratified_split(labels, 0.5, seed=2))
    for cls, n in labels.value_counts().items():
        assert a[labels.to_numpy() == cls].sum() == round(n * 0.5)


def _scored_phrasebank():
    frame = make_phrasebank()
    rng = np.random.default_rng(0)
    centre = frame["label"].map({"neutral": 0.0, "positive": 0.95, "negative": -0.95})
    frame["score"] = np.clip(centre + rng.normal(0, 0.05, len(frame)), -1, 1)
    return frame


def test_band_calibration_separates_neutral():
    frame = _scored_phrasebank()
    band, curve = ev.calibrate_band(frame["score"].to_numpy(), frame["label"].tolist())
    assert 0.1 < band < 0.9
    assert curve["macro_f1"].max() == pytest.approx(1.0)
    assert curve.loc[curve["band"] == 0.0, "macro_f1"].iloc[0] < 1.0  # binary misses neutral


def test_evaluate_scores_against_majority_baselines():
    result, _ = ev.evaluate_scores(_scored_phrasebank(), 0.5, seed=42)
    assert result["n_calibration"] + result["n_test"] == 100
    assert result["binary"]["accuracy"] == 1.0
    assert result["binary_majority_baseline"]["predicts"] == "positive"
    assert result["three_class"]["macro_f1"] > result["three_class_majority_baseline"]["macro_f1"]
    assert result["three_class_majority_baseline"]["predicts"] == "neutral"
    perf = {"latency_p50_ms": 1.0, "latency_p95_ms": 2.0, "throughput_texts_per_s": 9.0}
    other, _ = ev.evaluate_scores(_scored_phrasebank(), 0.5, seed=7)
    compact = ev.compact({"gate": result, "pb": other}, perf, "rid", "cand", "gate", ["pb"])
    assert compact["beats_baseline"] is True and compact["run_id"] == "rid"
    assert compact["gate_macro_f1"] == round(result["three_class"]["macro_f1"], 4)
    assert compact["pb_contaminated"] is True and "pb_macro_f1" in compact


def test_evaluate_scores_needs_both_polarities():
    frame = _scored_phrasebank()
    with pytest.raises(ValueError):
        ev.evaluate_scores(frame[frame["label"] != "negative"], 0.5, seed=42)


def test_most_confident_errors_are_ranked():
    frame = pd.DataFrame(
        {"sentence": ["a", "b", "c"], "label": ["positive"] * 3, "score": [-0.2, -0.9, 0.8]}
    )
    errors = ev.most_confident_errors(frame, np.array(["negative", "negative", "positive"]))
    assert [e["sentence"] for e in errors] == ["b", "a"]


# --------------------------------------------------------- incremental score
def _news_frame():
    news = make_news(weeks=2, per_week=4)
    news["week_start"] = (
        pd.to_datetime(news["published_at"])
        .dt.tz_localize(None)
        .dt.to_period("W-SUN")
        .dt.start_time
    ).dt.date
    return news


def _write_news(tmp_path):
    path = tmp_path / "news"
    path.mkdir()
    _news_frame().to_parquet(path / "part-0.parquet", index=False)
    return path


def test_incremental_scoring_reuses_and_rescores(tmp_path):
    news = sc.load_news(_write_news(tmp_path))
    calls = []

    def fake(texts):
        calls.append(len(texts))
        return np.full(len(texts), 0.5)

    first, counts = sc.score_incremental(news, sc.load_cache(tmp_path / "none.parquet"), "v1", fake)
    assert counts == {"reused": 0, "scored": len(news)}
    cache = first.copy()
    _, counts = sc.score_incremental(news, cache, "v1", fake)
    assert counts == {"reused": len(news), "scored": 0} and calls == [len(news)]
    _, counts = sc.score_incremental(news, cache, "v2", fake)  # new model version
    assert counts["scored"] == len(news)
    edited = news.copy()
    edited.loc[0, "text"] = "edited"
    edited["text_sha1"] = sc.text_hash(edited["text"].tolist())
    _, counts = sc.score_incremental(edited, cache, "v1", fake)
    assert counts == {"reused": len(news) - 1, "scored": 1}


def test_weekly_aggregate_counts_and_explanations(tmp_path):
    news = sc.load_news(_write_news(tmp_path))
    scores = np.tile([0.9, -0.8, 0.1, 0.95], len(news) // 4)
    merged, _ = sc.score_incremental(news, sc.load_cache(tmp_path / "x"), "v", lambda t: scores)
    scored = sc.finalise(merged, neutral_band=0.5)
    weekly = sc.weekly_aggregate(scored, top_n=1)
    assert len(weekly) == 2 * 2  # 2 tickers x 2 weeks
    row = weekly.iloc[0]
    assert row["article_count"] == 4
    assert (row["positive"], row["negative"], row["neutral"]) == (2, 1, 1)
    assert row["mean_score"] == pytest.approx(np.mean([0.9, -0.8, 0.1, 0.95]))
    assert row["top_positive"][0]["score"] == 0.95 and len(row["top_positive"]) == 1
    assert row["top_negative"][0]["score"] == -0.8
    assert scored["confidence"].between(0.5, 1.0).all()


def test_publish_is_idempotent_and_removes_stale(db, tmp_path):
    from finsight.common.mongo import NEWS_SENTIMENT, SENTIMENT_WEEKLY

    news = sc.load_news(_write_news(tmp_path))
    merged, _ = sc.score_incremental(
        news, sc.load_cache(tmp_path / "x"), "v", lambda t: np.zeros(len(t))
    )
    scored = sc.finalise(merged, 0.5)
    weekly = sc.weekly_aggregate(scored, 3)
    sc.publish(db, scored, weekly, "run1")
    again = sc.publish(db, scored, weekly, "run1")
    assert again["articles_upserted"] == 0 and again["weeks_upserted"] == 0
    assert db[NEWS_SENTIMENT].count_documents({}) == len(scored)
    assert db[SENTIMENT_WEEKLY].count_documents({}) == len(weekly)
    doc = db[SENTIMENT_WEEKLY].find_one({}, {"_id": 0})
    assert doc["week_start"].tzinfo is not None and "top_positive" in doc

    fewer = scored[scored["ticker"] == "AAPL"]
    res = sc.publish(db, fewer, sc.weekly_aggregate(fewer, 3), "run2")
    assert res["stale_removed"] == (len(scored) - len(fewer)) + 2
    assert db[NEWS_SENTIMENT].count_documents({"ticker": "MSFT"}) == 0


# -------------------------------------------------- packaging + stage chain
def test_three_label_model_scores_and_confidence(tmp_path):
    pytest.importorskip("transformers")
    from finsight.sentiment.scoring import SentimentScorer

    path = make_tiny_sentiment_model(tmp_path / "fin", labels=("positive", "negative", "neutral"))
    scorer = SentimentScorer(path, max_length=32, batch_size=4, threads=1)
    texts = ["profit rose", "loss fell", "the company reports results"]
    probs = scorer.probability_matrix(texts)
    scores, confidence = scorer.score_and_confidence(texts)
    assert scorer.n_labels == 3 and np.allclose(probs.sum(axis=1), 1)
    assert np.allclose(scores, probs[:, 0] - probs[:, 1], atol=1e-6)  # pos - neg
    assert np.allclose(confidence, probs.max(axis=1), atol=1e-6)
    assert (confidence >= 1 / 3 - 1e-9).all()


def test_parse_twitter_maps_labels_and_strips_urls():
    raw = pd.DataFrame(
        {
            "text": ["$AAPL beats https://t.co/abc", "Stocks fall", "Fed meets", "Fed meets"],
            "label": [1, 0, 2, 2],
        }
    )
    out = fetch.parse_twitter(raw)
    assert out["label"].tolist() == ["positive", "negative", "neutral"]  # duplicate dropped
    assert out["sentence"].iloc[0] == "$AAPL beats" and list(out.columns) == [
        "id",
        "sentence",
        "label",
    ]
    with pytest.raises(FinSightError):
        fetch.parse_twitter(raw.assign(label=[1, 0, 2, 5]))


def test_load_cache_backfills_confidence_for_day4_caches(tmp_path):
    old = pd.DataFrame(
        {
            "ticker": ["A"],
            "news_id": ["1"],
            "text_sha1": ["x"],
            "model_version": ["v"],
            "score": [-0.6],
        }
    )
    old.to_parquet(tmp_path / "c.parquet")
    cache = sc.load_cache(tmp_path / "c.parquet")
    assert cache["confidence"].iloc[0] == pytest.approx(0.8)


def test_scoring_uses_the_production_candidate(sentiment_project):
    root, mlflow, params = sentiment_project
    outcome = ev.run_evaluation(params, "tiny_a", root)
    info = json.loads((root / "models/sentiment/candidates/tiny_a/run.json").read_text())
    assert info["model_version"].startswith("0123456789ab-band")
    assert info["eval_fingerprint"].startswith("gate@aaaa")
    run = mlflow.get_run(info["run_id"])
    assert run.data.params["n_gate"] == "100" and "git_commit" in run.data.tags
    assert run.data.tags["candidate"] == "tiny_a"
    assert "gate_macro_f1" in run.data.metrics and "latency_p95_ms" in run.data.metrics
    assert outcome["metrics"]["pb_contaminated"] is False
    assert set(outcome["curves"]) == {"gate", "pb"}

    model_dir = root / "models/sentiment/candidates/tiny_a/model"
    assert not list(model_dir.rglob("__pycache__"))  # clean before anything loads it
    model = mlflow.pyfunc.load_model(str(model_dir))
    out = model.predict(pd.DataFrame({"text": ["profit rose", "loss fell", ""]}))
    assert list(out.columns) == ["label", "confidence", "score"]
    assert set(out["label"]) <= {"positive", "negative", "neutral"}
    assert out["confidence"].between(0.5, 1.0).all() and out["score"].abs().le(1).all()

    production = {
        "candidate": "tiny_a",
        "run_id": info["run_id"],
        "version": "1",
        "model_version": info["model_version"],
        "neutral_band": info["neutral_band"],
        "path": model_dir,
    }
    news_dir = _write_news(root)
    first = sc.run_scoring(params, news_dir, root, production=production)
    assert first["report"]["scored_this_run"] == first["report"]["articles"]
    assert first["report"]["candidate"] == "tiny_a"
    second = sc.run_scoring(params, news_dir, root, production=production)
    assert second["report"]["scored_this_run"] == 0  # everything reused from the cache
    pd.testing.assert_frame_equal(first["scored"], second["scored"])
    assert first["scored"]["week_start"].iloc[0] <= date(2026, 8, 31)


# ------------------------------------------------- progress + checkpoints (Day 6)
def test_chunked_scoring_matches_one_shot_and_goes_shortest_first(tmp_path):
    news = sc.load_news(_write_news(tmp_path))
    empty = sc.load_cache(tmp_path / "none.parquet")

    def fake(texts):
        return np.array([len(t) / 1000 for t in texts])

    one_shot, _ = sc.score_incremental(news, empty, "v1", fake)
    seen, calls = [], []

    def recording(texts):
        calls.append([len(t) for t in texts])
        return fake(texts)

    chunked, counts = sc.score_incremental(
        news, empty, "v1", recording, chunk_size=3, on_chunk=lambda rows, p: seen.append((rows, p))
    )
    pd.testing.assert_frame_equal(one_shot, chunked)
    assert counts == {"reused": 0, "scored": len(news)}
    assert all(len(c) <= 3 for c in calls)
    lengths = [n for c in calls for n in c]
    assert lengths == sorted(lengths)  # shortest first, across chunks
    rows, last = seen[-1]
    assert last.done == last.total == len(news) and last.done_chars == last.total_chars
    assert len(rows) == len(news) and list(rows.columns) == sc.CACHE_COLUMNS
    assert [p.done for _, p in seen] == sorted(p.done for _, p in seen)


def _scoring_params(tmp_path):
    return {
        "scores_path": "data/sentiment/article_scores.parquet",
        "weekly_path": "data/sentiment/weekly.parquet",
        "top_headlines": 2,
        "checkpoint_every": 4,
    }


def test_interrupted_scoring_resumes_from_checkpoint(tmp_path):
    news_dir = _write_news(tmp_path)
    params = _scoring_params(tmp_path)
    (tmp_path / "data/sentiment").mkdir(parents=True)
    production = {"model_version": "v1", "neutral_band": 0.3, "run_id": "r1", "version": "1"}
    cache_path = tmp_path / params["scores_path"]
    partial = sc.checkpoint_path(cache_path)
    calls = []

    def fake(texts):
        calls.append(len(texts))
        return np.array([((len(t) % 7) - 3) / 4 for t in texts])

    def crashes_on_second_chunk(texts):
        if calls:
            raise RuntimeError("laptop went to sleep")
        return fake(texts)

    with pytest.raises(RuntimeError):
        sc.run_scoring(params, news_dir, tmp_path, crashes_on_second_chunk, production)
    assert partial.exists() and not cache_path.exists()
    assert len(pd.read_parquet(partial)) == 4  # the first chunk survived

    calls.clear()
    resumed = sc.run_scoring(params, news_dir, tmp_path, fake, production)
    total = resumed["report"]["articles"]
    assert resumed["report"]["scored_this_run"] == total - 4 == sum(calls)
    assert resumed["report"]["reused_from_cache"] == 4
    assert cache_path.exists() and not partial.exists()

    fresh_root = tmp_path / "fresh"
    (fresh_root / "data/sentiment").mkdir(parents=True)
    fresh = sc.run_scoring(params, news_dir, fresh_root, fake, production)
    pd.testing.assert_frame_equal(resumed["scored"], fresh["scored"])


def test_progress_estimates_time_left_from_characters():
    now = [100.0]
    progress = sc.ProgressLogger(total=10, total_chars=1000, clock=lambda: now[0], started=100.0)
    now[0] = 110.0
    fields = progress(done=5, done_chars=250)  # half the articles, a quarter of the text
    assert fields["percent"] == 50.0 and fields["articles_per_s"] == 0.5
    assert fields["eta_s"] == 30  # 750 chars left at 25 chars/s, not 10 s
