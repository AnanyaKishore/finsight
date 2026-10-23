"""Tests for the LSTM forecaster: windows, split, network, metrics, training, packaging."""

from __future__ import annotations

import json
import subprocess
from datetime import date

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
mlflow = pytest.importorskip("mlflow")

from finsight.models.lstm import data, metrics  # noqa: E402
from finsight.models.lstm.network import LSTMForecaster  # noqa: E402
from finsight.models.lstm.pyfunc_model import input_columns  # noqa: E402
from tests.factories import make_prices  # noqa: E402

L, H = 20, 5


def _series(n=80, tickers=("AAPL", "MSFT")):
    idx = pd.bdate_range("2026-01-01", periods=n)
    return {
        t: pd.Series(100 + 10 * i + np.sin(np.arange(n) / 3) + np.arange(n) * 0.1, index=idx)
        for i, t in enumerate(tickers)
    }


# ---------------------------------------------------------------------- data
def test_scale_roundtrip_and_flat_window():
    raw = np.array([[10.0, 12.0, 11.0], [5.0, 5.0, 5.0]])
    scaled, lo, hi = data.scale_rows(raw)
    assert scaled[0].min() == 0 and scaled[0].max() == 1
    assert np.isfinite(scaled).all()  # flat window does not divide by zero
    assert np.allclose(data.unscale_rows(scaled, lo, hi), raw)


def test_windows_align_inputs_and_targets():
    ws = data.make_windows(_series(n=40), L, H)
    assert len(ws) == 2 * (40 - L - H + 1)
    s = _series(n=40)["AAPL"].to_numpy()
    first = np.where(ws.tickers == "AAPL")[0][0]
    assert np.allclose(ws.x_raw[first], s[:L]) and np.allclose(ws.y_raw[first], s[L : L + H])
    assert ws.origins[first] == _series(n=40)["AAPL"].index[L - 1]


def test_targets_are_scaled_with_input_window_only():
    ws = data.make_windows(_series(n=40), L, H)
    x, y, lo, hi = ws.scaled()
    assert x.shape == (len(ws), L, 1) and y.shape == (len(ws), H)
    assert np.allclose(lo, ws.x_raw.min(axis=1))  # never uses future prices


def test_split_is_purged_and_chronological():
    ws = data.make_windows(_series(n=120), L, H)
    m = data.split_masks(ws, val_origins=10, test_origins=10, purge=H)
    for t in ("AAPL", "MSFT"):
        pos = {k: ws.position[(ws.tickers == t) & m[k]] for k in m}
        # last train target day < first val target day, same for val -> test
        assert pos["train"].max() + H - 1 < pos["val"].min()
        assert pos["val"].max() + H - 1 < pos["test"].min()
        assert len(pos["test"]) == 10 and len(pos["val"]) == 10
        assert set(pos["train_val"]) == set(range(pos["val"].max() + 1))
    assert not (m["train_val"] & m["test"]).any()


def test_split_rejects_short_history():
    ws = data.make_windows(_series(n=40), L, H)
    with pytest.raises(ValueError):
        data.split_masks(ws, val_origins=10, test_origins=10, purge=H)


# ------------------------------------------------------------------- network
def test_untrained_network_is_the_naive_forecast():
    net = LSTMForecaster(horizon=H, hidden_size=8)
    x = torch.rand(4, L, 1)
    out = net(x)
    assert out.shape == (4, H)
    assert torch.allclose(out, x[:, -1, :].expand(-1, H))


# ------------------------------------------------------------------- metrics
def test_metrics_and_baselines():
    x = np.tile(np.arange(1.0, L + 1), (3, 1)) + 100  # linear uptrend
    y = x[:, -1:] + np.arange(1, H + 1)  # continues the trend
    tickers = np.array(["A", "A", "B"])
    res = metrics.compare_to_baselines(metrics.naive_forecast(x, H), x, y, tickers)
    assert res["pooled"]["drift"]["mape"] == pytest.approx(0.0, abs=1e-9)
    assert res["pooled"]["naive"]["directional_accuracy"] is None  # naive never moves
    assert res["skill_vs_naive"] == pytest.approx(0.0) and not res["beats_naive"]
    assert res["pooled"]["drift"]["directional_accuracy"] == 1.0
    assert len(res["per_horizon"]) == H and set(res["per_ticker"]) == {"A", "B"}


# -------------------------------------------------------------- train/package
PARAMS = {
    "lookback": L,
    "horizon": H,
    "val_origins": 6,
    "test_origins": 6,
    "grid": {"hidden_size": [8], "num_layers": [1], "learning_rate": [0.01]},
    "dropout": 0.0,
    "batch_size": 32,
    "max_epochs": 4,
    "patience": 2,
    "seed": 7,
    "torch_threads": 1,
    "selection_metric": "val_mape",
    "experiment_name": "test-lstm",
    "model_dir": "models/lstm",
    "train_report": "reports/lstm_train.json",
    "metrics_report": "reports/lstm_metrics.json",
}


def test_fit_is_deterministic_and_learns():
    from finsight.models.lstm.train import fit, grid_configs, set_seeds

    ws = data.make_windows(_series(n=90), L, H)
    m = data.split_masks(ws, 6, 6, H)
    cfg = grid_configs(PARAMS)[0]
    runs = []
    for _ in range(2):
        set_seeds(7, 1)
        runs.append(
            fit(
                cfg,
                ws.subset(m["train"]),
                ws.subset(m["val"]),
                epochs=4,
                patience=2,
                batch_size=32,
                seed=7,
            )
        )
    w0 = [p.detach().numpy() for p in runs[0].net.parameters()]
    w1 = [p.detach().numpy() for p in runs[1].net.parameters()]
    assert all(np.array_equal(a, b) for a, b in zip(w0, w1, strict=True))
    assert runs[0].history[-1]["train_loss"] <= runs[0].history[0]["train_loss"]
    assert 1 <= runs[0].best_epoch <= 4


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throw-away project: processed prices + local SQLite MLflow store."""
    from tests.conftest import PROJECT_ROOT

    prices = make_prices(start=date(2026, 3, 2), end=date(2026, 9, 30))
    prices["price"] = prices["adj_close"]
    path = tmp_path / "data/processed/prices"
    path.mkdir(parents=True)
    prices[["ticker", "date", "price"]].to_parquet(path / "part-00000.parquet", index=False)
    (tmp_path / "src").symlink_to(PROJECT_ROOT / "src")  # code_paths source
    monkeypatch.chdir(tmp_path)  # before MLflow resolves any relative artifact path
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test-lstm", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test-lstm")
    yield tmp_path, path
    mlflow.set_tracking_uri("")


def test_training_packages_a_working_pyfunc_and_evaluation(project):
    from finsight.models.lstm.evaluate import run_evaluation
    from finsight.models.lstm.train import run_training

    root, path = project
    outcome = run_training(PARAMS, path, root)
    info = outcome["run_info"]
    assert (root / "models/lstm/run.json").exists() and info["run_id"]

    run = mlflow.get_run(info["run_id"])
    assert run.data.tags["seed"] == "7" and "git_commit" in run.data.tags
    assert run.data.params["n_val"] == str(outcome["report"]["n_val"])
    children = mlflow.search_runs(filter_string=f"tags.parent_run = '{info['run_id']}'")
    assert len(children) == 1

    # Saved model is clean (loading it later writes bytecode; .dvcignore skips that).
    assert not list((root / "models/lstm/model").rglob("__pycache__"))
    model = mlflow.pyfunc.load_model(str(root / "models/lstm/model"))
    sample = pd.DataFrame(np.full((2, L), 100.0) + np.arange(L), columns=input_columns(L))
    out = model.predict(sample)
    assert list(out.columns) == [f"h_{i:02d}" for i in range(1, H + 1)]
    assert np.isfinite(out.to_numpy()).all()
    with pytest.raises(ValueError):
        model.predict(sample.assign(p_000=-1.0))
    # The MLflow signature rejects malformed input before our own checks run.
    with pytest.raises(mlflow.exceptions.MlflowException, match="missing inputs"):
        model.predict(sample.drop(columns=["p_005"]))

    report = run_evaluation(PARAMS, path, root)
    assert report["test_windows"] == 2 * PARAMS["test_origins"]
    assert {"model", "naive", "drift", "skill_vs_naive", "beats_naive"} <= set(report)
    assert mlflow.get_run(info["run_id"]).data.tags["evaluated"] == "true"
    assert (root / "reports/plots/lstm_horizon_mape.csv").exists()
    attribution = pd.read_csv(root / "reports/plots/lstm_input_attribution.csv")
    assert attribution["lag_from"].iloc[0] == 1 and len(attribution) == 2  # L=20, blocks of 10
    assert report["attribution_latest_block_pct"] >= 0


# ------------------------------------------------------------------- lineage
def test_git_state_ignores_dvc_outputs(tmp_path):
    from finsight.models.tracking import dvc_output_md5, git_state

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "code.py").write_text("a = 1\n")
    (tmp_path / "dvc.lock").write_text(
        "schema: '2.0'\nstages:\n  process:\n    outs:\n    - path: data/processed/prices\n"
        "      md5: abc.dir\n"
    )
    git("add", "-A")
    git("commit", "-qm", "init")
    (tmp_path / "dvc.lock").write_text((tmp_path / "dvc.lock").read_text() + "# changed\n")
    assert git_state(tmp_path)["git_dirty_code"] == "false"
    (tmp_path / "code.py").write_text("a = 2\n")
    state = git_state(tmp_path)
    assert state["git_dirty_code"] == "true" and "code.py" in state["git_dirty_files"]
    assert dvc_output_md5(tmp_path, "process", "data/processed/prices") == "abc.dir"
    assert dvc_output_md5(tmp_path, "process", "missing") is None
    assert json.dumps(state)


# ----------------------------------------------------- Day 4: extra channels
def test_extra_channel_windows_and_network():
    ws = data.make_windows(_series(n=40), L, H)
    vol = data.make_windows(
        {t: s.pct_change().abs().fillna(0) for t, s in _series(n=40).items()}, L, H
    )
    two = data.WindowSet(
        *[
            getattr(ws, f)
            for f in ("x_raw", "y_raw", "tickers", "origins", "position", "n_per_ticker")
        ],
        vol.x_raw[:, :, None],
    )
    x, y, lo, _ = two.scaled()
    assert x.shape == (len(ws), L, 2) and np.allclose(x[:, :, 0], ws.scaled()[0][:, :, 0])
    assert two.subset(two.tickers == "AAPL").x_extra.shape[0] == (two.tickers == "AAPL").sum()
    net = LSTMForecaster(horizon=H, hidden_size=8, input_size=2)
    xt = torch.from_numpy(x[:3])
    assert torch.allclose(net(xt), xt[:, -1, :1].expand(-1, H))  # residual anchor = price


def test_price_only_network_is_unchanged_by_input_size_default():
    torch.manual_seed(0)
    a = LSTMForecaster(horizon=H, hidden_size=8)
    torch.manual_seed(0)
    b = LSTMForecaster(horizon=H, hidden_size=8, input_size=1)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(p, q)  # Day 3 models load and train exactly as before


# ---------------------------------------------------- Day 4: explainability
def test_occlude_interpolates_and_flattens_edges():
    from finsight.models.lstm.explain import occlude

    x = np.array([[1.0, 5.0, 9.0, 3.0, 4.0]])
    assert np.allclose(occlude(x, 1, 3), [[1.0, 5 / 3, 7 / 3, 3.0, 4.0]])  # line 1 -> 3
    assert np.allclose(occlude(x, 0, 2), [[9.0, 9.0, 9.0, 3.0, 4.0]])
    assert np.allclose(occlude(x, 3, 5), [[1.0, 5.0, 9.0, 9.0, 9.0]])
    with pytest.raises(ValueError):
        occlude(x, 0, 5)


def test_attribution_of_naive_model_is_all_in_the_last_block():
    from finsight.models.lstm.explain import occlusion_attribution

    ws = data.make_windows(_series(n=60), L, H)
    table = occlusion_attribution(
        lambda x: np.repeat(x[:, -1:], H, axis=1), ws.x_raw, ws.y_raw, block=5
    )
    assert list(table["lag_from"]) == [1, 6, 11, 16]
    assert table.iloc[0]["mean_abs_change_pct"] > 0
    assert (table.iloc[1:]["mean_abs_change_pct"] == 0).all()


def test_feature_experiment_runs_and_decides_on_validation(project):
    from finsight.models.lstm import feature_experiment as fe
    from finsight.models.lstm.train import run_training

    root, path = project
    frame = pd.read_parquet(path)
    frame["volatility_20"] = frame.groupby("ticker")["price"].transform(
        lambda s: np.log(s).diff().rolling(20).std()
    )
    frame.to_parquet(path / "part-00000.parquet", index=False)
    run_training(PARAMS, path, root)
    prices, feats = fe.load_aligned(path, "volatility_20")
    base, extra = fe.build_windows(prices, feats, L, H)
    assert np.array_equal(base.x_raw, extra.x_raw) and extra.x_extra.shape[2] == 1

    mlflow.create_experiment("finsight-lstm-experiments")
    mlflow.set_experiment("finsight-lstm-experiments")
    verdict = fe.run_experiment(PARAMS, path, root, seeds=[7, 8])
    assert len(verdict["runs"]) == 4 and isinstance(verdict["recommend_feature"], bool)
    assert {r["variant"] for r in verdict["summary"]} == {"price", "price+volatility_20"}
    assert mlflow.get_run(verdict["run_id"]).data.tags["recommend_feature"] in ("true", "false")
