"""Tests for the MPT portfolio optimiser: grid, statistics, fallback, backtest, stage."""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from finsight.portfolio import mpt
from finsight.portfolio import optimize as opt
from tests.factories import make_prices

TICKERS = ["A", "B", "C", "D"]


def _returns(n=60, seed=0, drift=(0.002, 0.0005, 0.0, -0.001), vol=(0.01, 0.02, 0.015, 0.03)):
    rng = np.random.default_rng(seed)
    data = rng.normal(drift, vol, size=(n, len(drift)))
    return pd.DataFrame(
        data, columns=TICKERS[: len(drift)], index=pd.bdate_range("2026-01-01", periods=n)
    )


# --------------------------------------------------------------------- grid
def test_full_grid_for_ten_stocks_is_92378_portfolios():
    assert mpt.grid_size(10, 0.1) == math.comb(19, 9) == 92_378
    grid = mpt.weight_grid(10, 0.1)
    assert grid.shape == (92_378, 10)
    assert np.allclose(grid.sum(axis=1), 1.0) and (grid >= 0).all()
    assert np.allclose(np.round(grid * 10), grid * 10)  # multiples of 0.1
    assert len(np.unique(grid.round(6), axis=0)) == len(grid)


def test_small_grid_is_exactly_the_lattice():
    grid = mpt.weight_grid(3, 0.5)
    expected = {(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)}
    assert {tuple(row) for row in grid.tolist()} == expected
    assert mpt.weight_grid(1, 0.1).tolist() == [[1.0]]


def test_grid_rejects_bad_step():
    with pytest.raises(ValueError):
        mpt.weight_grid(3, 0.3)


# --------------------------------------------------------------- statistics
def test_portfolio_stats_match_manual_formula():
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    ret, vol = mpt.portfolio_stats(np.array([0.5, 0.5]), mu, cov)
    assert ret == pytest.approx(0.15)
    assert vol == pytest.approx(math.sqrt(0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.01))
    rets, vols = mpt.portfolio_stats(np.array([[1, 0], [0, 1]]), mu, cov)
    assert np.allclose(rets, mu) and np.allclose(vols, [0.2, 0.3])
    assert mpt.sharpe_ratio(0.1, 0.0, 0.02) == -np.inf


def test_optimise_finds_grid_maximum_and_beats_equal_weight_in_sample():
    returns = _returns()
    grid = mpt.weight_grid(4, 0.1)
    result = mpt.optimise(returns, grid, rf=0.0, periods=252, fallback="min_volatility")
    best = result["max_sharpe"]
    assert best.sharpe == pytest.approx(result["sharpe"].max())
    assert best.sharpe >= result["equal_weight"].sharpe
    assert result["min_volatility"].volatility == pytest.approx(result["vol"].min())
    assert not result["fallback_used"] and result["selected"] is best
    assert np.isclose(best.weights.sum(), 1.0)


def test_falling_market_falls_back_to_min_volatility():
    returns = _returns(drift=(-0.01, -0.012, -0.015, -0.011), vol=(0.005, 0.006, 0.007, 0.005))
    grid = mpt.weight_grid(4, 0.25)
    result = mpt.optimise(returns, grid, rf=0.04, periods=252, fallback="min_volatility")
    assert result["fallback_used"] and result["selected"].method == "min_volatility"
    plain = mpt.optimise(returns, grid, rf=0.04, periods=252, fallback="none")
    assert not plain["fallback_used"] and plain["selected"].method == "max_sharpe"


def test_continuous_optimiser_is_at_least_as_good_as_the_grid():
    returns = _returns()
    grid = mpt.weight_grid(4, 0.1)
    result = mpt.optimise(returns, grid, rf=0.0, periods=252, fallback="none")
    cont = mpt.continuous_max_sharpe(
        result["mu"],
        result["cov"],
        0.0,
        [result["equal_weight"].weights, result["max_sharpe"].weights],
    )
    assert cont["success"]
    gap = cont["sharpe"] - result["max_sharpe"].sharpe
    assert -1e-6 <= gap < 0.5  # never worse than the grid; grid is close
    assert np.isclose(cont["weights"].sum(), 1) and (cont["weights"] >= 0).all()


# ----------------------------------------------------------------- backtest
def test_walk_forward_never_looks_ahead():
    returns = _returns(n=80)
    grid = mpt.weight_grid(4, 0.25)
    kwargs = {"window": 30, "rebalance": 5, "rf": 0.0, "periods": 252, "fallback": "none"}
    base = mpt.walk_forward(returns, grid, **kwargs)
    assert len(base["daily"]) == len(returns) - 30
    assert base["daily"].index[0] == returns.index[30]
    # Perturb everything from day 50 on: weights chosen before day 50 must not change.
    shocked = returns.copy()
    shocked.iloc[50:] = shocked.iloc[50:] * -3
    other = mpt.walk_forward(shocked, grid, **kwargs)
    early = [r for r in base["rebalances"] if r["date"] <= str(returns.index[50])[:10]]
    assert early and all(a == b for a, b in zip(early, other["rebalances"], strict=False))
    assert set(base["turnover"]) == {"max_sharpe", "min_volatility", "equal_weight"}
    assert base["turnover"]["equal_weight"] == 0.0
    with pytest.raises(ValueError):
        mpt.walk_forward(returns.iloc[:30], grid, **kwargs)


def test_performance_known_values():
    daily = pd.Series([0.1, -0.5, 0.2])
    perf = mpt.performance(daily, rf=0.0, periods=252)
    assert perf["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert perf["max_drawdown"] == pytest.approx(-0.5)
    assert perf["days"] == 3


def test_frontier_is_monotone():
    returns = _returns()
    grid = mpt.weight_grid(4, 0.1)
    result = mpt.optimise(returns, grid, rf=0.0, periods=252, fallback="none")
    env = mpt.frontier(result["ret"], result["vol"])
    assert env["volatility"].is_monotonic_increasing
    assert env["expected_return"].is_monotonic_increasing


# -------------------------------------------------------------------- stage
PORT_PARAMS = {
    "window_days": 30,
    "weight_step": 0.25,
    "risk_free_rate": 0.039,
    "trading_days_per_year": 252,
    "rebalance_days": 5,
    "no_positive_sharpe_fallback": "min_volatility",
    "experiment_name": "test-portfolio",
    "report": "reports/portfolio.json",
    "backtest_report": "reports/portfolio_backtest.json",
    "prices_path": "data/processed/prices",
}


@pytest.fixture
def prices_dir(tmp_path):
    prices = make_prices(
        tickers=("AAPL", "MSFT", "NVDA"), start=date(2026, 5, 1), end=date(2026, 9, 30)
    )
    prices["price"] = prices["adj_close"]
    # One ticker misses a day: that date must be dropped for all tickers.
    prices = prices[~((prices["ticker"] == "NVDA") & (prices["date"] == date(2026, 6, 2)))]
    path = tmp_path / "data/processed/prices"
    path.mkdir(parents=True)
    prices[["ticker", "date", "price"]].to_parquet(path / "part-0.parquet", index=False)
    return path


def test_load_returns_aligns_dates(prices_dir):
    returns = opt.load_returns(prices_dir)
    assert list(returns.columns) == ["AAPL", "MSFT", "NVDA"]
    assert pd.Timestamp("2026-06-02") not in returns.index and not returns.isna().any().any()


def test_run_optimisation_is_deterministic_and_complete(prices_dir):
    a = opt.run_optimisation(PORT_PARAMS, prices_dir)["report"]
    b = opt.run_optimisation(PORT_PARAMS, prices_dir)["report"]
    a.pop("grid_seconds"), b.pop("grid_seconds")
    assert a == b
    assert a["grid_size"] == mpt.grid_size(3, 0.25)
    assert sum(a["allocation"]["weights"].values()) == pytest.approx(1.0)
    assert set(a["walk_forward"]["performance"]) == {"max_sharpe", "min_volatility", "equal_weight"}
    assert a["continuous_check"]["grid_sharpe_gap"] >= -1e-6
    assert json.dumps(a)  # JSON-serialisable for the report and MongoDB
    compact = opt.compact(a)
    assert compact["method"] in ("max_sharpe", "min_volatility")


def test_run_optimisation_needs_a_full_window(prices_dir):
    with pytest.raises(ValueError):
        opt.run_optimisation({**PORT_PARAMS, "window_days": 10_000}, prices_dir)


def test_publish_upserts_one_document_per_day(db, prices_dir):
    from finsight.common.mongo import PORTFOLIOS

    report = opt.run_optimisation(PORT_PARAMS, prices_dir)["report"]
    opt.publish(db, report, "run-1")
    opt.publish(db, report, "run-2")
    assert db[PORTFOLIOS].count_documents({}) == 1
    doc = db[PORTFOLIOS].find_one({})
    assert doc["mlflow_run_id"] == "run-2" and doc["allocation"]["weights"]


def test_figures_and_mlflow_logging(prices_dir, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    outcome = opt.run_optimisation(PORT_PARAMS, prices_dir)
    points = opt.frontier_points(outcome, sample=50)
    assert {"grid sample", "efficient frontier"} == set(points["series"])
    curve = opt.backtest_curve(outcome)
    assert list(curve.columns[:1]) == ["date"] and (curve.drop(columns="date") > 0).all().all()
    figures = {
        "allocation": opt.allocation_figure(outcome["report"]),
        "frontier": opt.frontier_figure(outcome, points),
        "backtest": opt.backtest_figure(curve),
    }
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test-portfolio", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment("test-portfolio")
    try:
        run_id = opt.log_to_mlflow(PORT_PARAMS, outcome, tmp_path, figures)
        run = mlflow.get_run(run_id)
        assert run.data.params["grid_size"] == str(outcome["report"]["grid_size"])
        assert "oos_max_sharpe_sharpe" in run.data.metrics and "weight_AAPL" in run.data.metrics
        assert run.data.tags["stage"] == "optimize_portfolio"
    finally:
        mlflow.set_tracking_uri("")
