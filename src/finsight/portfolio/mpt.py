"""Modern Portfolio Theory primitives: grid of weights, statistics, optimisation, backtest.

Method (proposal: "grid search over long-only weight combinations on recent
price history (last 30 days); maximises the Sharpe ratio")
---------------------------------------------------------------------------
* **Inputs** -- simple daily returns of the last ``window`` trading days.
  Expected return ``mu = mean * 252`` and covariance ``Sigma = cov * 252``
  (annualised, sample covariance).
* **Grid** -- *every* long-only weight vector whose weights are multiples of
  ``step`` and sum to 1. For 10 stocks at ``step = 0.1`` that is
  ``C(19, 9) = 92,378`` portfolios, evaluated exactly in one vectorised pass
  (a few milliseconds); the baseline repo sampled random weights instead.
* **Objective** -- Sharpe ratio ``(w.mu - r_f) / sqrt(w' Sigma w)``.
* **Degenerate case** -- if *no* portfolio has a positive excess return (a
  falling market), maximising Sharpe would pick the *most volatile*
  portfolio (a negative numerator divided by a larger volatility is closer to zero).
  The optimiser then falls back to the minimum-volatility portfolio and says so.
* **Grid resolution check** -- a continuous optimiser (SciPy SLSQP) solves the
  same problem; the Sharpe gap shows what the 10 % grid gives up.
* **Out-of-sample test** -- a walk-forward backtest re-optimises every
  ``rebalance`` days on the trailing window and holds the weights until the
  next rebalance, compared with equal weight and minimum volatility.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MAX_SHARPE, MIN_VOLATILITY, EQUAL_WEIGHT = "max_sharpe", "min_volatility", "equal_weight"


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def grid_size(n_assets: int, step: float) -> int:
    """Number of long-only portfolios with weights in multiples of ``step``."""
    units = round(1 / step)
    return math.comb(units + n_assets - 1, n_assets - 1)


def weight_grid(n_assets: int, step: float) -> np.ndarray:
    """All long-only weight vectors on the ``step`` lattice (stars and bars).

    Returns:
        Array of shape ``(grid_size, n_assets)``; rows sum to exactly 1.

    Raises:
        ValueError: If ``1 / step`` is not an integer or the grid is too large.
    """
    units = round(1 / step)
    if not math.isclose(units * step, 1.0, abs_tol=1e-9):
        raise ValueError(f"1 / step must be an integer, got step={step}")
    if grid_size(n_assets, step) > 2_000_000:
        raise ValueError("grid too large for this machine; use a coarser step")
    if n_assets == 1:
        return np.ones((1, 1))
    slots = units + n_assets - 1
    bars = np.array(list(itertools.combinations(range(slots), n_assets - 1)), dtype=np.int64)
    edges = np.hstack([np.full((len(bars), 1), -1), bars, np.full((len(bars), 1), slots)])
    return (np.diff(edges, axis=1) - 1) / units


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def annualised_moments(returns: pd.DataFrame, periods: int) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, Sigma)`` annualised from simple daily returns."""
    mu = returns.mean().to_numpy() * periods
    cov = returns.cov(ddof=1).to_numpy() * periods
    return mu, cov


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple:
    """Expected return and volatility for one (1-D) or many (2-D) weight vectors."""
    w = np.atleast_2d(weights)
    ret = w @ mu
    var = np.einsum("ij,jk,ik->i", w, cov, w)
    vol = np.sqrt(np.maximum(var, 0.0))
    if np.ndim(weights) == 1:
        return float(ret[0]), float(vol[0])
    return ret, vol


def sharpe_ratio(ret: Any, vol: Any, rf: float) -> Any:
    """Sharpe ratio; zero-volatility portfolios get ``-inf`` (never selected)."""
    ret, vol = np.asarray(ret, dtype="float64"), np.asarray(vol, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(vol > 1e-12, (ret - rf) / vol, -np.inf)
    return out if out.ndim else float(out)


@dataclass(frozen=True)
class Allocation:
    """One optimised portfolio."""

    method: str
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float

    def as_dict(self, tickers: list[str]) -> dict[str, Any]:
        """JSON-friendly representation (weights rounded to the grid)."""
        return {
            "method": self.method,
            "weights": {t: round(float(w), 4) for t, w in zip(tickers, self.weights, strict=True)},
            "expected_return": round(self.expected_return, 6),
            "volatility": round(self.volatility, 6),
            "sharpe": round(self.sharpe, 6),
        }


def _allocation(
    method: str, w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float
) -> Allocation:
    ret, vol = portfolio_stats(w, mu, cov)
    return Allocation(
        method, np.asarray(w, dtype="float64"), ret, vol, float(sharpe_ratio(ret, vol, rf))
    )


def optimise(
    returns: pd.DataFrame, grid: np.ndarray, rf: float, periods: int, fallback: str
) -> dict[str, Any]:
    """Evaluate the whole grid on one return window.

    Returns:
        Dict with ``selected`` (max Sharpe, or the fallback), ``max_sharpe``,
        ``min_volatility`` and ``equal_weight`` allocations, the grid's
        ``ret``/``vol``/``sharpe`` arrays, ``mu``/``cov`` and a ``fallback_used`` flag.
    """
    mu, cov = annualised_moments(returns, periods)
    ret, vol = portfolio_stats(grid, mu, cov)
    sharpe = sharpe_ratio(ret, vol, rf)
    best = int(np.argmax(sharpe))  # first index on ties -> deterministic
    least = int(np.argmin(vol))
    n = grid.shape[1]
    out = {
        "mu": mu,
        "cov": cov,
        "ret": ret,
        "vol": vol,
        "sharpe": sharpe,
        MAX_SHARPE: _allocation(MAX_SHARPE, grid[best], mu, cov, rf),
        MIN_VOLATILITY: _allocation(MIN_VOLATILITY, grid[least], mu, cov, rf),
        EQUAL_WEIGHT: _allocation(EQUAL_WEIGHT, np.full(n, 1 / n), mu, cov, rf),
    }
    fallback_used = sharpe[best] <= 0 and fallback == MIN_VOLATILITY
    out["fallback_used"] = bool(fallback_used)
    out["selected"] = out[MIN_VOLATILITY] if fallback_used else out[MAX_SHARPE]
    return out


def continuous_max_sharpe(
    mu: np.ndarray, cov: np.ndarray, rf: float, starts: list[np.ndarray]
) -> dict[str, Any]:
    """Long-only maximum-Sharpe portfolio by SLSQP (grid-resolution check).

    Several starting points guard against a poor local optimum.
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    n = len(mu)

    def negative_sharpe(w: np.ndarray) -> float:
        ret, vol = portfolio_stats(w, mu, cov)
        return -(ret - rf) / max(vol, 1e-12)

    best: Any = None
    for start in starts:
        res = minimize(
            negative_sharpe,
            np.asarray(start, dtype="float64"),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return {"success": False}
    w = np.clip(best.x, 0, None)
    w = w / w.sum()
    ret, vol = portfolio_stats(w, mu, cov)
    return {
        "success": True,
        "weights": w,
        "expected_return": ret,
        "volatility": vol,
        "sharpe": float(sharpe_ratio(ret, vol, rf)),
    }


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
def walk_forward(
    returns: pd.DataFrame,
    grid: np.ndarray,
    *,
    window: int,
    rebalance: int,
    rf: float,
    periods: int,
    fallback: str,
) -> dict[str, Any]:
    """Re-optimise every ``rebalance`` days on the trailing ``window``; hold until the next.

    Each strategy's weights are fixed for the holding period (daily
    rebalancing to target weights within the period, a standard simplification).
    No information from the holding period is used to choose its weights.

    Returns:
        ``{"daily": DataFrame[date x strategy] of out-of-sample returns,
        "rebalances": [...], "turnover": {strategy: mean one-way turnover}}``.

    Raises:
        ValueError: If there is not enough history for one rebalance.
    """
    if len(returns) <= window:
        raise ValueError(f"need more than {window} daily returns, have {len(returns)}")
    strategies = (MAX_SHARPE, MIN_VOLATILITY, EQUAL_WEIGHT)
    daily = {s: [] for s in strategies}
    dates: list[Any] = []
    rebalances, previous = [], {}
    turnover: dict[str, list[float]] = {s: [] for s in strategies}
    values = returns.to_numpy()
    for t in range(window, len(returns), rebalance):
        result = optimise(returns.iloc[t - window : t], grid, rf, periods, fallback)
        hold = values[t : t + rebalance]
        chosen = {
            MAX_SHARPE: result["selected"].weights,
            MIN_VOLATILITY: result[MIN_VOLATILITY].weights,
            EQUAL_WEIGHT: result[EQUAL_WEIGHT].weights,
        }
        for name, w in chosen.items():
            daily[name].extend((hold @ w).tolist())
            if name in previous:
                turnover[name].append(float(np.abs(w - previous[name]).sum() / 2))
            previous[name] = w
        dates.extend(returns.index[t : t + rebalance])
        rebalances.append(
            {
                "date": str(returns.index[t])[:10],
                "in_sample_sharpe": round(result["selected"].sharpe, 4),
                "fallback_used": result["fallback_used"],
                "weights": {
                    c: round(float(w), 4)
                    for c, w in zip(returns.columns, chosen[MAX_SHARPE], strict=True)
                },
            }
        )
    frame = pd.DataFrame(daily, index=pd.Index(dates, name="date"))
    return {
        "daily": frame,
        "rebalances": rebalances,
        "turnover": {s: round(float(np.mean(v)), 4) if v else 0.0 for s, v in turnover.items()},
    }


def performance(daily: pd.Series, rf: float, periods: int) -> dict[str, float]:
    """Annualised return / volatility / Sharpe, total return and max drawdown."""
    r = daily.to_numpy(dtype="float64")
    growth = np.cumprod(1 + r)
    years = len(r) / periods
    ann_return = float(growth[-1] ** (1 / years) - 1) if years > 0 else 0.0
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(periods)) if len(r) > 1 else 0.0
    excess = r - rf / periods
    sharpe = (
        float(np.mean(excess) / np.std(r, ddof=1) * np.sqrt(periods))
        if len(r) > 1 and np.std(r, ddof=1) > 0
        else 0.0
    )
    peak = np.maximum.accumulate(growth)
    return {
        "annual_return": round(ann_return, 6),
        "annual_volatility": round(ann_vol, 6),
        "sharpe": round(sharpe, 6),
        "total_return": round(float(growth[-1] - 1), 6),
        "max_drawdown": round(float(np.min(growth / peak - 1)), 6),
        "days": len(r),
    }


def frontier(ret: np.ndarray, vol: np.ndarray, bins: int = 40) -> pd.DataFrame:
    """Upper envelope of the grid cloud: best return per volatility bin."""
    edges = np.linspace(vol.min(), vol.max(), bins + 1)
    which = np.clip(np.digitize(vol, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        members = np.flatnonzero(which == b)
        if len(members):
            i = members[np.argmax(ret[members])]
            rows.append(
                {"volatility": round(float(vol[i]), 6), "expected_return": round(float(ret[i]), 6)}
            )
    out = pd.DataFrame(rows)
    # Keep only the efficient part (return increasing with volatility).
    return out[out["expected_return"] >= out["expected_return"].cummax()].reset_index(drop=True)
