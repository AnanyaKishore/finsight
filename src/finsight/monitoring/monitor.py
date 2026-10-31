"""The monitoring report: data drift, live model error, data freshness and quality.

:func:`build_report` reads MongoDB (and ``reports/``) and returns one JSON-ready
document; the scheduler computes it every ``monitoring.interval_s`` and after
each run, stores it in ``monitoring_reports`` and exports it to Prometheus
(:mod:`finsight.monitoring.metrics`). Sections:

``returns``     per ticker: PSI and KS of the last ``current_days`` daily log
                returns against the ``reference_days`` before them; drift is
                *detected* when at least ``drift_ticker_share`` of tickers
                exceed ``psi_threshold`` (this alert starts a retrain).
``sentiment``   PSI / mean shift of article sentiment scores, recent vs before
                (an alert, but no retrain: the sentiment model is pretrained).
``forecast``    live error of stored LSTM forecasts whose target days have now
                traded, against the naive "price stays the same" forecast on
                the same points (this alert also starts a retrain).
``freshness``   trading days the newest price lags the latest expected session.
``quality``     failed checks in the last validation report.
``drill``       whether a drift drill is active (see :mod:`finsight.monitoring.drill`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finsight.common.mongo import FORECASTS, MONITORING_DRILL, NEWS_SENTIMENT, PRICES
from finsight.common.timeutils import is_trading_day, latest_expected_session
from finsight.monitoring.drift import ks_statistic, psi


# ----------------------------------------------------------------- loading
def load_prices(db: Any) -> pd.DataFrame:
    """Prices as a frame with the modelling price (``adj_close`` rule)."""
    rows = list(db[PRICES].find({}, {"_id": 0, "ticker": 1, "date": 1, "close": 1, "adj_close": 1}))
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "price"])
    frame = pd.DataFrame(rows)
    if "adj_close" not in frame:
        frame["adj_close"] = np.nan
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    has_all_adj = frame.groupby("ticker")["adj_close"].transform(lambda s: s.notna().all())
    frame["price"] = frame["adj_close"].where(has_all_adj, frame["close"]).astype(float)
    return frame.sort_values(["ticker", "date"])[["ticker", "date", "price"]]


def active_drill(db: Any, now: datetime) -> dict[str, Any] | None:
    """The drift drill in force, if any."""
    doc = db[MONITORING_DRILL].find_one({"_id": "drill"}, {"_id": 0})
    if not doc:
        return None
    until = doc["until"] if doc["until"].tzinfo else doc["until"].replace(tzinfo=UTC)
    return doc if until > now else None


# ------------------------------------------------------------------ returns
def returns_drift(
    prices: pd.DataFrame, params: Mapping[str, Any], drill: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """PSI / KS per ticker of recent daily log returns against the year before."""
    cur_n, ref_n = int(params["current_days"]), int(params["reference_days"])
    threshold, bins = float(params["psi_threshold"]), int(params["psi_bins"])
    per_ticker: dict[str, dict[str, Any]] = {}
    for ticker, group in prices.groupby("ticker"):
        series = group["price"].to_numpy(dtype=float)
        series = series[series > 0]
        returns = np.diff(np.log(series))
        if len(returns) < cur_n + max(bins * 3, 60):
            per_ticker[ticker] = {"status": "insufficient_data", "n": int(len(returns))}
            continue
        current = returns[-cur_n:]
        reference = returns[-cur_n - ref_n : -cur_n]
        if drill:
            current = current * float(drill.get("scale", 4.0))
        value = psi(reference, current, bins)
        per_ticker[ticker] = {
            "psi": round(value, 4),
            "ks": round(ks_statistic(reference, current), 4),
            "drifted": value > threshold,
            "current_volatility": round(float(np.std(current, ddof=1)), 5),
            "reference_volatility": round(float(np.std(reference, ddof=1)), 5),
            "n_reference": int(len(reference)),
        }
    scored = [v for v in per_ticker.values() if "psi" in v]
    share = (sum(v["drifted"] for v in scored) / len(scored)) if scored else 0.0
    return {
        "per_ticker": per_ticker,
        "drifted_share": round(share, 3),
        "max_psi": round(max((v["psi"] for v in scored), default=0.0), 4),
        "detected": bool(scored) and share >= float(params["drift_ticker_share"]),
        "threshold": threshold,
    }


# ---------------------------------------------------------------- sentiment
def sentiment_drift(
    db: Any, params: Mapping[str, Any], now: datetime, drill: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Shift of article sentiment scores: last N days against the period before."""
    cur_days = int(params["sentiment_current_days"])
    ref_days = int(params["sentiment_reference_days"])
    split = now - timedelta(days=cur_days)
    start = split - timedelta(days=ref_days)
    rows = list(
        db[NEWS_SENTIMENT].find(
            {"published_at": {"$gte": start}}, {"_id": 0, "score": 1, "published_at": 1}
        )
    )
    if not rows:
        return {"status": "insufficient_data", "n_current": 0, "n_reference": 0}
    frame = pd.DataFrame(rows)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    current = frame.loc[frame["published_at"] >= split, "score"].to_numpy(dtype=float)
    reference = frame.loc[frame["published_at"] < split, "score"].to_numpy(dtype=float)
    if len(current) < 50 or len(reference) < 100:
        return {
            "status": "insufficient_data",
            "n_current": int(len(current)),
            "n_reference": int(len(reference)),
        }
    if drill:
        current = np.clip(current + float(drill.get("sentiment_shift", -0.5)), -1, 1)
    value = psi(reference, current, int(params["psi_bins"]))
    return {
        "psi": round(value, 4),
        "ks": round(ks_statistic(reference, current), 4),
        "mean_current": round(float(current.mean()), 4),
        "mean_reference": round(float(reference.mean()), 4),
        "n_current": int(len(current)),
        "n_reference": int(len(reference)),
        "detected": value > float(params["sentiment_psi_threshold"]),
    }


# ----------------------------------------------------------------- forecast
def forecast_error(
    db: Any, prices: pd.DataFrame, params: Mapping[str, Any], now: datetime
) -> dict[str, Any]:
    """Live MAPE of stored forecasts vs the naive forecast on the same realised points."""
    since = now - timedelta(days=int(params["error_window_days"]))
    actual = {(r.ticker, r.date.date()): r.price for r in prices.itertuples(index=False)}
    errors, naive, per_ticker, versions = [], [], {}, set()
    for doc in db[FORECASTS].find({"as_of": {"$gte": since}}, {"_id": 0}):
        last = float(doc.get("last_price") or 0)
        for point in doc.get("forecast", []):
            day = pd.Timestamp(point["date"]).date()
            real = actual.get((doc["ticker"], day))
            if not real or real <= 0 or last <= 0:
                continue
            ape = abs(float(point["price"]) - real) / real * 100
            naive_ape = abs(last - real) / real * 100
            errors.append(ape)
            naive.append(naive_ape)
            per_ticker.setdefault(doc["ticker"], []).append(ape)
            versions.add(str(doc.get("model_version")))
    n = len(errors)
    report: dict[str, Any] = {"points": n, "model_versions": sorted(versions)}
    if n < int(params["min_error_points"]):
        return report | {"status": "insufficient_data", "detected": False}
    mape, naive_mape = float(np.mean(errors)), float(np.mean(naive))
    ratio = mape / naive_mape if naive_mape > 0 else None
    detected = (
        ratio is not None
        and ratio > float(params["error_ratio_to_naive"])
        and mape > float(params["error_mape_floor"])
    )
    return report | {
        "mape": round(mape, 3),
        "naive_mape": round(naive_mape, 3),
        "ratio_to_naive": round(ratio, 3) if ratio is not None else None,
        "per_ticker_mape": {t: round(float(np.mean(v)), 3) for t, v in sorted(per_ticker.items())},
        "detected": bool(detected),
    }


# --------------------------------------------------------- freshness, quality
def freshness(prices: pd.DataFrame, now: datetime) -> dict[str, Any]:
    """Trading days between each ticker's newest price and the latest expected session."""
    expected = latest_expected_session(now)
    lag: dict[str, int] = {}
    for ticker, group in prices.groupby("ticker"):
        last = group["date"].max().date()
        days, day = 0, last
        while day < expected:
            day += timedelta(days=1)
            days += is_trading_day(day)
        lag[ticker] = days
    return {
        "expected_session": expected.isoformat(),
        "lag_trading_days": lag,
        "max_lag": max(lag.values(), default=None),
    }


def data_quality(root: Path, report_path: str = "reports/data_quality.json") -> dict[str, Any]:
    """Failed checks in the last validation report (``validate`` stage)."""
    try:
        report = json.loads((root / report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "no_report"}
    failed = [c for c in report.get("checks", []) if not c.get("success", True)]
    return {
        "status": report.get("status"),
        "as_of_session": report.get("as_of_session"),
        "errors": sum(c.get("severity") == "error" for c in failed),
        "warnings": sum(c.get("severity") == "warn" for c in failed),
        "failed_checks": [c.get("name") for c in failed],
    }


def production_versions(root: Path) -> dict[str, str | None]:
    """Production model versions from ``reports/promotion.json``."""
    try:
        report = json.loads((root / "reports" / "promotion.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for model in ("lstm", "sentiment"):
        version = ((report.get(model) or {}).get("production") or {}).get("version")
        out[model] = str(version) if version is not None else None
    return out


# ------------------------------------------------------------------- report
def build_report(
    db: Any,
    root: Path,
    params: Mapping[str, Any],
    now: datetime | None = None,
    quality_report: str = "reports/data_quality.json",
) -> dict[str, Any]:
    """Compute every section; a failing section is reported, never raised."""
    now = now or datetime.now(UTC)
    drill = active_drill(db, now)
    report: dict[str, Any] = {"generated_at": now, "drill": drill}
    prices = load_prices(db)
    sections = {
        "returns": lambda: returns_drift(prices, params, drill),
        "sentiment": lambda: sentiment_drift(db, params, now, drill),
        "forecast": lambda: forecast_error(db, prices, params, now),
        "freshness": lambda: freshness(prices, now),
        "quality": lambda: data_quality(root, quality_report),
        "models": lambda: production_versions(root),
    }
    for name, compute in sections.items():
        try:
            report[name] = compute()
        except Exception as exc:  # noqa: BLE001 -- one broken section must not hide the rest
            report[name] = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"[:300]}
    return report


def summarise(report: Mapping[str, Any]) -> str:
    """One line for the run record and the logs."""
    parts = []
    r = report.get("returns") or {}
    if "per_ticker" in r:
        n = len([v for v in r["per_ticker"].values() if "psi" in v])
        drifted = sum(v.get("drifted", False) for v in r["per_ticker"].values())
        parts.append(f"returns drift on {drifted}/{n} tickers (max PSI {r['max_psi']})")
    s = report.get("sentiment") or {}
    if "psi" in s:
        parts.append(f"sentiment PSI {s['psi']}")
    f = report.get("forecast") or {}
    if "mape" in f:
        parts.append(f"live MAPE {f['mape']}% vs naive {f['naive_mape']}% (n={f['points']})")
    elif "points" in f:
        parts.append(f"live error: {f['points']} realised points so far")
    if report.get("drill"):
        parts.append("DRIFT DRILL ACTIVE")
    return "; ".join(parts) or "no data yet"
