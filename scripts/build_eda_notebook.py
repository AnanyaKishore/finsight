"""Generate notebooks/01_eda.ipynb (kept as code so the notebook is reviewable in diffs).

python scripts/build_eda_notebook.py
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# FinSight — Exploratory Data Analysis (Day 2)

**Purpose (guidelines §II.A):** understand the data before modelling — coverage,
distributions, patterns, missing values and outliers — and record the findings
in `docs/DATA_CARD.md`.

**Input:** the DVC-versioned snapshot `data/raw/{prices,news}.parquet` produced by the
`export` stage, plus `reports/data_quality.json` from `validate`. Run `dvc repro` first.
Figures are written to `reports/figures/` (tracked with Git LFS).
""")

code("""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

prices = pd.read_parquet(ROOT / "data/raw/prices.parquet")
news = pd.read_parquet(ROOT / "data/raw/news.parquet")
quality = json.loads((ROOT / "reports/data_quality.json").read_text())
prices["date"] = pd.to_datetime(prices["date"])
has_all_adj = prices.groupby("ticker")["adj_close"].transform(lambda s: s.notna().all())
prices["price"] = prices["adj_close"].where(has_all_adj, prices["close"])
prices = prices.sort_values(["ticker", "date"])
prices["log_return"] = prices.groupby("ticker")["price"].transform(lambda s: np.log(s / s.shift()))
TICKERS = sorted(prices["ticker"].unique())
print(f"{len(prices):,} price rows, {len(news):,} articles, {len(TICKERS)} tickers")
print("Data-quality status:", quality["status"], quality["summary"])
""")

md("## 1. Coverage and missing trading days")
code("""
coverage = pd.DataFrame(quality["per_ticker"]).T[[
    "rows", "first_date", "last_date", "expected_trading_days", "missing_trading_days",
    "sessions_behind", "price_field", "sources", "news_articles", "news_thin_weeks"]]
coverage
""")
code("""
missing = {t: v["missing_dates"] for t, v in quality["per_ticker"].items() if v.get("missing_dates")}
print("Trading days absent per ticker (forward-filled by the Spark ETL if <= etl.max_ffill_days):")
for ticker, dates in missing.items():
    print(f"  {ticker:<6} {', '.join(dates)}")
if not missing:
    print("  none")
""")

md("## 2. Price paths (normalised to 1.0 at the start)")
code("""
fig, ax = plt.subplots(figsize=(11, 5))
for ticker, g in prices.groupby("ticker"):
    ax.plot(g["date"], g["price"] / g["price"].iloc[0], label=ticker, linewidth=1.2)
ax.set_title("Growth of $1 over the snapshot window")
ax.set_ylabel("Normalised price")
ax.legend(ncol=5, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(FIG / "eda_price_paths.png")
""")

md("""
## 3. Daily log-return distributions

Fat tails (kurtosis > 3) are expected for equities and are why the validation
suite flags outliers with a *robust* z-score (median/MAD) instead of mean/std.
""")
code("""
stats = prices.groupby("ticker")["log_return"].agg(
    mean="mean", std="std", skew="skew", kurtosis=lambda s: s.kurt() + 3, min="min", max="max")
stats["annualised_vol"] = stats["std"] * np.sqrt(252)
stats["annualised_return"] = stats["mean"] * 252
stats.round(4)
""")
code("""
fig, axes = plt.subplots(2, 5, figsize=(14, 5), sharex=True)
for ax, ticker in zip(axes.flat, TICKERS):
    r = prices.loc[prices["ticker"] == ticker, "log_return"].dropna()
    ax.hist(r, bins=40, color="#3b6ea8", alpha=0.85)
    ax.set_title(ticker, fontsize=9)
fig.suptitle("Daily log returns")
fig.tight_layout(); fig.savefig(FIG / "eda_return_hist.png")
""")

md("## 4. Largest moves (outlier review)")
code("""
top = prices.dropna(subset=["log_return"]).reindex(
    prices["log_return"].abs().sort_values(ascending=False).index).head(10)
top[["ticker", "date", "price", "log_return", "source"]]
""")

md("""
## 5. Correlation of daily returns

High correlation between tickers limits diversification — directly relevant to
the portfolio optimiser (Day 4).
""")
code("""
wide = prices.pivot(index="date", columns="ticker", values="log_return")
corr = wide.corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, fontsize=8)
ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Return correlation"); ax.grid(False)
fig.tight_layout(); fig.savefig(FIG / "eda_return_correlation.png")
print("Mean pairwise correlation:", round(corr.values[np.triu_indices(len(corr), 1)].mean(), 3))
""")

md("""
## 6. Rolling volatility — evidence of non-stationarity

Changing volatility regimes are the data-drift the proposal's monitoring
(Day 8) must detect.
""")
code("""
prices["vol_20"] = prices.groupby("ticker")["log_return"].transform(
    lambda s: s.rolling(20).std() * np.sqrt(252))
fig, ax = plt.subplots(figsize=(11, 4.5))
for ticker, g in prices.groupby("ticker"):
    ax.plot(g["date"], g["vol_20"], linewidth=1, label=ticker)
ax.set_title("20-day rolling annualised volatility"); ax.legend(ncol=5, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(FIG / "eda_rolling_volatility.png")
half = prices["date"].min() + (prices["date"].max() - prices["date"].min()) / 2
drift = prices.assign(half=np.where(prices["date"] < half, "first_half", "second_half")).groupby(
    ["ticker", "half"])["log_return"].std().unstack() * np.sqrt(252)
drift["ratio"] = drift["second_half"] / drift["first_half"]
drift.round(3)
""")

md("## 7. Trading volume")
code("""
vol = prices.groupby("ticker")["volume"].median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(vol.index, vol.values / 1e6, color="#3b6ea8")
ax.set_ylabel("Median daily volume (millions)"); ax.set_title("Liquidity by ticker")
fig.tight_layout(); fig.savefig(FIG / "eda_volume.png")
""")

md("## 8. News coverage")
code("""
news["published_at"] = pd.to_datetime(news["published_at"], utc=True)
news["week"] = (news["published_at"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time)
weekly = news.groupby(["week", "ticker"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(weekly.T, aspect="auto", cmap="Blues", vmin=0, vmax=max(1, weekly.values.max()))
fig.colorbar(im, ax=ax, shrink=0.8, label="articles")
ax.set_yticks(range(len(weekly.columns)), weekly.columns, fontsize=8)
step = max(1, len(weekly) // 10)
ax.set_xticks(range(0, len(weekly), step), [d.strftime("%d %b") for d in weekly.index[::step]],
              fontsize=8)
ax.set_title("Articles per ticker per week (cap: ingestion.max_articles_per_window)"); ax.grid(False)
fig.tight_layout(); fig.savefig(FIG / "eda_news_weekly.png")
print(weekly.describe().T[["mean", "min", "max"]].round(1))
""")
code("""
print("Top sources:")
print(news["source"].value_counts().head(10).to_string())
lengths = news["text"].str.split().str.len()
print("\\nWords per article text: median", int(lengths.median()), "| p95", int(lengths.quantile(0.95)))
print("Share with a summary:", round((news["summary"].fillna("").str.strip() != "").mean(), 3))
""")

md("""
## 9. Findings summary

The cell below prints the numbers to copy into `docs/DATA_CARD.md` (section *Observed
characteristics*).
""")
code("""
c = coverage
print(f"- Snapshot: {prices['date'].min().date()} to {prices['date'].max().date()}, "
      f"{len(prices):,} price rows, {len(news):,} articles, {len(TICKERS)} tickers.")
print(f"- Missing trading days per ticker: min {int(c['missing_trading_days'].min())}, "
      f"max {int(c['missing_trading_days'].max())} (total {int(c['missing_trading_days'].sum())}).")
print(f"- Price provider mix: {prices['source'].value_counts().to_dict()}.")
print(f"- Annualised volatility range: {stats['annualised_vol'].min():.1%} "
      f"({stats['annualised_vol'].idxmin()}) to {stats['annualised_vol'].max():.1%} "
      f"({stats['annualised_vol'].idxmax()}).")
print(f"- Return kurtosis range: {stats['kurtosis'].min():.1f} to {stats['kurtosis'].max():.1f} "
      "(normal distribution = 3; above 3 means fat tails).")
print(f"- Mean pairwise return correlation: {corr.values[np.triu_indices(len(corr), 1)].mean():.2f}.")
print(f"- Largest daily move: {top.iloc[0]['ticker']} {top.iloc[0]['log_return']:+.1%} "
      f"on {top.iloc[0]['date'].date()}.")
print(f"- Volatility regime shift (2nd half / 1st half): {drift['ratio'].min():.2f}x to "
      f"{drift['ratio'].max():.2f}x.")
print(f"- News: {weekly.values.mean():.1f} articles per ticker-week on average; "
      f"{news['source'].nunique()} distinct sources.")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (finsight)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = ROOT / "notebooks" / "01_eda.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f"written {out}")
