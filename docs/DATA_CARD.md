# FinSight — Data Card

Satisfies guidelines §II.A: *document data sources, formats, and potential biases*,
and records the EDA findings that shaped the pipeline. Numbers in **Observed
characteristics** come from `notebooks/01_eda.ipynb` (section 9) and
`reports/data_quality.json` for the snapshot committed with this file.

## 1. Sources

| Dataset | Provider | Endpoint | Fields used | Refresh | Licence / terms |
|---|---|---|---|---|---|
| End-of-day prices | MarketStack (primary) | `GET /v2/eod` | open, high, low, close, adj_close, volume | daily, gap-fill | free plan, 100 req/month |
| End-of-day prices | Yahoo Finance via `yfinance` (fallback) | unofficial | same | only when MarketStack is over quota / failing | unofficial; personal / academic use |
| Company news | Finnhub | `GET /company-news` | id, datetime, headline, summary, source, url, category | daily, gap-fill | free plan, 60 calls/min |

Universe: 10 large-cap US equities fixed in `configs/tickers.yaml`
(AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, WMT, V).

## 2. Lineage and storage

```
providers ──► MongoDB (prices, news)            operational store, idempotent upserts
          ──► data/raw/*.parquet                 `export` stage, DVC-versioned snapshot
          ──► reports/data_quality.json          `validate` stage, expectation suite
          ──► data/processed/{prices,news,news_weekly}/   `process` stage (Spark)
```

* Every price row carries `source` (`marketstack` | `yfinance`); every article carries `provider`.
* A snapshot is identified by the SHA-256 hashes in `reports/export_manifest.json`
  and by `dvc.lock`; `git checkout <commit> && dvc checkout` restores it exactly.

## 3. Formats

### `data/raw/prices.parquet` (one row per ticker × trading day)
| column | type | notes |
|---|---|---|
| ticker | string | upper-case symbol |
| date | date | NYSE trading day (no time zone) |
| open, high, low, close | float64 | USD, unadjusted |
| adj_close | float64, nullable | split/dividend-adjusted close when the provider supplies it |
| volume | float64 | shares |
| source | string | provider lineage |

### `data/raw/news.parquet` (one row per ticker × article)
| column | type | notes |
|---|---|---|
| ticker, news_id | string | unique key |
| published_at | timestamp (UTC, µs) | |
| headline, summary | string | summary may be empty |
| text | string | `headline. summary`, ≤ 1000 chars — the sentiment model input |
| source, url, provider, category | string | |

### `data/processed/prices/ticker=*/part-00000.parquet`
Raw columns plus: `price` (model price, see §5), `is_filled`, `fill_gap`, `log_return`,
`day_of_week`, and for each window *n* in `etl.feature_windows` (5, 20):
`ret_mean_n`, `volatility_n`, `ma_n`, `price_to_ma_n`, `volume_ma_n` (null until a full window exists).

### `data/processed/news/ticker=*/` and `data/processed/news_weekly/`
Cleaned articles with `week_start` (Monday, UTC); weekly `article_count`,
`distinct_sources`, first/last publication time.

## 4. Known limitations and potential biases

| # | Issue | Effect | Mitigation |
|---|---|---|---|
| B1 | **Survivorship / selection bias**: the universe is today's mega-caps, chosen with hindsight. | Back-tested returns and Sharpe ratios look better than a forward-looking investor would have achieved. | Stated in the UI disclaimer; portfolio compared with an equal-weight baseline on the *same* universe. |
| B2 | **Missing or invalid bars** in the free MarketStack feed: 2–3 missing trading days per ticker, and `close = 0` (with normal volume) for META on 2026-04-07, 2026-04-08 and 2026-06-04. | Gaps distort returns and LSTM windows; a zero price produces infinite returns. | `validate` blocked the pipeline on the zero prices; the bars were re-fetched from yfinance with `ingestion repair` (run record `mode: repair`), and the parser now rejects close ≤ 0. Gaps ≤ 2 days are forward-filled and flagged `is_filled`. |
| B3 | **News sampling cap** (`ingestion.max_articles_per_window` = 40 per ticker-week), spread evenly through the week. | Busy weeks are under-sampled, so article counts saturate at 40 and weekly sentiment is an estimate. | Cap documented; the chart shows the sampled count. |
| B4 | **News source mix and syndication**: whatever Finnhub aggregates (7 sources); the same story often appears under several IDs. | Sentiment reflects that editorial mix, and duplicates would over-weight syndicated stories. | Top sources reported in the EDA; the ETL removes duplicates by normalised headline (156 removed in this snapshot). |
| B5 | **Domain mismatch of the sentiment model** (DistilBERT fine-tuned on SST-2 movie reviews). | Financial language ("beats estimates", "cuts guidance") may be misread; no neutral class. | Measured on a labelled financial-news sample on Day 4 (accuracy, F1). |
| B6 | **Mixed providers** after fallback (MarketStack vs Yahoo adjusted closes can differ slightly). | Small artificial jumps at the switch. | `source` lineage + provider-mix check; model price rule (§5). |
| B7 | **Unadjusted corporate actions** after the backfill (a split stored by incremental runs). | A −50 % / −75 % "return" corrupts training. | `prices.no_extreme_returns` is a *blocking* check; remedy: `ingestion backfill --prices-only` refreshes adjusted history. |
| B8 | **Non-stationarity**: volatility regimes change over the year. | A model trained once decays (the proposal's core problem). | Drift monitoring and retraining (Day 8); EDA §6 quantifies it. |
| B9 | Short history (12 months, free-plan limit). | Few independent LSTM windows; rare events under-represented. | One global model across tickers; naive baseline comparison. |

## 5. Processing rules

* **Model price**: per ticker, `adj_close` if every bar has it, otherwise `close` (never mixed within a series).
* **Calendar**: NYSE trading days from `finsight.common.timeutils` (weekends + full-day holidays).
* **De-duplication**: prices by (ticker, date) at ingestion; news by (ticker, news_id) and by normalised headline in the ETL.
* **Quality gate**: blocking expectations — schema, required nulls, unique keys, positive prices, universe complete, ≥ 114 rows per ticker (100-day lookback + 14-day horizon), ≤ 10 missing days, no |log return| > 0.5.

## 6. Observed characteristics

Snapshot: `reports/export_manifest.json` sha256 prices = `9200f049c246163a1b7dfd148af0121160ec753ac216ca12909d64546cd0117e` (22 Oct 2026)

- Snapshot: 2025-10-21 to 2026-10-21, 2,492 price rows, 10,756 articles, 10 tickers.
- Missing trading days per ticker: min 2, max 3 (total 28); all gaps ≤ 2 consecutive days, so all 28 were forward-filled (2,520 rows after alignment = 10 × 252 trading days).
- Price provider mix: 2,489 MarketStack, 3 yfinance (repaired bars, see B2).
- Annualised volatility range: 22.0% (V) to 45.2% (TSLA).
- Return kurtosis range: 3.6 to 14.2 (normal = 3): every ticker has fat tails, which justifies the robust (median/MAD) outlier rule.
- Mean pairwise return correlation: 0.16 (low; diversification is possible).
- Largest daily move: TSLA −15.7% on 2026-07-23 (real move; flagged as an outlier warning, not blocked).
- Volatility regime shift (2nd half / 1st half): 0.81× to 1.27× (non-stationarity, see B8).
- News: 39.8 articles per ticker-week on average against a cap of 40, so the cap is active almost every week (see B3); 7 distinct sources; 156 syndicated duplicate headlines removed by the ETL.
- Data-quality warnings (non-blocking): 14 OHLC-inconsistent bars (high/low outside the open–close range; only close is used for modelling), 7 return outliers.
