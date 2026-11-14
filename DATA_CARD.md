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
| Labelled financial sentences (evaluation only) | Financial PhraseBank v1.0 (Malo et al., 2014) via Hugging Face `takala/financial_phrasebank@8d3fe0c3` | raw ZIP, `Sentences_AllAgree.txt` | sentence, label (positive / negative / neutral) | fixed (pinned commit) | CC BY-NC-SA 3.0 (non-commercial; academic use) |
| Pretrained model | `distilbert/distilbert-base-uncased-finetuned-sst-2-english@714eb0fa` | Hugging Face Hub | weights, tokenizer | fixed (pinned commit) | Apache-2.0 |
| Pretrained model (challenger) | `ProsusAI/finbert@0bc2ac10` (BERT-base, fine-tuned on Financial PhraseBank) | Hugging Face Hub | weights, tokenizer | fixed (pinned commit) | CC BY-NC-SA 4.0 (non-commercial; academic use) |
| Labelled finance headlines (promotion gate) | `zeroshot/twitter-financial-news-sentiment@ccbe24de`, `sent_valid.csv` | Hugging Face Hub | text, label (bearish / bullish / neutral → negative / positive / neutral) | fixed (pinned commit) | MIT |

Universe: 10 large-cap US equities fixed in `configs/tickers.yaml`
(AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, WMT, V).

## 2. Lineage and storage

```
providers ──► MongoDB (prices, news)            operational store, idempotent upserts
          ──► data/raw/*.parquet                 `export` stage, DVC-versioned snapshot
          ──► reports/data_quality.json          `validate` stage, expectation suite
          ──► data/processed/{prices,news,news_weekly}/   `process` stage (Spark)
          ──► data/sentiment/{article_scores,weekly}.parquet `score_sentiment` ──► MongoDB news_sentiment, sentiment_weekly
          ──► reports/portfolio*.json            `optimize_portfolio` ──► MongoDB portfolios
Hugging Face Hub ──► models/sentiment/base/<candidate>   `fetch_sentiment@<candidate>` (once)
                 ──► data/external/{twitter_financial_news,financial_phrasebank}.csv   `fetch_sentiment_data` (once)
evaluate_sentiment@<candidate> ──► register (MLflow registry gate) ──► reports/promotion.json ──► score_sentiment
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

### `data/external/financial_phrasebank.csv` (evaluation only)
`id`, `sentence`, `label` (positive / negative / neutral), `agreement` (`AllAgree`).
2,264 sentences: 1,391 neutral, 570 positive, 303 negative (100 % annotator agreement).

### `data/external/twitter_financial_news.csv` (gate dataset)
`id`, `sentence` (URLs removed, whitespace collapsed, duplicates dropped), `label`.
Neither candidate model was trained on it; half calibrates the neutral band, half decides promotion.

### `data/sentiment/article_scores.parquet` (one row per cleaned article)
News fields plus `text_sha1`, `model_version` (base commit + neutral band + max length),
`score` = p(positive) − p(negative) ∈ [−1, 1], `confidence` = top-class probability (0.5–1 for two-label, 1/3–1 for three-label models),
`label` (positive / negative / neutral, neutral if |score| < calibrated band).

### `data/sentiment/weekly.parquet` / MongoDB `sentiment_weekly` (one row per ticker × week)
`article_count`, `mean_score`, `positive` / `negative` / `neutral` counts,
`top_positive` / `top_negative` (≤ 3 headlines each with score, source, url: the explanation).

### MongoDB `portfolios` (one document per `as_of` trading day)
Selected allocation (weights, expected return, volatility, Sharpe, method), in-sample
comparison with equal weight and min volatility, SciPy check, walk-forward performance, MLflow run ID.

## 4. Known limitations and potential biases

| # | Issue | Effect | Mitigation |
|---|---|---|---|
| B1 | **Survivorship / selection bias**: the universe is today's mega-caps, chosen with hindsight. | Back-tested returns and Sharpe ratios look better than a forward-looking investor would have achieved. | Stated in the UI disclaimer; portfolio compared with an equal-weight baseline on the *same* universe. |
| B2 | **Missing trading days** in the free MarketStack feed. | Gaps distort returns and LSTM windows. | Detected by `validate` (per-ticker dates listed); ≤ 2 consecutive days forward-filled and flagged `is_filled`; longer gaps excluded. |
| B3 | **News sampling cap** (`ingestion.max_articles_per_window` = 40 per ticker-week), spread evenly through the week. | Busy weeks are under-sampled, so article counts saturate at 40 and weekly sentiment is an estimate. | Cap documented; the chart shows the sampled count. |
| B4 | **News source mix** is whatever Finnhub aggregates (wire services, Yahoo, MarketWatch …). | Sentiment reflects that editorial mix, not "the market". | Top sources reported in the EDA. |
| B5 | **Domain mismatch of the sentiment model** (DistilBERT fine-tuned on SST-2 movie reviews). | Financial language ("beats estimates", "cuts guidance") may be misread; no neutral class. | Measured on Financial PhraseBank (`reports/sentiment_eval.json`: binary and 3-class accuracy / macro-F1 vs majority baseline); neutral band calibrated on a separate half; most confident errors logged to MLflow. |
| B10 | **Evaluation-set bias**: PhraseBank (AllAgree) holds short, unambiguous English sentences from news about Finnish listed companies (published 2014), not 2026 US headlines plus summaries. | Measured accuracy is an *optimistic* estimate for our articles. | Stated in the report; the band is calibrated on the same distribution it is tested on (separate halves); AllAgree chosen so labels are reliable. |
| B12 | **FinBERT was fine-tuned on PhraseBank.** | Its PhraseBank score is inflated (test-set leakage). | PhraseBank is never used to choose models; the gate uses Twitter financial news, which neither candidate saw. The `*_contaminated` flag is in every evaluation report. |
| B13 | **Gate dataset is tweets** (short, informal, cashtags), while our inputs are headline + summary. | Gate metrics are a proxy, not a measurement on our articles. | Same headline-style domain (finance news); stated in the report; both datasets reported. |
| B11 | **30-day estimation window** for the portfolio (proposal). | Mean returns over 30 days are extremely noisy; in-sample Sharpe ratios of 3–6 are common and do not persist. | Walk-forward out-of-sample comparison with equal weight and min volatility is reported next to every allocation. |
| B6 | **Mixed providers** after fallback (MarketStack vs Yahoo adjusted closes can differ slightly). | Small artificial jumps at the switch. | `source` lineage + provider-mix check; model price rule (§5). |
| B7 | **Unadjusted corporate actions** after the backfill (a split stored by incremental runs). | A −50 % / −75 % "return" corrupts training. | `prices.no_extreme_returns` is a *blocking* check; remedy: `ingestion backfill --prices-only` refreshes adjusted history. |
| B8 | **Non-stationarity**: volatility regimes change over the year. | A model trained once decays (the proposal's core problem). | Drift monitoring and retraining (Day 8); EDA §6 quantifies it. |
| B9 | Short history (12 months, free-plan limit). | Few independent LSTM windows; rare events under-represented. | One global model across tickers; naive baseline comparison. |

## 5. Processing rules

* **Model price**: per ticker, `adj_close` if every bar has it, otherwise `close` (never mixed within a series).
* **Calendar**: NYSE trading days from `finsight.common.timeutils` (weekends + full-day holidays).
* **De-duplication**: prices by (ticker, date) at ingestion; news by (ticker, news_id) and by normalised headline in the ETL.
* **Sentiment input**: `text` (headline + summary, ≤ 1,000 chars), whitespace-collapsed, truncated to 256 tokens.
* **Portfolio input**: simple daily returns of `price` over the last 30 aligned trading days (dates missing for any ticker dropped), annualised ×252.
* **Quality gate**: blocking expectations — schema, required nulls, unique keys, positive prices, universe complete, ≥ 114 rows per ticker (100-day lookback + 14-day horizon), ≤ 10 missing days, no |log return| > 0.5.

## 6. Observed characteristics (fill in from the EDA notebook, section 9)

Snapshot: `reports/export_manifest.json` sha256 prices = `________`

- Snapshot window / rows / articles: …
- Missing trading days per ticker: …
- Price provider mix: …
- Annualised volatility range: …
- Return kurtosis range: …
- Mean pairwise return correlation: …
- Largest daily move: …
- Volatility regime shift (2nd half / 1st half): …
- News articles per ticker-week (mean) and distinct sources: …
