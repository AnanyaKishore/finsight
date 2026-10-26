# FinSight — Low-Level Design (LLD)

Draft (Day 5). Completed on Day 10 with the frontend, scheduler and monitoring sections.
The machine-readable API contract is [`docs/api/openapi.json`](api/openapi.json)
(regenerate with `python scripts/export_openapi.py`); interactive docs run at
`http://localhost:8000/docs`.

> All analytical responses carry a `disclaimer`: analytical insight only, not investment advice.

---

## 1. Conventions

| Aspect | Rule |
|---|---|
| Base URL | `http://localhost:8000` (configurable in the frontend; loose coupling) |
| Format | JSON, UTF-8; dates `YYYY-MM-DD`; timestamps ISO-8601 UTC |
| Tickers | Case-insensitive path parameter; must be one of the 10 symbols in `configs/tickers.yaml` |
| Errors | Always `{"error": {"code", "message", "request_id"}}` with a stable `code` (table §3) |
| Tracing | Every response has `X-Request-ID` (echoed if the client sent one); every request is one JSON log line |
| CORS | Only origins in `FRONTEND_ORIGINS` (default `http://localhost:8080` and `http://127.0.0.1:8080`); methods `GET`, `POST` |
| Auth | None (single-user, localhost-bound deployment); every port is bound to `127.0.0.1` |

## 2. Endpoints

### 2.0 `GET /api/stocks`
The whole universe in configuration order (portal home page). No parameters.

```json
{"stocks": [{"symbol": "AAPL", "name": "Apple", "sector": "Technology"}, "…"]}
```
Status: `200`.

### 2.1 `GET /api/search?q={text}&limit={n}`
Type-ahead over the fixed universe.

| Param | Type | Rules |
|---|---|---|
| `q` | string | 1–40 characters (required) |
| `limit` | int | 1–10, default 10 |

Ranking: symbol prefix → name substring → alias substring; ties by symbol.

```json
{"query": "goo", "results": [{"symbol": "GOOGL", "name": "Alphabet",
  "sector": "Communication Services", "matched": "symbol"}]}
```
Status: `200`, `422 invalid_request`.

### 2.2 `GET /api/stock/{ticker}?period={1W|1M|3M|6M|1Y|ALL}`
Overview card (green/red indicator) and price-history chart. Default `period=1M`
(calendar window ending at the latest stored bar).

| Field | Meaning |
|---|---|
| `overview.date`, `close` | latest stored session (raw close, as quoted by exchanges) |
| `overview.change`, `change_pct`, `direction` | vs the previous stored session; `up` / `down` / `flat` |
| `history[]` | `date, open, high, low, close, volume`, oldest first |
| `source` | provider of the latest bar (`marketstack` / `yfinance`) |

Status: `200`, `404 unknown_ticker`, `404 no_data`, `422 invalid_request`, `503 database_unavailable`.

### 2.3 `GET /api/prediction/{ticker}`
History plus a dashed 14-day forecast.

Processing:
1. Read the last 100 **model prices** from MongoDB, using the same rule as training: `adj_close` if every bar of the ticker has it, else `close`.
2. Look up the production LSTM version in the MLflow registry (cached 60 s).
3. If `forecasts` already holds this ticker / origin / version, return it (`cache`).
4. Otherwise `POST lstm-serve/invocations` with columns `p_000…p_099`. The response has columns `h_01…h_14`, which are mapped to the next 14 NYSE trading days; the result is stored and returned (`model`).
5. If the model server fails after retries, return the latest stored forecast with `stale: true` (`stale`). If none exists, return `503 model_unavailable`.

| Field | Meaning |
|---|---|
| `as_of` | forecast origin (last observed trading day) |
| `history[]` | last 60 model prices `{date, price}` |
| `forecast[]` | 14 `{date, price}` on NYSE trading days |
| `model` | `{name: finsight-lstm, version, run_id}` |
| `stale` | `true` only when served from an older stored forecast |

Status: `200`, `404 unknown_ticker`, `503 insufficient_history`, `503 model_unavailable`.

### 2.4 `GET /api/sentiment/{ticker}?weeks={1..52}`
Weekly sentiment bar chart (default 12 weeks), from `sentiment_weekly`.

| Field | Meaning |
|---|---|
| `weeks[].mean_score` | mean of article scores, −1…+1 (score = P(pos) − P(neg)) |
| `weeks[].article_count`, `positive`, `negative`, `neutral` | counts (neutral = within the calibrated band) |
| `weeks[].top_positive[]`, `top_negative[]` | ≤ 3 headlines each `{headline, score, source, url, published_at}`: the explanation for the bar |
| `overall_mean_score` | article-weighted mean over the returned weeks |
| `model_version` | fingerprint of the model that scored the newest article |

Status: `200` (an empty `weeks` list if nothing is scored yet), `404 unknown_ticker`, `422`.

### 2.5 `POST /api/sentiment/analyze`
Live scoring with the production sentiment model (`sentiment-serve`).

Request: `{"texts": ["...", ...]}` with 1–32 texts of ≤ 2,000 characters each.
Response: `{"results": [{text, label, confidence, score}], "model": {...}}`.
Status: `200`, `422 invalid_request`, `503 model_unavailable`.

### 2.6 `GET /api/portfolio`
Latest daily optimisation (`portfolios`, newest `as_of`).

| Field | Meaning |
|---|---|
| `allocation` | `{method, weights{ticker: w}, expected_return, volatility, sharpe}`; method is `max_sharpe`, or `min_volatility` if `fallback_used` |
| `equal_weight`, `min_volatility` | in-sample baselines on the same 30-day window |
| `out_of_sample` | walk-forward performance per strategy (`sharpe`, `annual_return`, `annual_volatility`, `max_drawdown`, …) |
| `beats_equal_weight_out_of_sample` | the honest verdict shown next to the pie chart |
| `window_start`, `window_end`, `risk_free_rate`, `grid_size` | provenance |

Status: `200`, `404 no_portfolio`.

### 2.7 Operations
| Endpoint | Response |
|---|---|
| `GET /api/health` | `{status: ok/degraded, dependencies: {mongodb, lstm-serve, sentiment-serve, mlflow-registry: {ok, detail}}, version}`; always `200` (liveness), with `degraded` when any dependency is down |
| `GET /api/model-info` | production `{name, version, run_id, tags}` per model (tags = evaluation metrics, acceptance, candidate), `registry_reachable` |
| `GET /metrics` | Prometheus text format (§6) |

## 3. Error codes

| HTTP | `code` | When |
|---|---|---|
| 404 | `unknown_ticker` | symbol not in the universe |
| 404 | `no_data` | no prices stored for the ticker |
| 404 | `no_portfolio` | the optimisation has not run yet |
| 422 | `invalid_request` | parameter/body validation failed (message names the field) |
| 503 | `insufficient_history` | fewer than 100 prices for a forecast |
| 503 | `model_unavailable` | model server down/timing out (after retries) and no fallback exists |
| 503 | `database_unavailable` | MongoDB unreachable |
| 500 | `internal_error` | unexpected; logged with stack trace and `request_id` |

## 4. Model-serving contracts (MLflow scoring server)

Both servers run `mlflow models serve -m models:/<name>@production --env-manager local`.

| | `lstm-serve` (:5001) | `sentiment-serve` (:5002) |
|---|---|---|
| Model | `finsight-lstm@production` | `finsight-sentiment@production` |
| Input | `dataframe_split`: columns `p_000…p_099` (USD, oldest first, finite, > 0) | `dataframe_split`: column `text` |
| Output | `h_01…h_14` (USD) | `label`, `confidence`, `score` |
| Validation | MLflow signature (missing columns rejected) + finite/positive check | signature + `text` column check |
| Pre-processing | per-window MinMax *inside* the model (same `scale_rows` as training) | tokenisation, truncation to 256 tokens, calibrated neutral band *inside* the model |

Backend client: timeout 10 s, 2 retries with exponential backoff and full jitter on
connection errors, 5xx and 429; 4xx is never retried.

## 5. Data stores

### 5.1 MongoDB collections

| Collection | Key (unique index) | Written by | Read by |
|---|---|---|---|
| `prices` | `(ticker, date)` | ingestion | export, API |
| `news` | `(ticker, news_id)` | ingestion | export |
| `news_sentiment` | `(ticker, news_id)` | `score_sentiment` | API (model version) |
| `sentiment_weekly` | `(ticker, week_start)` | `score_sentiment` | API |
| `portfolios` | `as_of` | `optimize_portfolio` | API |
| `forecasts` | `(ticker, as_of, model_version)` | API | API; monitoring (Day 8: realised error) |
| `pipeline_runs` | `run_id` | every DVC stage, ingestion, rollback | Pipeline Console (Day 7) |
| `api_usage` | `(provider, period)` | ingestion quota ledger | ingestion |

`forecasts` document: `{ticker, as_of, model_name, model_version, run_id, forecast: [{date, price}] (14), last_price, generated_at}`.

### 5.2 MLflow Model Registry

| Item | Value |
|---|---|
| Registered models | `finsight-lstm`, `finsight-sentiment` |
| Aliases | `@production` (served), `@staging` (latest challenger that did not win) |
| Stages (UI) | `Production` / `Staging` / `Archived`, kept in sync, best effort (deprecated in MLflow 3) |
| Version tags | LSTM: `test_mape`, `naive_mape`, `skill_vs_naive`, `accepted`, `selected_config`. Sentiment: `candidate`, `gate_macro_f1`, `gate_baseline_macro_f1`, `accepted`, `eval_fingerprint`, `model_version`, `neutral_band`, `trained_on`. Both: `rolled_back`, `rollback_reason` after a rollback |
| Model tag | `production_history`: JSON list of versions that held `@production` |

`reports/promotion.json` (Git-tracked DVC output): per family, `decision {promote, reason,
challenger_*, champion_*, metric}` and `production {registered_name, version, run_id,
model_uri[, candidate, model_version, neutral_band]}`.

### 5.3 Promotion rule (`finsight.registry.gate`)
Promote the challenger **iff** it passes acceptance **and** beats the champion on the
*same* evaluation data by more than `min_improvement`. If there is no champion yet, the
best candidate is promoted (bootstrap) and its acceptance status is recorded.

| Family | Metric (direction) | Acceptance | Same-data guarantee |
|---|---|---|---|
| LSTM | test MAPE (lower) | MAPE ≤ 1.02 × naive MAPE | the champion is re-scored on today's test windows |
| Sentiment | 3-class macro-F1 on the Twitter financial-news test half (higher) | macro-F1 > majority baseline | the champion is re-evaluated if its `eval_fingerprint` differs |

Rollback (`python -m finsight.registry.rollback --model {lstm,sentiment} [--to-version N] --reason "…"`):
the current version is tagged `rolled_back` (never auto-promoted again), `@production`
moves to the target, `promotion.json` is updated, a `pipeline_runs` record is written,
and the matching serving container must be restarted.

## 6. Observability (`GET /metrics`)

| Metric | Type | Labels |
|---|---|---|
| `finsight_http_requests_total` | counter | `method`, `route` (template), `status` |
| `finsight_http_request_duration_seconds` | histogram | `method`, `route` |
| `finsight_model_requests_total` | counter | `model`, `outcome` (`ok`/`error`/`rejected`) |
| `finsight_model_request_duration_seconds` | histogram | `model` |
| `finsight_forecasts_served_total` | counter | `source` (`model`/`cache`/`stale`) |
| `finsight_dependency_up` | gauge | `dependency` |

## 7. Graceful degradation matrix

| Failure | `/api/stock`, `/search` | `/api/prediction` | `/api/sentiment` | `/analyze` | `/api/portfolio` | `/api/health` |
|---|---|---|---|---|---|---|
| `lstm-serve` down | ✓ | stored today ✓, else stale ✓, else 503 | ✓ | ✓ | ✓ | degraded |
| `sentiment-serve` down | ✓ | ✓ | ✓ (precomputed) | 503 | ✓ | degraded |
| MLflow down | ✓ | ✓ (version shown as `unknown`/last) | ✓ | ✓ | ✓ | degraded |
| MongoDB down | 503 | 503 | 503 | ✓ | 503 | degraded |

Verified in the Day 5 rehearsal (lstm-serve stopped → stored forecast served, unseen ticker → 503, health `degraded`, `finsight_dependency_up{dependency="lstm-serve"} 0`).

## 8. Web portal (Day 6)

| Aspect | Design |
|---|---|
| Delivery | Static files in `frontend/`, served by `nginx-unprivileged` (uid 101, port 8080). No build step. |
| Coupling | The browser calls the backend directly at `BACKEND_URL`; nginx generates `/config.js` (`window.FINSIGHT_CONFIG`) from the environment at start-up. A malformed URL stops the container (`15-finsight-check-env.sh`). |
| Security | CSP `default-src 'self'`, `connect-src 'self' <BACKEND_URL>`, no third-party hosts (Chart.js 4.4.7 is vendored, MIT); `X-Frame-Options: DENY`; `nosniff`; `no-referrer`. All API text is HTML-escaped; only `http(s)` headline URLs become links. |
| Routing | Hash routes: `#/` overview, `#/stock/{SYMBOL}`; unknown symbols show a "not covered" page. |
| Client resilience | 15 s timeout per call (30 s for forecasts and analysis); GETs retried twice with exponential backoff (400 ms, 800 ms + jitter) on network errors, 502 and 504; requests are cancelled on navigation and stale answers are dropped. |
| States | Each card loads independently with a skeleton, an empty state and an error state with *Try again*; the header pill polls `/api/health` every 30 s (green / amber / red). |
| Charts | One y-axis per chart; price (line), forecast (solid history + dashed forecast), weekly sentiment (bars, −1…+1, by sign) with article counts as a separate chart, allocation (doughnut, ≤ 7 slices). Each chart has a hover tooltip and a data-table view. |
| Accessibility | Combobox search (↓ ↑ Enter Esc, `/` to focus), radio-group period controls with arrow keys, skip link, `aria-live` regions, colour never the only cue (arrows, signs, labels), light and dark mode, mobile layout (checked at 390 px). |

| Screen element | Endpoint |
|---|---|
| Stock tiles | `GET /api/stocks` |
| Search box | `GET /api/search` (150 ms debounce) |
| Overview + price chart | `GET /api/stock/{t}?period=` |
| Forecast chart | `GET /api/prediction/{t}` + `GET /api/model-info` (back-test tags) |
| Sentiment charts + headlines | `GET /api/sentiment/{t}?weeks=` |
| Portfolio card | `GET /api/portfolio` |
| "Try the sentiment model" | `POST /api/sentiment/analyze` |
| Status pill | `GET /api/health`, `GET /api/model-info` |

## 9. Code structure

| Package | Responsibility |
|---|---|
| `finsight.common` | settings, JSON logging, retry/backoff, HTTP, Mongo client + indexes, NYSE calendar, run records |
| `finsight.ingestion` | providers (MarketStack, yfinance, Finnhub), quota ledger, repository, service, CLI |
| `finsight.data` | `export` (Mongo → Parquet), `validate` (expectation suite) |
| `finsight.etl` | Spark session, ETL stage, throughput benchmark |
| `finsight.models` | lineage tags; `lstm/` (data, network, metrics, pyfunc, train, evaluate, explain, feature experiment) |
| `finsight.sentiment` | scorer, fetch, evaluate (per candidate), pyfunc, incremental scoring + publishing |
| `finsight.portfolio` | MPT grid, SciPy check, walk-forward, stage + publishing |
| `finsight.registry` | gate rules, registration stage, rollback CLI, production pointer |
| `finsight.api` | FastAPI app, schemas, repository, model client, metrics, entry point |
| `finsight.ci` | unit-test gate stage |
| `frontend/` | portal pages (`index.html`, `help.html`), `assets/js` (`api`, `search`, `charts`, `format`, `app`), `nginx/` template + start-up check |
