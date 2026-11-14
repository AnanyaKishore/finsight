# FinSight — Low-Level Design (LLD)

Covers the backend API (§2–§4), data stores (§5), observability (§6), degradation (§7), the portal (§8), security (§8b) and code structure (§9). Architecture: [ARCHITECTURE.md](ARCHITECTURE.md); design rationale: [HLD.md](HLD.md).
The machine-readable API contract is [`api/openapi.json`](api/openapi.json)
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

### 2.8 Pipeline Console (`/api/pipeline/*`, Day 7)
| Endpoint | Response |
|---|---|
| `GET /api/pipeline/status` | `{reachable, busy, running_run_id, current_step, next_daily_run, schedule: {time, days, timezone}, triggers_enabled, detail}`; when the scheduler is down, `reachable: false` and `busy` comes from Mongo |
| `GET /api/pipeline/runs?limit=1..200&status=&kind=daily\|retrain` | `{runs: [{run_id, kind, trigger, reason, status, started_at, finished_at, duration_s, current_step, git_commit, failed_steps, summary}]}`, newest first, no logs |
| `GET /api/pipeline/runs/{run_id}` | the summary plus `steps: [{name, label, status, critical, started_at, finished_at, duration_s, exit_code, detail, error, log_tail, metrics}]` and `stages: [{stage, status, started_at, duration_s, metrics, error, git_commit}]`; `404 unknown_run` |
| `GET /api/pipeline/dag` | `{nodes: [{name, status: success\|failed\|running\|never, stale, last_run_at, duration_s, error, metrics}], edges: [[from, to]], generated_at}`; `404 no_dag` before the scheduler first starts |
| `GET /api/pipeline/stats?days=1..365` | `{days, runs, by_status, success_rate, median_duration_s: {daily, retrain}, latest_throughput: {etl_rows_per_s, scoring_articles_per_s, ingest_rows_per_s}}` |
| `POST /api/pipeline/trigger` `{kind: daily\|retrain, reason}` | `202 {run_id, kind}`; `409 pipeline_busy`; `503 scheduler_unavailable` (down, or no token) |

Run status: `running`, `success`, `partial` (finished, a non-critical step failed or was partial), `failed` (a critical step failed), `interrupted` (the scheduler stopped mid-run; closed at its next start). Step status adds `pending` and `skipped`.

### 2.8b Monitoring (`GET /api/monitoring`, Day 8)
`{report, alerts: [{name, severity, action, summary, starts_at}], alertmanager_reachable, detail}`.
`report` is the newest `monitoring_reports` document: `generated_at`, `drill`,
`returns {per_ticker: {T: {psi, ks, drifted, current_volatility, reference_volatility, n_reference} | {status: insufficient_data}}, drifted_share, max_psi, detected, threshold}`,
`sentiment {psi, ks, mean_current, mean_reference, n_current, n_reference, detected}`,
`forecast {points, model_versions, mape, naive_mape, ratio_to_naive, per_ticker_mape, detected}`,
`freshness {expected_session, lag_trading_days, max_lag}`, `quality {status, errors, warnings, failed_checks}`,
`models {lstm, sentiment}`; a section that failed is `{status: error, error}`.
Alerts come from Alertmanager (`active, unsilenced, uninhibited`, sorted critical → info).
`404 no_monitoring` only when there is no report **and** Alertmanager is down.

### 2.9 Scheduler API (internal, `127.0.0.1:8001`)
| Endpoint | Behaviour |
|---|---|
| `GET /health` | `{status: ok, busy}` (compose healthcheck) |
| `GET /status` | busy, the running run document, next daily run, last finished run, schedule |
| `POST /run` `{kind, reason, trigger: console\|api}` | token required; `202 {run_id}`, `409 busy`, `401`, `503 trigger_disabled` |
| `POST /retrain` | Alertmanager webhook body; token as `Authorization: Bearer`; reason = alert names; `status: resolved` → `200 ignored`; within `monitoring.retrain_cooldown_h` (24 h) of the last retrain → `200 {ignored: cooldown, retry_after_s}`; while a daily run is busy → `202 {queued: true, after_run_id}` and the retrain starts as soon as that run ends (Alertmanager does not retry a 4xx, so a 409 would lose it); while a retrain is already running → `200 {ignored: a retrain is already running}`; otherwise `202 {run_id}` |
| `POST /monitor` | token required; recompute the monitoring report now → `{summary, report}`; `503 monitoring_disabled` |
| `GET /metrics` | `finsight_pipeline_runs_total{kind,status}`, `_last_run_status{kind}` (1 / 0.5 / 0), `_last_run_timestamp_seconds{kind}`, `_last_run_duration_seconds{kind}`, `_step_duration_seconds{step}`, `_step_failures_total{step}`, `finsight_pipeline_running`, `finsight_pipeline_throughput{measure}`, `finsight_retrains_triggered_total{trigger}`, `finsight_retrains_skipped_total`, and the monitoring gauges (§6.2) |

Run plans (`finsight.scheduler.plan`; critical steps stop the run):

| Step | daily | retrain | Command | Critical |
|---|---|---|---|---|
| ingest | ✓ | ✓ | `python -m finsight.ingestion.cli incremental` (exit 3 = partial) | |
| data | ✓ | | `dvc repro process` | ✓ |
| score_sentiment | ✓ | | `dvc repro --single-item score_sentiment` | |
| optimize_portfolio | ✓ | | `dvc repro --single-item optimize_portfolio` | |
| retrain | | ✓ | `dvc repro` | ✓ |
| forecasts | ✓ | ✓ | `GET backend /api/prediction/{t}` for the 10 tickers (waits for `lstm-serve` to reload after a promotion) | |
| monitor | ✓ | ✓ | in-process: build, store and export the monitoring report (`partial` if a section errored) | |
| version | ✓ | ✓ | `dvc push`; `git commit -- dvc.lock reports` (local only) | |

Timeouts per step: `params.yaml` → `scheduler.timeouts_s`.

## 3. Error codes

| HTTP | `code` | When |
|---|---|---|
| 404 | `unknown_ticker` | symbol not in the universe |
| 404 | `no_data` | no prices stored for the ticker |
| 404 | `no_portfolio` | the optimisation has not run yet |
| 404 | `unknown_run` | no pipeline run with that `run_id` |
| 404 | `no_dag` | the scheduler has not stored the DAG yet |
| 409 | `pipeline_busy` | a pipeline run is already in progress |
| 422 | `invalid_request` | parameter/body validation failed (message names the field) |
| 503 | `insufficient_history` | fewer than 100 prices for a forecast |
| 503 | `model_unavailable` | model server down/timing out (after retries) and no fallback exists |
| 503 | `database_unavailable` | MongoDB unreachable |
| 503 | `scheduler_unavailable` | scheduler down, refused, or `SCHEDULER_TOKEN` not configured |
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
| `pipeline_runs` | `run_id` | every DVC stage, ingestion, rollback (with `pipeline_run_id` when the scheduler started it) | Pipeline Console |
| `pipeline_executions` | `run_id` | scheduler (one document per run, steps updated live) | Pipeline Console |
| `pipeline_dag` | `_id: "dvc"` | scheduler (after each run and at start-up) | Pipeline Console |
| `api_usage` | `(provider, period)` | ingestion quota ledger | ingestion |
| `monitoring_reports` | TTL 30 days on `generated_at` | scheduler (every `monitoring.interval_s` and each run) | API `/api/monitoring` |
| `monitoring_drill` | `_id: "drill"` | `python -m finsight.monitoring.drill` | monitor (while `until` is in the future) |

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

`/metrics` also refreshes `finsight_dependency_up` when it is scraped (at most every 10 s), so a model server going down shows within one scrape even without traffic.

### 6.1 Prometheus stack (Day 8)
| Service | Port (localhost only) | Scrapes / does |
|---|---|---|
| Prometheus v3.5.0 | 9090 | every 15 s: backend, scheduler (30 s), node-exporter, cAdvisor (30 s, filtered to `container_*` used by the dashboards), mongodb-exporter, nginx-exporter, Alertmanager, Grafana (60 s), itself; rules every 30 s; 15 d / 1 GB retention |
| Alertmanager v0.28.1 | 9093 | `action="retrain"` → `POST scheduler:8001/retrain` (bearer token from a file written at start, group_wait 1 m, repeat 24 h, no resolved); others to the UI only; inhibition: backend down mutes API alerts, scheduler down mutes monitoring alerts |
| Grafana 12.1.1 | 3000 | datasource and 5 dashboards provisioned from `monitoring/grafana/` (generated by `scripts/build_dashboards.py`); refresh 30 s |
| node-exporter, cAdvisor, mongodb-exporter, nginx-exporter | internal | host, every container (incl. mlflow and the model servers), MongoDB, the frontend's nginx (`stub_status` on internal port 8081) |

### 6.2 Monitoring gauges (scheduler `/metrics`)
| Metric | Labels | Meaning |
|---|---|---|
| `finsight_drift_psi`, `finsight_drift_ks` | `feature` (`returns`/`sentiment`), `ticker` | latest PSI / KS |
| `finsight_drift_drifted_ticker_share` | | share of stocks with returns PSI > threshold |
| `finsight_drift_detected` | `feature` | 1 when the drift rule holds |
| `finsight_sentiment_mean_score` | `window` (`current`/`reference`) | mean article score |
| `finsight_forecast_live_mape_percent` | `model` (`lstm`/`naive`) | live MAPE on realised points |
| `finsight_forecast_live_mape_by_ticker_percent` | `ticker` | |
| `finsight_forecast_realised_points`, `finsight_forecast_error_ratio_to_naive`, `finsight_forecast_degraded` | | |
| `finsight_data_lag_trading_days` | `ticker` | trading days behind the latest session |
| `finsight_data_quality_failed_checks` | `severity` | from `reports/data_quality.json` |
| `finsight_model_production_version` | `model` | from `reports/promotion.json` |
| `finsight_drift_drill_active`, `finsight_monitoring_last_update_timestamp_seconds` | | |

### 6.3 Alert rules (`monitoring/prometheus/rules/finsight.yml`)
| Alert | Condition | For | Severity |
|---|---|---|---|
| TargetDown | `up == 0` | 2m | critical |
| ModelServerDown / DependencyDown | `finsight_dependency_up == 0` (model servers / mongodb, mlflow) | 5m | critical / warning |
| ApiHighErrorRate | 5xx ratio (5 m) > 1 % | 5m | critical |
| ApiHighLatency | p95 > 300 ms (excluding prediction and analyze) | 5m | warning |
| ApiModelEndpointSlow | p95 of prediction / analyze > 2 s | 10m | warning |
| ForecastsServedStale | a stale forecast was served in 30 m | – | warning |
| PipelineRunFailed / PipelineRunPartial | last run status 0 / 0.5 | 1m | critical / warning |
| PipelineNotRunning | no daily run for 80 h | 10m | warning |
| DataStale | max lag > 2 trading days | 30m | warning |
| DataQualityFailed | failed error-severity checks > 0 | 1m | critical |
| MonitoringStale | report older than 15 min | 5m | warning |
| **ReturnsDriftDetected** | `finsight_drift_detected{feature="returns"} == 1` | 5m | warning, `action=retrain` |
| **ForecastErrorDegraded** | `finsight_forecast_degraded == 1` | 5m | warning, `action=retrain` |
| SentimentShift | sentiment PSI > 0.25 | 15m | info |
| DriftDrillActive | drill running | – | info |
| HostMemoryLow / HostDiskLow | < 10 % available | 5m / 10m | warning |
| ContainerNearMemoryLimit | working set > 90 % of `mem_limit` | 5m | warning |
| ContainerOomKilled | OOM event in 10 m | – | critical |

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
| Routing | Hash routes: `#/` overview, `#/stock/{SYMBOL}`, `#/pipeline` (Day 7); unknown symbols show a "not covered" page. |
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
| Pipeline Console: scheduler card, buttons | `GET /api/pipeline/status` (every 5 s while a run is in progress, else 30 s), `POST /api/pipeline/trigger` |
| Pipeline Console: DAG, history, details, charts | `GET /api/pipeline/dag`, `/runs`, `/runs/{id}`, `/stats` |

## 8b. Security (Day 9)

| Property | How | Checked by |
|---|---|---|
| Secrets out of Git | `.env` (API keys, passwords, tokens) is git-ignored; a pre-commit hook refuses `.env`, `*.key`, `*.pem` | `test_secrets_and_keys_are_git_ignored_and_backups_too`, hook `no-plaintext-secrets` |
| Secrets encrypted at rest | `secrets/.env.enc` = Fernet (AES-128-CBC + HMAC-SHA256) of `.env`; safe to commit; `python -m finsight.security.secrets encrypt / decrypt / check` | `test_env_encrypt_decrypt_and_check` |
| Key handling | one key at `~/.config/finsight/secret.key` (or `$FINSIGHT_KEY_FILE`), created with mode 600, never overwritten, refused if inside the repository or readable by others | `test_key_is_private_never_overwritten_and_never_in_the_repo`, `secrets check` |
| Snapshots and model artifacts encrypted at rest | `python -m finsight.security.backup create [--include-models]`: MongoDB (Extended JSON), reports, params, `dvc.lock`, configs, optionally `models/`, streamed as tar.gz through the encrypting writer into `backups/*.tar.enc` (mode 600, git-ignored); `manifest.json` holds a SHA-256 per file; `verify` checks all of it; `restore` never writes the live database | `test_backup_round_trip_keeps_types_and_files`, `test_backup_without_models_and_tampering` |
| Tamper evidence | file = `FSENC1\n` + chunks (4-byte length, Fernet token); each chunk's plaintext starts with its index and a last-chunk flag, so a modified, dropped, reordered or truncated chunk is refused | `test_dropped_reordered_or_truncated_chunks_are_detected`, `test_wrong_key_and_modified_bytes_are_refused` |
| Database authentication | MongoDB starts with a root user from `.env`; every client authenticates | `test_no_plaintext_secrets_in_compose_and_mongo_requires_auth`, e2e `test_mongodb_refuses_connections_without_credentials` |
| Network exposure | every published port is bound to `127.0.0.1`; exporters and the nginx status page are not published at all | `test_every_published_port_is_bound_to_localhost` |
| Least privilege | every image we build ends with a non-root `USER`; the scheduler runs as the host user | `test_images_we_build_run_as_a_non_root_user` |
| Authenticated control endpoints | scheduler `/run`, `/retrain`, `/monitor` need `SCHEDULER_TOKEN`; Grafana needs a login | e2e `test_scheduler_and_mlflow_are_up` (401 without the token), `test_alertmanager_and_grafana_are_up` (Grafana API 401 without a login) |
| Browser hardening | CORS limited to the portal origin; the portal sends CSP, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` | `test_backend_cors_is_not_open_to_everyone`, e2e `test_cors_allows_the_portal_only`, `test_portal_serves_the_app_with_security_headers` |
| Dependency hygiene | versions pinned in `requirements/*.txt`; `make audit` (pip-audit) lists known CVEs | manual, before a release |

Encrypted file format (`finsight.security.crypto`):

```
MAGIC  b"FSENC1\n"
repeat: uint32 big-endian length N, then N bytes = Fernet token of
        (uint64 chunk index, uint8 last-flag, up to 4 MiB of plaintext)
```

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
| `finsight.api` | FastAPI app, schemas, repository, model client, metrics, pipeline endpoints, entry point |
| `finsight.scheduler` | `schedule` (when), `plan` (steps), `runner` (subprocess + timeout), `store` (Mongo), `dag`, `versioning` (dvc push, local commit), `service` (loop, one run at a time, metrics), `app` (HTTP API) |
| `docker/serve_watch.py` | serving containers: serve the `@production` version, exit when the alias moves |
| `finsight.security` | `crypto` (key file, chunked Fernet streams), `secrets` (`.env` ↔ `secrets/.env.enc`), `backup` (encrypted snapshots) |
| `finsight.ci` | unit-test gate stage |
| `frontend/` | portal pages (`index.html`, `help.html`), `assets/js` (`api`, `search`, `charts`, `format`, `app`, `pipeline`), `nginx/` template + start-up check |
