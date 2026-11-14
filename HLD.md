# FinSight — High-Level Design (HLD)

This document explains **what** FinSight does, **how it is put together**, and **why** each
major choice was made. Block-by-block detail is in [ARCHITECTURE.md](ARCHITECTURE.md), and the
interface contract in [LLD.md](LLD.md).

---

## 1. Problem and goals

**Problem.** A retail investor who looks at a stock usually needs three separate tools:
- a price chart;
- an opinion on where the price may go;
- a sense of whether the news is good or bad.

None of these tools says how reliable it is.

**What FinSight offers.** For ten large US stocks (Apple, Microsoft, Alphabet, Amazon,
Tesla, NVIDIA, Meta, JPMorgan Chase, Walmart, Visa), one page brings together:
- the price history;
- a 14-trading-day forecast;
- weekly news sentiment, with the headlines behind it;
- a suggested portfolio split.

Every estimate is shown next to an honest baseline. FinSight is an analysis tool, not
investment advice.

**Engineering goal.** Build it as an *operated* AI product, not a notebook. That means:
- it updates itself every day;
- every model is versioned, tracked and gated before it serves;
- the system watches its own data and models;
- it retrains itself when the world changes;
- all of this runs on one 8 GB laptop with no cloud.

### Success metrics (measured, see [TEST_REPORT.md](TEST_REPORT.md))

| Goal | Metric | Target |
|---|---|---|
| Forecast is useful | test MAPE vs "price stays the same" | ≤ 1.02 × naive (latest: 3.21 % vs 3.34 %) |
| Sentiment is meaningful | macro-F1 vs majority class, on data neither candidate was trained on | above baseline (latest: 0.657 vs 0.264, FinBERT) |
| Portal feels instant | p95 latency, cached endpoints, 10 users | < 300 ms (latest: 17 ms) |
| System is reliable | scheduled runs succeeding | ≥ 90 % |
| Data can be trusted | blocking data-quality failures | 0 |

## 2. Users and use cases

| User | Use cases |
|---|---|
| **Investor** (non-technical) | Search a stock; read its price history; see the forecast with its back-tested error; see the weekly news mood and the headlines behind it; look at the suggested portfolio and its out-of-sample reality check; paste a headline and score it. |
| **Operator** (the developer) | See every pipeline run, its steps, logs and failures (Pipeline Console); start a run or a retrain; watch drift, live error and infrastructure (console, Grafana); roll back a model; reproduce any result from a commit. |

## 3. System context and logical layers

![Logical layers](diagrams/hld_layers.png)

| Layer | Contents | Talks to |
|---|---|---|
| 1. Presentation | Static web portal (nginx) | Service layer only, via REST |
| 2. Service | FastAPI backend | Inference (REST), data (MongoDB), operations (scheduler, Alertmanager) |
| 3. Inference | MLflow model servers (LSTM, sentiment) | MLflow registry (load `@production`) |
| 4. Learning pipeline | DVC DAG (Spark ETL, training, evaluation, gate), MLflow tracking + registry | Data layer, inference (by promoting) |
| 5. Data | MongoDB (operational), Git + DVC remote (versioned files) | — |
| 6. Operations | Scheduler, Prometheus, Alertmanager, Grafana | Every layer |

Each layer can be replaced without touching the ones above it. A new model is a registry
promotion; a new UI only needs the REST contract; a new data source is a provider class
behind the ingestion service.

## 4. Main flows

1. **User request.**
   - The browser loads static files, then calls the backend.
   - The backend reads MongoDB. On the first forecast request of the day for a stock, it calls
     `lstm-serve` and stores the result, so later requests come from the cache.
   - If the model server is down, the last saved forecast is shown and labelled as such.
2. **Daily run** (06:30 IST, Tue–Sat, plus catch-up after the laptop sleeps):
   - ingest new prices and news;
   - unit-test gate, export, validation, Spark ETL;
   - score news sentiment and optimise the portfolio;
   - refresh the forecasts;
   - compute the monitoring report;
   - `dvc push` and a local Git commit of `dvc.lock` and the reports.
3. **Retrain loop.**
   - Drift or forecast-error gauges cross a threshold, and the Prometheus rule fires.
   - Alertmanager calls the scheduler's webhook, which runs a retrain.
   - The promotion gate compares challenger and champion on the same test windows.
   - If the challenger is better and accepted, `@production` moves, and the model server
     restarts with the new version.
   - Otherwise nothing changes.
4. **Reproduction.** `git checkout <commit>`, then `dvc pull`, then
   `mlflow run . -e train_lstm` gives the same metrics
   (`scripts/repro_drill.py`, [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

Sequence diagrams: [daily run](diagrams/daily_run.png), [retrain loop](diagrams/retrain_loop.png).
CI DAG: [dvc_dag.png](diagrams/dvc_dag.png).

## 5. Design choices and rationale

### 5.1 Technology

| Need | Choice | Why | Alternatives considered |
|---|---|---|---|
| Pipeline / CI | **DVC** (`dvc.yaml`, `dvc repro`) | The guidelines name DVC for CI. It re-runs only what changed, and versions data and models next to the code. Tests are the first stage, so a failing test stops data from moving. | GitHub Actions (cloud), Airflow (2–3 GB of RAM) |
| Data engineering | **Apache Spark**, local mode (`local[4]`, 1 GB driver) | One of Ray/Spark/K8s is required. Spark fits the memory budget, and its DataFrame API expresses the calendar alignment and joins cleanly. | Ray (competes for the same RAM without adding a capability we need), Kubernetes (cluster overhead) |
| Experiment tracking | **MLflow** server (SQLite + artifact store) | Required. It also records lineage beyond autolog: Git commit, DVC data hash, export SHA, seed, package versions. | — |
| Model registry and serving | **MLflow registry + `mlflow models serve`**, aliases `@production` / `@staging` | Promotion becomes a registry operation and serving follows the alias. Aliases are MLflow's recommended replacement for stages. | Ray Serve (more memory), hand-written Flask (no registry link) |
| API | **FastAPI** | Required. Typed models give the LLD's I/O spec and OpenAPI for free. Async-friendly and fast. | Flask (the baseline; no schema) |
| Frontend | **Static HTML/JS + nginx** | Nothing to compile and nothing server-side. It is truly decoupled from the backend, and runs in a 64 MB container. | React/Vite (build chain), Streamlit (couples UI and Python) |
| Operational store | **MongoDB 7** | Prices, news, forecasts and runs are documents with different shapes. Upserts with unique keys make ingestion idempotent. | PostgreSQL (fine too; MongoDB matches the baseline) |
| Scheduling | **Own scheduler service** (APScheduler-style loop + FastAPI) | Small (one container). It knows the run plan, the cooldown and the queue, and exposes `/metrics`. | Airflow (too heavy for 8 GB), cron (no state, no API) |
| Monitoring | **Prometheus + Alertmanager + Grafana** + exporters | Required. The alerts drive retraining, so monitoring is part of the control loop, not just charts. | — |
| Packaging | **Docker Compose**, 5 own images, one dependency set | Environment parity between dev, test and prod; one command to start. | Kubernetes (not needed on one machine) |
| Security | Localhost-only ports, Mongo auth, non-root images, tokens, **Fernet-encrypted** secrets and backups | The guidelines ask for encryption of data and models, and cloud key services are ruled out. | Vault / SOPS / GPG (a server or extra tooling for one laptop) |

### 5.2 Models

| Model | Design | Why |
|---|---|---|
| **Price forecast**: LSTM | One global model across the ten stocks. 100-day input, direct 14-step output. Per-window min-max scaling. Purged chronological split. 8-configuration grid, selected on validation MAPE. | Per-window scaling removes look-ahead leakage and lets one model learn from 10× the data. Direct output avoids compounding one-step errors. The purge gap stops test targets overlapping training windows. |
| **News sentiment**: DistilBERT-SST-2 (default) vs FinBERT (challenger) | Both packaged as MLflow pyfunc. Score = p(positive) − p(negative), with a calibrated neutral band. The promotion gate compares the two on Twitter financial news, which neither was trained on. | Movie-review sentiment may not transfer to finance, so it is measured before use. FinBERT was trained on PhraseBank, so PhraseBank cannot be the judge. |
| **Portfolio**: Modern Portfolio Theory | Exhaustive 10 % grid (92,378 long-only portfolios), max Sharpe on the last 30 trading days, a SciPy SLSQP check, a min-volatility fallback, and a weekly walk-forward backtest against equal weight. | Deterministic and fast. The walk-forward test answers the real question: does optimising help *afterwards*? The answer (often no) is shown to the user. |

**Explainability.**
- **Forecast:** occlusion attribution over the 100-day input window shows which past prices
  drove it.
- **Sentiment:** each weekly bar lists its most positive and most negative headlines.
- **Portfolio:** the walk-forward "reality check" sits under the pie.

### 5.3 Data

- **Sources:**
  - MarketStack for daily prices (free tier, 100 requests/month, protected by a quota
    ledger), with yfinance as the fallback;
  - Finnhub for company news (about one year).
  - Every row records its `source` (lineage).
- **Quality:** 21 expectations run before Spark, with severities `error` (stops the
  pipeline) and `warn`. A real-data example: zero prices from MarketStack were caught and
  repaired. See [DATA_CARD.md](DATA_CARD.md).
- **Versioning:**
  - MongoDB is exported to Parquet by the `export` stage, and the DVC hash of that output
    pins exactly which data trained which model;
  - `dvc.lock` is committed after each run.

### 5.4 Operations (the AIOps part)

| Concern | How |
|---|---|
| Automation | Every lifecycle stage runs without a person: ingest, validate, ETL, train, evaluate, gate, deploy (alias), monitor, retrain. |
| CI | DVC DAG with tests first; data and model stages run only on validated inputs. |
| Continuous delivery of models | Promotion gate: register every candidate; promote only when better than production on the same windows *and* within the acceptance band; serving follows the alias. Rollback CLI: `make rollback-lstm`. |
| Monitoring | Application, pipeline, model-quality and infrastructure metrics; 20+ alert rules with unit tests; five Grafana dashboards; the Pipeline Console in the portal. |
| Drift | PSI over 60 days of daily returns vs the year before, KS alongside; drift "detected" when ≥ 30 % of stocks exceed 0.25. Live forecast error vs naive on realised prices. Sentiment-shift alert (informational). |
| Retraining | Alert-driven through the Alertmanager webhook, with a 24 h cooldown. A request during a busy run is queued, not lost. A drift drill exercises the whole loop without touching stored data. |
| Reproducibility | Git commit + MLflow run ID for every model; `scripts/repro_drill.py` proves it from a fresh clone. |

## 6. Non-functional requirements

| Quality | Design response | Evidence |
|---|---|---|
| **Performance** | Forecasts computed once per stock per day and cached; reads are indexed MongoDB queries; static frontend | p95 17 ms at 10 users (AC-08) |
| **Scalability** | Spark ETL scales with data volume (benchmarked at 1×/10×/50×); stateless backend can run more replicas; model servers are separate and can scale independently | `reports/etl_benchmark.json` |
| **Reliability** | Retries with back-off and jitter; idempotent writes; per-step timeouts; partial runs; graceful degradation (stale forecast, then 503); catch-up after sleep; health checks and restart policies | e2e and unit tests; AC-11 |
| **Security** | See LLD §8b: localhost-only ports, authenticated MongoDB, non-root containers, tokens on control endpoints, CORS, CSP, encrypted secrets and backups | `tests/unit/test_security.py`, AC-01 |
| **Observability** | JSON logs with request/run IDs; Prometheus metrics from every component; alert rules; dashboards | Grafana, Prometheus targets (e2e) |
| **Maintainability** | Typed, documented Python (ruff + black, pre-commit); one config file (`params.yaml`); 285 automated tests | [TEST_REPORT.md](TEST_REPORT.md) |
| **Usability** | Plain-language portal with help page, keyboard support, light/dark and mobile layouts, a table view for every chart | [USER_MANUAL.md](USER_MANUAL.md) |
| **Resource budget** | Memory limits per container; about 3.8 GB idle and 6.6 GB peak on an 8 GB laptop | `docker stats`, Grafana Infrastructure dashboard |

## 7. Constraints and trade-offs

- **No cloud.** The DVC remote, MLflow, the registry and monitoring all run locally. The
  cost is that there is no off-site backup unless you copy the encrypted backups yourself.
- **Free data tiers.** MarketStack allows 100 requests a month, so daily ingestion uses one
  request per stock per day under a ledger, with yfinance as the fallback. The data is
  end-of-day, not live.
- **Small data.** About 250 trading days × 10 stocks. Spark's start-up overhead dominates at
  this size, which the benchmark reports honestly; the LSTM is kept small and pooled across
  stocks.
- **Honest models over impressive ones.** Daily prices are close to a random walk. The
  forecast is judged against "no change", and the portfolio against equal weight out of
  sample, and both comparisons are shown to the user even when unflattering.

## 8. Risks

| Risk | Mitigation |
|---|---|
| A data provider fails or changes its API | Retries, a fallback provider, partial runs, a freshness alert (`DataStale`) |
| A bad model reaches production | Promotion gate with an acceptance band; rollback CLI; serving follows the alias |
| Drift makes the model stale | Drift and live-error monitoring; alert-driven retraining with cooldown |
| Laptop asleep at run time | Catch-up run on wake |
| Lost secrets or data | `secrets/.env.enc` in Git; encrypted, verified backups including models |
