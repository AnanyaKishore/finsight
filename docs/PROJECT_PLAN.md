# FinSight — 11-Day Delivery Plan (21 → 31 October 2026)

**Goal:** a reproducible, self-hosted, one-command (`docker compose up`) AI
application that meets every item of the *AI Application Development
Guidelines* and the *AI Application Evaluation Guideline*, built on the
MLOps_AI_App and MLops_AiApp_rnn_model_tracking baselines.

**Hardware envelope:** 12 CPU cores, ~8 GB RAM, no GPU, no cloud.
Deviations from the proposal and their rationale: `docs/DESIGN_DECISIONS.md`.

---

## 1. Target architecture (what we are building towards)

```
                 ┌────────────── Docker Compose network "finsight" ───────────────┐
 MarketStack ──┐ │                                                                │
 (yfinance fb) ├─► scheduler ──► ingestion ──► MongoDB ◄── backend (FastAPI) ◄────┼── frontend (nginx)
 Finnhub ──────┘ │  (APScheduler)   │                       │   │      ▲         │     ▲ browser
                 │      │ dvc repro │                       │   │      │/metrics │     │
                 │      ▼           ▼                       │   │      │         │
                 │  DVC DAG: export→validate→[unit tests]→process(Spark)→drift   │
                 │        →train_lstm→evaluate→register   + sentiment, portfolio │
                 │      │                                   │   │                │
                 │      ▼                                   ▼   ▼                │
                 │  MLflow server (tracking + registry) ─► lstm-serve, sentiment-serve
                 │                                         (mlflow models serve) │
                 │  Prometheus ◄─ scrapes backend, scheduler, node, mongo exporters
                 │      │ alert rules ─► Alertmanager ─► webhook ─► scheduler /retrain
                 │      ▼                                                         │
                 │  Grafana dashboards (API, pipeline, model quality, drift, infra)
                 └────────────────────────────────────────────────────────────────┘
```

### Memory budget (all services carry `mem_limit` in compose)

| Service | Limit | Notes |
|---|---|---|
| mongo | 512 MB | WiredTiger cache capped at 0.25 GB |
| mlflow (tracking + registry) | 768 MB | SQLite backend + served artifacts; ~510 MB measured with background jobs disabled |
| lstm-serve | 1 GB | measured ~0.9 GB: `mlflow models serve` launcher (~250 MB) + worker with torch |
| sentiment-serve | 2 GB | ~1.4 GB with DistilBERT, ~1.6 GB with FinBERT (launcher + worker); stop it during heavy pipeline runs if needed |
| backend (FastAPI) | 384 MB | ~150-200 MB (no pandas/torch in the image) |
| frontend (nginx) | 64 MB | static files |
| scheduler (+ Spark/training jobs when running) | 2 GB peak | Spark `local[4]`, driver 1 GB |
| prometheus / alertmanager / grafana / exporters | ~600 MB total | |
| **Total** | **≈ 6.6 GB peak (pipeline run + all services), ≈ 3.8 GB idle** | Give Docker 6 GB; stop `sentiment-serve` while `dvc repro` scores news if the laptop swaps. |

---

## 2. Rubric coverage map

| Rubric item (points) | Delivered by | Day |
|---|---|---|
| UI/UX, responsiveness, user manual (3) | separate frontend, foolproof states, disclaimers, `docs/USER_MANUAL.md` | 6, 10 |
| ML pipeline visualisation + console (2) | Pipeline Console page (DAG, run history, failures, throughput, links) | 7 |
| Design docs, HLD/LLD, architecture, loose coupling (2) | `docs/ARCHITECTURE.md`, `HLD.md`, `LLD.md`; REST-only frontend | 5, 10 |
| Implementation: style, logging, exceptions, inline docs, unit tests (2) | ruff + black (PEP 8), JSON logs, exception hierarchy, Google docstrings, pytest | every day |
| Testing: plan, cases, report, acceptance criteria (1) | `docs/TEST_PLAN.md`, `docs/TEST_REPORT.md` | 9 |
| Data engineering with Spark, throughput (2) | Spark ETL stage + `reports/etl_metrics.json` + Grafana | 2, 8 |
| Source control & CI with DVC DAG; Git/LFS/DVC (2) | `dvc.yaml` DAG with gating test stage; LFS for binaries; DVC for data/models | 1–5 |
| Experiment tracking beyond autolog (2) | MLflow: params, metrics, artifacts, git commit, data hash, seed, backtests, plots | 3, 4 |
| Prometheus instrumentation + Grafana NRT (2) | custom metrics in backend/scheduler/pipeline; provisioned dashboards; alerts | 5, 7, 8 |
| Packaging: MLflow serving, MLproject, FastAPI, Docker, compose (2) | `mlflow models serve` containers, `MLproject` + `python_env.yaml`, FastAPI, separate images | 3–6 |
| Viva (5) | design-decision log, this plan, rehearsal | 11 |

---

## 3. Day-by-day plan

Each day ends with: tests green → `git commit` → `dvc push` (when data/models
changed) → `git push` → `git tag dayN`. Every day also appends to
`docs/DESIGN_DECISIONS.md`.

### Day 1 — Wed 21 Oct · Foundations + data ingestion ✅ (files delivered today)
**Role in the pipeline:** the entry point. Everything downstream (ETL,
training, UI) reads what ingestion writes to MongoDB.
- Repo scaffold, Python 3.11 env, pinned requirements, PEP 8 tooling.
- Git + Git LFS (binary docs) + DVC with a **local** remote.
- `docker-compose.yml` v1: MongoDB (auth, localhost-only, memory-capped) + ingestion job image (multi-stage Dockerfile, non-root).
- Common library: config (env + `params.yaml`), JSON logging with `run_id`, retry with exponential backoff + jitter, HTTP client with error classification, Mongo indexes.
- Ingestion: MarketStack (primary) → yfinance (fallback) prices; Finnhub news; quota ledger; idempotent upserts; run records in `pipeline_runs`; CLI (`backfill`, `incremental`, `status`, `quota`, `--dry-run`).
- 50 unit tests (91 % coverage) + 4 integration tests against real Mongo.
- **Run the 12-month price backfill and 6-month news backfill.**

### Day 2 — Thu 22 Oct · Data engineering: EDA, validation, Spark ETL, DVC DAG v1
**Role:** turns raw Mongo documents into versioned, validated, model-ready datasets.
- `notebooks/01_eda.ipynb`: coverage, missing days, return distributions, outliers, volume, news counts per week, provider mix → findings into `docs/DATA_CARD.md` (sources, formats, biases).
- `export` stage: Mongo → `data/raw/{prices,news}.parquet` + `manifest.json` (row counts, date range, SHA-256). `always_changed: true` so new data flows in; DVC skips downstream if the hash is unchanged.
- `validate` stage: schema, nulls, duplicates, OHLC consistency (low ≤ open/close ≤ high), non-positive prices, robust-z outliers on log returns, staleness, minimum history per ticker (≥ 114 rows) → `reports/data_quality.json`; **non-zero exit blocks the DAG**.
- `unit_tests` gating stage (pytest → `reports/unit_tests.xml`); downstream stages depend on its output.
- `process` stage (PySpark `local[4]`): calendar alignment, gap handling, log returns, rolling vol/means, weekly news aggregates → partitioned Parquet in `data/processed/`; rows/s and seconds per step → `reports/etl_metrics.json`.
- Add OpenJDK 17 + PySpark to the pipeline image. `dvc repro`, `dvc dag`, `dvc push`.

### Day 3 — Fri 23 Oct · Price-forecasting model + MLflow tracking + MLproject
**Role:** produces the 14-day forecast model and its full audit trail.
- MLflow tracking server container (SQLite backend, local artifact store).
- Windowing: 100-day input, 14-day direct output, per-window MinMax, global model across tickers; chronological split (last 20 trading days of targets held out).
- Baselines: naive last-value and drift; metrics RMSE, MAPE, directional accuracy, per ticker + pooled.
- `train_lstm` (small grid ≤ 8 configs, `torch.set_num_threads(6)`, fixed seeds) logging params, per-epoch losses, **git commit**, **DVC data hash** (from `dvc.lock`), seed, library versions, learning curves, backtest plots, per-ticker metrics table.
- Custom `mlflow.pyfunc` wrapper (raw closes in → prices out; scaling inside), signature + input example.
- `MLproject` (entry points `train_lstm`, `evaluate_lstm`, later `sentiment_eval`, `optimize_portfolio`) + `python_env.yaml` mirroring `requirements/*.txt`.
- DVC stages `train_lstm` → `evaluate_lstm`; metrics + plots registered with DVC (`dvc metrics show`, `dvc plots show`).

### Day 4 — Sat 24 Oct · Sentiment model + portfolio optimisation ✅ (done 23 Oct)
**Role:** the other two analyses.
**Delivered:** DVC stages `fetch_sentiment` → `evaluate_sentiment` → `score_sentiment` and
`optimize_portfolio`; LSTM occlusion attribution (in `evaluate_lstm`); feature-impact
experiment (`make feature-experiment`, MLflow only); MongoDB collections `news_sentiment`,
`sentiment_weekly`, `portfolios`; 132 unit tests. Details in DESIGN_DECISIONS.md (D-09, D-11, D-12, Day 4).
- Sentiment: DistilBERT-SST-2 wrapped as pyfunc (text → label, confidence, score ∈ [−1, 1]); evaluation on a labelled financial-news sample (e.g. Financial PhraseBank, neutral removed) → accuracy, F1, confusion matrix, latency; logged to MLflow.
- Batch scoring of new articles (only unscored docs), weekly aggregates with article counts.
- Portfolio: vectorised long-only grid (step 0.1 → 92 378 weightings), Sharpe/return/volatility with risk-free rate param; walk-forward comparison with equal weight; result stored in Mongo (`portfolio`) + MLflow.
- DVC stages `evaluate_sentiment`, `optimize_portfolio`.

### Day 5 — Sun 25 Oct · Registry, model serving, FastAPI backend ✅ (done 24 Oct)
**Delivered:** `register` DVC stage (champion/challenger gate, aliases + stages, idempotent),
rollback CLI, FinBERT challenger on a fair gate dataset, `lstm-serve` / `sentiment-serve`
(`mlflow models serve` from `@production`), FastAPI backend (8 endpoints, Prometheus
metrics, stored forecasts, graceful degradation), `docs/LLD.md` + `docs/api/openapi.json`,
155 unit tests.
**Role:** turns models into APIs.
- `register` stage: champion/challenger gate (promote only if the new model beats Production on the same validation set) → stages + `@staging/@production` aliases; `scripts/rollback.py`.
- `lstm-serve` and `sentiment-serve` containers via `mlflow models serve -m models:/<name>@production`.
- FastAPI backend: `/api/search`, `/api/stock/{ticker}?period=`, `/api/portfolio`, `/api/sentiment/{ticker}`, `/api/prediction/{ticker}`, `/api/pipeline/*`, `/health`, `/metrics`; Pydantic I/O models (→ LLD); timeouts, retries, caching, graceful degradation; CORS for the configurable frontend origin.
- Prometheus client metrics in the backend (request count/latency histogram/errors, model-call latency).
- Draft `docs/LLD.md` from the OpenAPI schema.

### Day 6 — Mon 26 Oct · Frontend
**Role:** the user-facing portal (3 rubric points).
- Separate nginx container; `config.js` generated at start-up from `BACKEND_URL`.
- Type-ahead search, overview card (price, change, green/red), price chart with 1W/1M/3M/6M/1Y, portfolio pie + Sharpe/return/volatility, weekly sentiment bars + article counts, prediction chart (history + dashed 14-day forecast), model "as of" timestamps.
- Foolproof: loading skeletons, empty/error states, disabled invalid actions, keyboard navigation, mobile layout, disclaimer banner, Help → user manual.

### Day 7 — Tue 27 Oct · Scheduler + Pipeline Console
**Role:** automation and the pipeline-visualisation rubric item.
- Scheduler service: weekday job (ingest → `dvc repro` daily targets → score news → refresh forecasts → portfolio), `/retrain` webhook, per-stage run records, auto-commit of `dvc.lock` + `dvc push`.
- Pipeline Console page: DVC DAG diagram, run history (success/failed/partial), error drill-down by `run_id`, throughput and duration charts, deep links to MLflow, Grafana, Prometheus.

### Day 8 — Wed 28 Oct · Monitoring, drift and automated retraining
**Role:** closes the AIOps loop.
- Prometheus scrape config; exporters (node, MongoDB); custom metrics: pipeline run status/duration/throughput, data-quality failures, drift scores (PSI/KS on returns, sentiment-score shift), rolling forecast error vs realised prices.
- Grafana dashboards provisioned as code; alert rules (p95 latency, error rate, pipeline failure, drift > threshold, rolling MAPE > threshold).
- Alertmanager → webhook → scheduler `/retrain` → `dvc repro register` → gate → promote or keep. Demo by injecting drift.

### Day 9 — Thu 29 Oct · Testing, security, code quality
- Test plan with acceptance criteria (model metrics vs baselines, p95 < 300 ms for cached endpoints, error rate < 1 %, pipeline success), unit + integration + e2e (compose up → API → UI smoke), load test.
- Security: Fernet-encrypted `secrets/.env.enc` (key outside Git), Mongo auth, localhost-only ports, non-root containers, encrypted exported snapshots / model artifacts at rest.
- Pre-commit hooks; final coverage report → `docs/TEST_REPORT.md` (passed/failed counts).

### Day 10 — Fri 30 Oct · Documentation + reproducibility drill
- `ARCHITECTURE.md` (diagram + block explanations), `HLD.md`, `LLD.md`, `USER_MANUAL.md` (screenshots, non-technical), README.
- Reproducibility drill (as in q4demo): fresh clone → `git checkout <commit>` → `dvc pull` → `mlflow run` → identical metrics; record commit hash + run ID.

### Day 11 — Sat 31 Oct · Buffer, dress rehearsal, viva prep
- Full `docker compose up` on a clean machine state; fix anything found.
- Demo script; viva Q&A sheet (Assignments 1–10, tool choices, defending D-01…D-10).
- `git tag v1.0`.

---

## 4. Risk register

| Risk | Mitigation |
|---|---|
| MarketStack quota (100/month) runs out | Ledger + reserve + yfinance fallback; daily runs skip up-to-date tickers. Expected use: 10 (backfill) + ≤ 7 weekday runs × 10 = 80. |
| 8 GB RAM exhausted | `mem_limit` per service; Spark `local[4]`; jobs run sequentially; Grafana/Prometheus retention short. |
| LSTM does not beat the naive baseline | Expected for daily prices; acceptance criteria are set relative to naive, the champion/challenger gate prevents regressions, and the honest comparison is a viva strength. |
| DistilBERT (SST-2) is not finance-tuned | Measured on a labelled financial sample and reported; it is the proposal's model. |
| Docker Desktop memory default too low | README step: allocate 6 GB. |
| Time overrun | Day 11 buffer; Day 9 security items are small; docs drafted from code (OpenAPI → LLD). |
