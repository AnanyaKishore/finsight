# FinSight — Architecture

FinSight is a self-hosted stock-analysis web application for ten large US stocks. It shows
**price history**, a **14-day LSTM forecast**, **weekly news sentiment** and an **optimised
portfolio**. It is operated with AIOps practices: a DVC pipeline as CI, MLflow tracking and a
model registry, Spark ETL, Prometheus/Grafana monitoring and drift-triggered retraining.
Everything runs on one laptop with Docker Compose; no cloud service is used.

Related documents: [HLD.md](HLD.md) (design choices and rationale) · [LLD.md](LLD.md)
(endpoints and I/O) · [USER_MANUAL.md](USER_MANUAL.md) · [TEST_PLAN.md](TEST_PLAN.md).

## 1. Architecture diagram

![FinSight architecture](docs/diagrams/architecture.png)

*Source: [`diagrams/architecture.mmd`](docs/diagrams/architecture.mmd) (Mermaid; also
[`architecture.svg`](docs/diagrams/architecture.svg)). The numbers on the arrows match the
interactions in §3.*

## 2. The blocks

Fifteen services in [`docker-compose.yml`](docker-compose.yml), in six groups. Five images
are built from this repository (`docker/Dockerfile.*`); the rest are pinned upstream images.

### Presentation

| Block | What it is | Responsibility |
|---|---|---|
| **frontend** `:8080` | nginx serving static HTML, CSS and JavaScript (`frontend/`) | The web portal: Overview (ten stocks, portfolio, "try the sentiment model"), the stock page (history, forecast, sentiment), the Pipeline Console and the help page. It has **no server-side logic** and never proxies the API: the browser calls the backend directly at the URL in `config.js`, which nginx generates from the `BACKEND_URL` environment variable. |

### Application

| Block | What it is | Responsibility |
|---|---|---|
| **backend** `:8000` | FastAPI (`src/finsight/api`) | The single REST API the portal uses (`/api/stocks`, `/search`, `/stock/{t}`, `/prediction/{t}`, `/sentiment/{t}`, `/sentiment/analyze`, `/portfolio`, `/health`, `/model-info`, `/pipeline/*`, `/monitoring`). It validates input, returns stable error codes, caches the day's forecast in MongoDB, degrades gracefully (a stale forecast with a label, then a 503) and exports Prometheus metrics. It is the only component the browser talks to for data. |

### Model inference

| Block | What it is | Responsibility |
|---|---|---|
| **lstm-serve** `:5001` | `mlflow models serve` of the `finsight-lstm@production` model (`docker/Dockerfile.serve`) | `POST /invocations`: last 100 closing prices → 14 forecast prices. |
| **sentiment-serve** `:5002` | Same image, `finsight-sentiment@production` | `POST /invocations`: headlines → label, confidence and a score in −1…+1 (DistilBERT-SST-2 or FinBERT, whichever the promotion gate chose). |

Both servers watch the registry alias (`docker/serve_watch.py`). When a promotion moves
`@production`, the server exits and Compose restarts it with the new version. No manual
deployment step is needed.

### Pipeline and MLOps

| Block | What it is | Responsibility |
|---|---|---|
| **scheduler** `:8001` | Python service (`src/finsight/scheduler`) in the pipeline image, with the repository mounted at `/app` | Runs the **daily run** (06:30 IST, Tue–Sat, catch-up after sleep): ingest, then `dvc repro`, then refresh the forecasts, monitor, `dvc push` and a local commit. It also runs **retrains** from the console or the Alertmanager webhook (`/retrain`, with a 24 h cooldown and a queue), and computes the **monitoring report** every 2 minutes (drift, live error, freshness, data quality). Records every run and step in MongoDB. |
| **DVC pipeline** | `dvc.yaml`, executed by the scheduler (or by you with `dvc repro`) | The CI DAG: `unit_tests` → `export` → `validate` → `process` (Spark) → `train_lstm` → `evaluate_lstm` → `register` (promotion gate); also `fetch/evaluate_sentiment@candidate`, `score_sentiment` and `optimize_portfolio`. See [`diagrams/dvc_dag.png`](docs/diagrams/dvc_dag.png). |
| **mlflow** `:5000` | MLflow tracking server + model registry (SQLite backend, artifacts on a volume) | Every training run with its parameters, metrics, lineage tags (Git commit, DVC data hash), plots and the packaged model; registered versions with `@production` / `@staging` aliases. |
| **ingestion** | One-off job (Compose profile `jobs`), same image as the scheduler | Manual backfill / incremental load from MarketStack (yfinance fallback) and Finnhub. |

### Data

| Block | What it is | Responsibility |
|---|---|---|
| **mongo** `:27017` | MongoDB 7 with authentication | Operational store: `prices`, `news`, `news_sentiment`, `sentiment_weekly`, `forecasts`, `portfolios`, `pipeline_runs`, `pipeline_executions`, `pipeline_dag`, `api_usage` (quota ledger), `monitoring_reports`. Unique indexes and upserts make every write idempotent. |
| **Git + DVC remote** | The repository and a local DVC remote directory | Code and small reports in Git (images via Git LFS); data, models and large outputs in DVC. `dvc.lock` pins the exact data and model of every commit. |

### Monitoring

| Block | What it is | Responsibility |
|---|---|---|
| **prometheus** `:9090` | Prometheus 3.5 | Scrapes the backend, scheduler, exporters, Alertmanager and Grafana every 15–30 s; evaluates the alert rules (`monitoring/prometheus/rules/`), which are unit-tested with `promtool`. |
| **alertmanager** `:9093` | Alertmanager 0.28 | Routes alerts. Alerts labelled `action=retrain` go to the scheduler's `/retrain` webhook with a bearer token. |
| **grafana** `:3000` | Grafana 12 | Five provisioned dashboards (Overview, API, Pipeline, Models & drift, Infrastructure), generated by `scripts/build_dashboards.py`. |
| **exporters** | node-exporter, cAdvisor, mongodb-exporter, nginx-exporter | Host, per-container (including MLflow and the model servers), database and web-server metrics. Not published to the host. |

## 3. How the blocks interact

| # | From → to | Protocol | When |
|---|---|---|---|
| 1 | Browser → frontend | HTTP, static files | Opening the portal |
| 2 | Browser → backend | REST/JSON; base URL from `config.js`; CORS allows only the portal's origin | Every page and button |
| 3 | backend → lstm-serve / sentiment-serve | `POST /invocations` (MLflow scoring protocol), retries with back-off | First forecast of the day per stock; "Analyse" button |
| 4 | backend ↔ mongo | MongoDB wire protocol, authenticated | Every data request; forecast cache |
| 5 | External APIs → scheduler | HTTPS (MarketStack, yfinance, Finnhub), quota ledger, retries | Daily run |
| 6 | scheduler → DVC pipeline | `dvc repro` subprocesses with per-step timeouts | Daily run, retrain |
| 7 | pipeline → mlflow | MLflow tracking/registry REST | Training, evaluation, registration |
| 8 | mlflow → model servers | Model download by alias | Start-up and after a promotion |
| 9 | prometheus → everything | HTTP scrape of `/metrics` | Every 15–30 s |
| 10 | prometheus → alertmanager | Alert notifications | When a rule fires |
| 11 | alertmanager → scheduler | `POST /retrain` webhook (token) | Drift or forecast-error alert |

Not drawn, for readability: the backend also asks the scheduler for status and starts runs
from the Pipeline Console (`/api/pipeline/*`, token); asks Alertmanager for active alerts
(`/api/monitoring`); and reads model versions from MLflow (`/api/model-info`). The scheduler
asks the backend to refresh the ten forecasts after each run. Grafana queries Prometheus.

The two most important flows:

**Daily run** (automation, CI and continuous delivery of data):

![Daily run](docs/diagrams/daily_run.png)

**Drift → retrain → promote** (monitoring closing the loop):

![Retrain loop](docs/diagrams/retrain_loop.png)

## 4. Loose coupling between the UI and the model inference

The rubric requires the frontend and the model inference engine to be independent blocks
connected only through configurable REST calls. In FinSight:

* The frontend is a separate container with only static files. It contains no model, no
  Python and no database access.
* It reaches the backend only through `fetch(BACKEND_URL + "/api/...")`. `BACKEND_URL` is
  an environment variable of the frontend container, written into `config.js` at start-up,
  so the backend can move without rebuilding the frontend.
* The backend reaches the models only through `LSTM_SERVE_URL` and `SENTIMENT_SERVE_URL`
  (REST, MLflow's scoring protocol). Swapping a model is a registry operation; neither the
  backend nor the frontend changes.
* The e2e test `test_portal_reaches_the_backend_only_through_its_configured_url` checks
  that the portal does not proxy `/api`, and unit tests check that `config.js` is generated.

## 5. Deployment view

| Concern | Choice |
|---|---|
| Packaging | 5 images built from this repo: `pipeline` (ingestion, Spark, training, scheduler), `mlflow`, `serve`, `backend`, `frontend`; all end with a non-root `USER` |
| Orchestration | Docker Compose; `restart: unless-stopped`; health checks gate start order |
| Network | One internal bridge network; every published port bound to `127.0.0.1` |
| Resources | Memory limit per service (largest: scheduler 3 GB during training, sentiment-serve 2 GB); about 3.8 GB idle and 6.6 GB at peak |
| State | Named volumes for MongoDB, MLflow, Prometheus and Grafana; the repository is mounted into the scheduler so DVC updates your working copy |
| Secrets | `.env` (git-ignored) plus `secrets/.env.enc` (Fernet-encrypted, committed); key outside the repository |
| Environment parity | One Python 3.11 dependency set (`requirements/*.txt`) used by `environment.yml`, the MLproject and every image |
