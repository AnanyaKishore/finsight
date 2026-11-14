# FinSight: an AIOps-driven stock analysis platform

## Overview

FinSight is a self-hosted web application for financial analysis of ten large-cap US stocks
(AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, WMT, V). It gives three independent
analyses in one portal:

- **Price forecasting**: an LSTM predicts each stock's price for the next 14 days from the last 100 days.
- **News sentiment**: a transformer model (DistilBERT-SST-2 or FinBERT, whichever wins the promotion gate) scores company news from −1 to +1 and shows a weekly trend.
- **Portfolio optimisation**: Modern Portfolio Theory picks the long-only allocation with the best Sharpe ratio over the last 30 trading days.

The models are operated with AIOps practices: a DVC pipeline as CI, Apache Spark ETL,
MLflow tracking and model registry, Prometheus and Grafana monitoring, and drift-triggered
retraining. Everything runs on one laptop with Docker Compose. No cloud service is used.

> **Analytical insight only. FinSight does not give investment advice.**

![FinSight architecture](./docs/diagrams/architecture.png)
_System architecture: the 15 Docker Compose services and how they interact_

## Repository Structure

```
finsight/
├── src/finsight/            # all Python code (one package)
│   ├── ingestion/           # MarketStack / Finnhub clients, quota guard, MongoDB loader
│   ├── data/                # export (MongoDB -> Parquet) and data-quality checks
│   ├── etl/                 # Spark ETL and throughput benchmark
│   ├── models/lstm/         # LSTM training, evaluation, explainability
│   ├── sentiment/           # sentiment evaluation, calibration and scoring
│   ├── portfolio/           # MPT grid search and walk-forward backtest
│   ├── registry/            # promotion gate, registration, rollback
│   ├── scheduler/           # daily runs, retrains, Pipeline Console API
│   ├── monitoring/          # drift (PSI/KS), live forecast error, drift drill
│   ├── security/            # secret encryption and encrypted backups
│   ├── api/                 # FastAPI backend
│   ├── ci/                  # unit-test gate stage of the pipeline
│   └── common/              # config, logging, retries, MongoDB helpers
├── frontend/                # web portal: static HTML/CSS/JS served by nginx
├── monitoring/              # Prometheus rules, Alertmanager, Grafana dashboards
├── docker/                  # Dockerfiles for the five images built here
├── tests/                   # unit, integration and end-to-end tests
├── notebooks/               # exploratory data analysis
├── scripts/                 # preflight check, test report, load test, builders
├── docs/                    # design documents, test plan, user manual
├── configs/tickers.yaml     # the fixed 10-stock universe
├── params.yaml              # every tunable value (tracked by DVC)
├── dvc.yaml                 # the CI pipeline DAG
├── MLproject                # MLflow Project entry points
├── docker-compose.yml       # one-command deployment
├── Makefile                 # shortcuts (run `make help`)
└── README.md                # this file
```

`data/` and `models/` are versioned with DVC, not Git. `reports/` holds the small JSON
metrics that each pipeline run writes.

## Component Descriptions

### Data Engineering

A scheduler runs every trading morning (06:30 IST, Tuesday to Saturday). It loads
end-of-day prices (MarketStack, with yfinance as fallback) and company news (Finnhub)
into MongoDB, exports a snapshot, validates it with 21 data-quality checks, and processes
it with Apache Spark.

![DVC pipeline](./docs/diagrams/dvc_dag.png)
_The DVC pipeline: tests and data checks gate every stage that follows_

### Model Training and Registry

The LSTM, sentiment and portfolio stages are DVC stages, so a run can be reproduced from
a Git commit and an MLflow run ID. Every run is tracked in MLflow with its parameters,
metrics, data version and commit. A new model is promoted to `@production` in the MLflow
Model Registry only if it beats the current production model on the same test data.

### Model Serving and Backend

The two production models are served with `mlflow models serve`, behind a FastAPI backend
that the portal calls through a configurable URL. When a model is promoted, its server
restarts with the new version on its own. If a model server is down, the backend falls
back to the last stored forecast.

### Web Portal

Type-ahead stock search, a stock page (price history, 14-day forecast, weekly sentiment),
the optimised portfolio, and a **Pipeline Console** that shows the DVC DAG, every run with
its steps and logs, and the monitoring status.

### Monitoring and Automated Retraining

Prometheus collects API, pipeline, model and infrastructure metrics, and Grafana shows them
on five dashboards. Alert rules watch data drift and the live forecast error. When one
fires, Alertmanager asks the scheduler to retrain, and the promotion gate decides whether
the new model is used.

![Drift, retrain, promote](./docs/diagrams/retrain_loop.png)
_Monitoring closes the loop: drift or error alert, retrain, promote only if better_

## Documentation

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture diagram and every block explained |
| [HLD.md](HLD.md) | High-level design: goals, flows, design choices |
| [LLD.md](LLD.md) and [openapi.json](docs/api/openapi.json) | Endpoint input/output specifications, data stores, error codes |
| [DATA_CARD.md](docs/DATA_CARD.md) | Data sources, formats, lineage, known biases |
| [TEST_PLAN.md](TEST_PLAN.md), [TEST_CASES.md](TEST_CASES.md), [TEST_REPORT.md](TEST_REPORT.md) | Test plan, acceptance criteria, latest results |
| [USER_MANUAL.md](USER_MANUAL.md) | How to use the portal (non-technical) |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Retraining from a commit gives identical metrics |
| [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) | Every deviation from the proposal, with its reason |
| [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Plan, memory budget and risks |

## Getting Started

### Prerequisites

- Docker with Compose v2, with at least 6 GB of memory for Docker
- Miniforge (mamba), Git and Git LFS
- Free API keys for [MarketStack](https://marketstack.com/signup/free) and [Finnhub](https://finnhub.io/register)

### Installation and Setup

1. Clone the repository and create the Python environment:

   ```bash
   git clone https://github.com/AnanyaKishore/finsight.git
   cd finsight
   mamba env create -f environment.yml && mamba activate finsight
   pip install -e .
   ```

2. Set up the secrets, either from the encrypted copy (needs the key) or from the template:

   ```bash
   make secrets-decrypt          # secrets/.env.enc -> .env
   # or: cp .env.example .env and fill in the API keys and passwords
   ```

3. Fetch the versioned data and models, then start every service:

   ```bash
   dvc pull
   docker compose up -d          # about a minute
   make preflight                # checks that everything is up
   ```

### Where to find everything

| What | URL |
|---|---|
| Web portal | http://localhost:8080 |
| Pipeline Console | http://localhost:8080/#/pipeline |
| API and its interactive docs | http://localhost:8000/docs |
| MLflow (experiments and registry) | http://localhost:5000 |
| Grafana (dashboards) | http://localhost:3000 |
| Prometheus alerts / Alertmanager | http://localhost:9090/alerts, http://localhost:9093 |

Every port is bound to `127.0.0.1` only.

## Testing

```bash
make test           # unit tests, no services needed
make test-int       # integration tests against MongoDB
make e2e            # end-to-end smoke tests against the running stack
make test-report    # every suite -> docs/TEST_REPORT.md
```

`make help` lists every other shortcut.
