# FinSight — an AIOps-driven stock analysis platform

Price forecasting (LSTM), news sentiment (DistilBERT) and portfolio
optimisation (Modern Portfolio Theory) for 10 large-cap US stocks, operated
with AIOps practices: DVC pipelines as CI, MLflow tracking + registry,
Spark ETL, Prometheus/Grafana monitoring and drift-triggered retraining —
all self-hosted with Docker Compose.

> **Analytical insight only — not investment advice.**

| Document | Purpose |
|---|---|
| [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) | 11-day plan, rubric map, memory budget, risks |
| [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) | Every deviation from the proposal / baseline, with rationale |
| [`docs/DATA_CARD.md`](docs/DATA_CARD.md) | Data sources, formats, lineage, biases, EDA findings |
| [`docs/LLD.md`](docs/LLD.md) + [`docs/api/openapi.json`](docs/api/openapi.json) | Endpoint I/O specs, error codes, data stores, registry, degradation matrix (draft) |
| `docs/ARCHITECTURE.md`, `HLD.md`, `TEST_PLAN.md`, `USER_MANUAL.md` | Added Days 6–10 |

## Repository layout (grows day by day)

```
finsight/
├── configs/tickers.yaml        # the fixed 10-stock universe
├── params.yaml                 # every tunable value (DVC-tracked params)
├── docker-compose.yml          # one-command deployment
├── docker/Dockerfile.pipeline  # ingestion / ETL / training / scheduler image
├── dvc.yaml                    # CI pipeline DAG (dvc repro)
├── MLproject                   # MLflow Project entry points (conda_env: environment.yml)
├── src/finsight/
│   ├── common/                 # config, JSON logging, retry, HTTP, Mongo, calendar, run records
│   ├── ingestion/              # providers, quota ledger, repository, service, CLI
│   ├── ci/                     # unit-test gate stage
│   ├── data/                   # export (Mongo -> Parquet) and validate (expectation suite)
│   ├── etl/                    # Spark ETL stage + throughput benchmark
│   ├── models/                 # MLflow tracking/lineage; lstm/ (data, network, metrics, pyfunc, train, evaluate, explain, feature_experiment)
│   ├── sentiment/              # fetch (pinned model + PhraseBank), evaluate/calibrate/package, incremental scoring
│   ├── portfolio/              # MPT grid search, SciPy check, walk-forward backtest, publishing
│   ├── registry/               # promotion gate, register stage, rollback CLI, production pointer
│   └── api/                    # FastAPI backend: routes, schemas, Mongo repository, model client, metrics
├── tests/{unit,integration}/   # pytest suites
├── notebooks/01_eda.ipynb      # exploratory data analysis (built by scripts/build_eda_notebook.py)
├── scripts/                    # preflight check, notebook builder
├── docs/                       # plan, decisions, design docs
├── data/ models/               # DVC-managed (not in Git)
└── reports/                    # small JSON metrics committed to Git
```

---

## Day 1 runbook — setup and first data load

Time needed: ~45 minutes (the news backfill itself takes ~5 minutes because
Finnhub's free tier allows 60 calls/minute).

### 0. Prerequisites (once per machine)
- **Docker** with Compose v2 (`docker compose version`). On Docker Desktop
  (Windows/macOS) set *Settings → Resources → Memory* to **6 GB**. On WSL 2,
  create `%UserProfile%\.wslconfig` containing the two lines below, then run `wsl --shutdown`:
  ```ini
  [wsl2]
  memory=6GB
  ```
- **Miniforge** (mamba), **git**, **git-lfs** (`git lfs version`).
- Free API keys:
  - MarketStack: <https://marketstack.com/signup/free> (100 requests/month).
  - Finnhub: <https://finnhub.io/register> (60 calls/minute).

### 1. Repository
```bash
# create an EMPTY private repo "finsight" on GitHub first, then:
git clone git@github.com:<you>/finsight.git && cd finsight
# copy the Day-1 files into this folder, then:
git lfs install                 # once per machine; .gitattributes already lists LFS patterns
```

### 2. Python environment (Python 3.11, same versions as the Docker images)
```bash
mamba env create -f environment.yml
mamba activate finsight
pip install -e .
```

### 3. DVC with a local remote (no cloud — guidelines §III)
```bash
dvc init
mkdir -p "$HOME/finsight-dvc-remote"
dvc remote add -d localremote "$HOME/finsight-dvc-remote"
git add .dvc/config .dvcignore && git commit -m "chore: init DVC with local remote"
```

### 4. Secrets
```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(24))"   # paste as MONGO_ROOT_PASSWORD
# edit .env: MARKETSTACK_API_KEY, FINNHUB_API_KEY, MONGO_ROOT_PASSWORD
```

### 5. Start MongoDB and verify
```bash
docker compose up -d mongo
docker compose ps                       # mongo must show "healthy"
python scripts/check_env.py             # all checks should be OK
```

### 6. Quality gates
```bash
ruff check src tests scripts && black --check src tests scripts
pytest tests/unit -q --cov=finsight     # 50 tests
pytest tests/integration -m integration -q   # 4 tests against the real Mongo
```

### 7. First data load
```bash
python -m finsight.ingestion.cli backfill --dry-run   # expect 10 MarketStack calls
python -m finsight.ingestion.cli backfill             # ~5-6 min
python -m finsight.ingestion.cli status               # ~250 price rows per ticker
```

### 8. Prove the containerised path (spends no MarketStack quota today)
```bash
docker compose --profile jobs build ingestion
docker compose run --rm ingestion incremental
docker compose run --rm ingestion status
```

### 9. Commit
```bash
git add -A
git commit -m "feat(day1): ingestion service, quota guard, tests, compose"
git push -u origin main && git tag day1 && git push --tags
```

`make help` lists shortcuts for all of the above (Linux/macOS/WSL).

---

## Day 2 runbook — data engineering pipeline (DVC + Spark)

```
unit_tests ──► export ──► validate ──► process (Spark)
 CI gate      Mongo →     expectation    calendar alignment, features,
              Parquet     suite (gate)   news cleaning → data/processed/
```

```bash
docker compose up -d mongo                    # MongoDB must be running for `export`
pip install -r requirements/dev.txt           # adds pyspark, pyarrow, jupyterlab, matplotlib
python scripts/check_env.py                   # now also checks Java 17+ and pyspark
python -m finsight.ingestion.cli incremental  # top up to the latest session
dvc repro                                     # run the whole pipeline (~1-2 min)
dvc dag                                       # show the DAG
dvc metrics show                              # unit-test counts + ETL throughput
dvc repro                                     # again: validate/process are SKIPPED (unchanged data)
make eda                                      # EDA notebook -> reports/figures/*.png
make benchmark                                # Spark throughput at 1x/10x/50x
git add -A && git commit -m "feat(day2): ..." && dvc push && git push
```

| Stage | Command | Reads | Writes | Fails when |
|---|---|---|---|---|
| `unit_tests` | `python -m finsight.ci.run_unit_tests` | `src/`, `tests/` | `reports/unit_tests.json` | any unit test fails |
| `export` | `python -m finsight.data.export` | MongoDB | `data/raw/*.parquet`, `reports/export_manifest.json` | Mongo unreachable / empty |
| `validate` | `python -m finsight.data.validate` | raw snapshot | `reports/data_quality.json` | a blocking expectation fails |
| `process` | `python -m finsight.etl.process` | raw snapshot | `data/processed/**`, `reports/etl_metrics.json` | Spark error |

Each stage also writes a run record (`pipeline_runs`, `pipeline: "dvc"`) for the Pipeline Console.

---

## Day 3 runbook — LSTM forecasting + MLflow tracking

```
… process ──► train_lstm ──► evaluate_lstm
              grid search      packaged model on untouched
              (child runs),    test windows vs naive/drift
              refit, package
```

```bash
pip install -r requirements/dev.txt          # adds CPU-only torch + mlflow (pyarrow -> 25.0.1)
printf '\nMLFLOW_TRACKING_URI=http://localhost:5000\nMLFLOW_DISABLE_AGENT_HINT=1\n' >> .env   # once
docker compose up -d --build mlflow          # tracking server + registry at http://localhost:5000
python scripts/check_env.py                  # torch (CPU build), mlflow, server reachable
dvc repro                                    # unit tests ... train_lstm, evaluate_lstm (~3-5 min)
dvc metrics show                             # lstm_train.json (validation), lstm_metrics.json (test)
dvc plots show                               # dvc_plots/index.html: learning curve, MAPE by horizon
mlflow run . -e train_lstm --env-manager local --experiment-name finsight-lstm   # MLproject entry point
```

| Stage | Reads | Writes | MLflow |
|---|---|---|---|
| `train_lstm` | `data/processed/prices`, `params.yaml: lstm` | `models/lstm/` (pyfunc model + `run.json`), `reports/lstm_train.json`, learning curve CSV | parent run + 8 child runs (one per configuration), lineage tags, model with signature |
| `evaluate_lstm` | `models/lstm/`, processed prices | `reports/lstm_metrics.json`, MAPE-by-horizon CSV, forecast PNG | `test_*` metrics, `beats_naive` tag, full per-ticker JSON |

**Reproduce any result**: `git checkout <commit> && dvc checkout && dvc repro -f train_lstm evaluate_lstm`;
the MLflow run for that commit is in `reports/lstm_train.json` (`run_id`).

---

## Day 4 runbook — news sentiment + portfolio optimisation + explainability

```
unit_tests ─┬─► export ─► validate ─► process ─┬─► train_lstm ─► evaluate_lstm (+ input attribution)
            │                                  ├─► optimize_portfolio ──────────► MongoDB portfolios
            │                                  └─► score_sentiment ─────────────► MongoDB news_sentiment,
            └─► evaluate_sentiment ◄─ fetch_sentiment        ▲                     sentiment_weekly
                     └── models/sentiment/model ─────────────┘
```

```bash
pip install -r requirements/dev.txt          # adds transformers 5.20.2 + scipy 1.17.1
docker compose up -d mongo mlflow            # if not already running
python scripts/check_env.py                  # + transformers, scipy, Hugging Face Hub reachable
python -m finsight.ingestion.cli incremental # top up today's prices + news
dvc repro                                    # first run ~15-20 min (see table); later runs ~1-2 min
dvc metrics show reports/sentiment_eval.json reports/sentiment_scoring.json reports/portfolio.json
dvc plots show                               # dvc_plots/index.html: + attribution, band curve, backtest, frontier
make feature-experiment                      # LSTM price vs price+volatility (MLflow only, ~5-8 min)
```

| Stage | Reads | Writes | MLflow | First run |
|---|---|---|---|---|
| `fetch_sentiment` | Hugging Face Hub (pinned commits) | `models/sentiment/base` (~268 MB), `data/external/financial_phrasebank.csv` | — | 1-3 min download, then never again |
| `evaluate_sentiment` | base model, PhraseBank | `models/sentiment/model` (pyfunc), `run.json`, `reports/sentiment_eval.json`, band curve, confusion matrix | `finsight-sentiment`: metrics vs majority baseline, latency, errors, packaged model | ~2-3 min |
| `score_sentiment` | processed news, packaged model | `data/sentiment/*.parquet`, `reports/sentiment_scoring.json`, MongoDB | — | ~5 min (all articles); daily: seconds |
| `optimize_portfolio` | processed prices | `reports/portfolio*.json`, plots, MongoDB | `finsight-portfolio`: weights, in-/out-of-sample Sharpe, figures | ~10 s |
| `train_lstm` / `evaluate_lstm` | (unchanged inputs) | + `reports/plots/lstm_input_attribution.csv` | + attribution figure and metrics | retrains once today (its code changed); same metrics as Day 3 |

After a MongoDB reset, re-publish with `dvc repro -f score_sentiment optimize_portfolio`.

---

## Day 5 runbook — model registry, serving and the REST backend

```
… evaluate_lstm ─────────────────────────┐
  evaluate_sentiment@distilbert_sst2 ────┼─► register ──► reports/promotion.json ──► score_sentiment
  evaluate_sentiment@finbert ────────────┘   (MLflow registry: @production / @staging)

MLflow registry ──► lstm-serve :5001 ┐
                ──► sentiment-serve :5002 ┼─► backend :8000 ◄── frontend (Day 6)
MongoDB ─────────────────────────────────┘     /metrics ◄── Prometheus (Day 8)
```

```bash
pip install -r requirements/dev.txt                 # + prometheus-client (FastAPI/uvicorn already present)
rm -rf models/sentiment                             # Day 4 layout replaced by base/<candidate>, candidates/<candidate>
git rm -q reports/sentiment_eval.json reports/plots/sentiment_neutral_band.csv reports/plots/sentiment_confusion.png
docker compose up -d mongo mlflow
python scripts/check_env.py
python -m finsight.ingestion.cli incremental
dvc repro                                           # ~25-40 min first time (see below)
cat reports/promotion.json                          # gate decisions + production versions
dvc metrics show reports/sentiment_eval_distilbert_sst2.json reports/sentiment_eval_finbert.json
docker compose up -d --build lstm-serve sentiment-serve backend   # first build ~10 min
docker compose ps                                   # all healthy after ~1-2 min
curl -s localhost:8000/api/health | python -m json.tool
# then open http://localhost:8000/docs
```

| Stage / service | What it does | First run |
|---|---|---|
| `fetch_sentiment@finbert`, `fetch_sentiment_data` | FinBERT (~440 MB) + Twitter financial news + PhraseBank, pinned commits | 1-3 min |
| `evaluate_sentiment@<candidate>` | both datasets, band calibrated on the gate set, packaged pyfunc, MLflow run | ~2-4 min each |
| `register` | registers versions, champion/challenger gate, aliases + stages → `reports/promotion.json` | < 1 min |
| `score_sentiment` | re-scores all articles with the production model (new band / model) | 10-20 min once; then seconds |
| `lstm-serve`, `sentiment-serve` | `mlflow models serve` of `@production` (localhost:5001 / 5002) | ~1-2 min to load |
| `backend` | FastAPI on localhost:8000 (`/docs`, `/metrics`) | seconds |

**Rollback drill** (versions are listed in `reports/promotion.json` → `sentiment.candidates`):
```bash
python -m finsight.registry.rollback --model sentiment --to-version <other> --reason "drill" --keep-eligible
docker compose restart sentiment-serve              # now serves the other candidate
curl -s localhost:8000/api/model-info               # (registry cache: up to 60 s)
python -m finsight.registry.rollback --model sentiment --to-version <original> --reason "end of drill" --keep-eligible
docker compose restart sentiment-serve && dvc repro # pointer restored; nothing is re-scored
```

---

## Ingestion CLI reference

| Command | What it does |
|---|---|
| `backfill [--dry-run]` | Re-fetches the full window (`price_backfill_days`, `news_backfill_days`) for every ticker. |
| `incremental` | Gap-fills from each ticker's last stored date; skips tickers that are already current. This is what the scheduler runs daily. |
| `--tickers AAPL MSFT` | Restrict to a subset. |
| `--prices-only` / `--news-only` | Restrict to one data type. |
| `status` | Rows and date coverage per ticker, quota usage, last 5 runs. |
| `quota --set-used N` | Sync the ledger with the MarketStack dashboard. |

Exit codes: `0` success, `3` partial (some tickers failed, rest stored), `1` failed, `2` configuration error.

## Data sources

| Data | Provider | Free-tier limits | Fallback |
|---|---|---|---|
| End-of-day OHLCV | MarketStack | 100 requests/month, 12 months history, 1 request per ticker | yfinance (unmetered) |
| Company news | Finnhub | 60 calls/minute, ~1 year history | none (news is optional for the other analyses) |

Every stored price bar carries `source`, and every article carries `provider`, so lineage is visible in the data-quality report (Day 2).
