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
| `docs/ARCHITECTURE.md`, `HLD.md`, `LLD.md`, `TEST_PLAN.md`, `USER_MANUAL.md` | Added Days 5–10 |

## Repository layout (grows day by day)

```
finsight/
├── configs/tickers.yaml        # the fixed 10-stock universe
├── params.yaml                 # every tunable value (DVC-tracked params)
├── docker-compose.yml          # one-command deployment
├── docker/Dockerfile.pipeline  # ingestion / ETL / training / scheduler image
├── dvc.yaml                    # CI pipeline DAG (dvc repro)
├── src/finsight/
│   ├── common/                 # config, JSON logging, retry, HTTP, Mongo, calendar, run records
│   ├── ingestion/              # providers, quota ledger, repository, service, CLI
│   ├── ci/                     # unit-test gate stage
│   ├── data/                   # export (Mongo -> Parquet) and validate (expectation suite)
│   └── etl/                    # Spark ETL stage + throughput benchmark
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
