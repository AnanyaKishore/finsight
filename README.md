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
│   ├── scheduler/              # daily/retrain runs: schedule, step plan, runner, DAG, git/DVC, HTTP API (Day 7)
│   ├── monitoring/             # drift (PSI/KS), live forecast error, freshness, report gauges, drift drill (Day 8)
│   └── api/                    # FastAPI backend: routes, schemas, Mongo repository, model client, metrics, pipeline endpoints
├── frontend/                   # web portal: static HTML/CSS/JS + nginx template (Day 6)
├── monitoring/                 # Prometheus config + alert rules (+ tests), Alertmanager, Grafana provisioning + dashboards (Day 8)
├── tests/{unit,integration}/   # pytest suites
├── notebooks/01_eda.ipynb      # exploratory data analysis (built by scripts/build_eda_notebook.py)
├── scripts/                    # preflight check, notebook and dashboard builders
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
                ──► sentiment-serve :5002 ┼─► backend :8000 ◄── frontend :8080
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

## Day 6 runbook — the web portal

```
browser ──► frontend :8080 (nginx, static files + generated /config.js)
   │
   └──── fetch(BACKEND_URL + /api/...) ──► backend :8000 ──► Mongo, lstm-serve, sentiment-serve, MLflow
```

The portal is plain HTML, CSS and JavaScript (no build step, no Node.js). nginx
serves it and generates `/config.js` from `BACKEND_URL` at start-up, so the same
image works against any backend. The browser calls the API directly; the
backend's CORS list (`FRONTEND_ORIGINS`) must contain the portal's address.

```bash
pip install -r requirements/dev.txt                  # nothing new; refreshes the editable install
docker image prune -f                                # removes the untagged ~2 GB serving image from Day 5
pytest tests/unit -q                                 # expect all to pass (~150)
dvc repro                                            # score_sentiment re-runs once (its code changed): scored_this_run 0
docker compose up -d --build lstm-serve sentiment-serve backend frontend
docker compose ps                                    # frontend healthy within seconds
# open http://localhost:8080
```

| What to try | Where |
|---|---|
| Type-ahead search: "app", "iphone", "nv"; ↓/↑/Enter/Esc; `/` focuses the box | header |
| Overview card: price, daily change, green/red arrow | stock page |
| Price history with 1W/1M/3M/6M/1Y (arrow keys switch period) | stock page |
| 14-day forecast (dashed) after 60 days of history, back-test caption | stock page |
| Weekly sentiment bars + article counts; click a bar for its headlines | stock page |
| Portfolio pie, Sharpe / return / volatility, walk-forward reality check | overview |
| Live sentiment model ("Try the sentiment model") | overview |
| System status pill: green / amber (degraded) / red (API unreachable) | header |

**Graceful degradation drill** (screenshots for the report):
```bash
docker compose stop lstm-serve       # status pill turns amber; forecasts show the saved copy (or a clear error)
docker compose start lstm-serve
docker compose stop backend          # every card shows "Cannot reach the FinSight API" + Try again
docker compose start backend
```

**Long sentiment re-scores now report progress** (`dvc repro` output), e.g.
`scored 1500/11015 (13.6%), 18.2 articles/s, ~9 min left`, and save a checkpoint
every 500 articles. If a run is interrupted, the next `dvc repro` resumes from
`data/sentiment/article_scores.partial.parquet` instead of starting over.

**Serving another backend address:** set `BACKEND_URL` (what the browser uses)
and add the portal's address to `FRONTEND_ORIGINS` in `.env`, then
`docker compose up -d frontend backend`. A malformed `BACKEND_URL` stops the
frontend container at start-up with a clear message.

**Developing the portal without Docker:** `make web-dev` serves `frontend/` at
http://localhost:8080 with Python's built-in server (uses `frontend/config.js`).

## Day 7 runbook — the scheduler and the Pipeline Console

```
scheduler :8001 (pipeline image, repo mounted at /app)
   ├─ every Tue–Sat 06:30 IST (or as soon as the laptop wakes):
   │     ingest → dvc repro process → dvc repro --single-item score_sentiment
   │     → dvc repro --single-item optimize_portfolio → forecasts (via backend) → dvc push + local commit
   ├─ POST /run      ← backend /api/pipeline/trigger  ← "Run now" / "Retrain now" in the console
   └─ POST /retrain  ← Alertmanager webhook (Day 8)
         every step → Mongo pipeline_executions; every DVC stage → pipeline_runs (same run_id)
browser ──► #/pipeline ──► backend /api/pipeline/{status,runs,runs/{id},dag,stats,trigger}
lstm-serve / sentiment-serve: restart themselves when @production moves (serve_watch.py)
```

The daily run never retrains: retraining (`dvc repro`, i.e. train → evaluate →
promotion gate) only happens from **Retrain now** or the `/retrain` webhook.
The scheduler commits `dvc.lock` and `reports/` **locally** (author "FinSight
scheduler"); it never pushes to GitHub. You push when you want to.

**1. One-time settings in `.env`** (see `.env.example`):
```bash
echo "SCHEDULER_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
echo "DVC_REMOTE_DIR=$(dvc remote list | awk '{print $2}')" >> .env    # must be an absolute path
id -u; id -g     # if either is not 1000, add FINSIGHT_UID=... and FINSIGHT_GID=... to .env
```

**2. Build and start** (the pipeline image gains DVC, git and the scheduler; the
serving image gains the alias watcher):
```bash
pip install -r requirements/dev.txt
pytest tests/unit -q                                   # expect all to pass (~195)
docker compose build ingestion lstm-serve              # pipeline + serve images (~5-10 min)
docker compose up -d --build backend frontend   # their images contain the new code
docker compose up -d lstm-serve sentiment-serve scheduler
docker compose ps                                      # scheduler "healthy" after ~30 s
docker compose logs -f scheduler                       # Ctrl+C to stop following
```
Today is a run day and it is past 06:30, so the scheduler starts a
**catch-up** daily run within a minute of starting. Open
http://localhost:8080/#/pipeline and watch it (the page refreshes every 5 s
while a run is in progress). Expect 5–25 minutes, mostly Spark and scoring.

**Do not run `dvc repro` on the host while the scheduler is running a step**:
both would work on the same repository and DVC would refuse with a lock error.
Use **Run now** in the console instead, or `docker compose stop scheduler` first.

| What to try | Where |
|---|---|
| Live progress: current step, DAG boxes turning green | Scheduler card, DAG |
| Click a DAG stage: last run, duration, metrics, upstream/downstream | DAG |
| Run history, filter Failed/Partial/Succeeded, click a run for steps + log tail | Run history, Run details |
| Duration (daily / retrain) and throughput (ETL rows/s, articles/s) charts | bottom |
| **Run now** (a second daily run; the schedule still runs tomorrow) | Scheduler card |
| **Retrain now** (20–60 min; the gate decides whether v-next is promoted) | Scheduler card |
| Scheduler metrics for Day 8: `curl -s localhost:8001/metrics \| grep finsight_pipeline` | terminal |

**Failure drill** (screenshot for the report): break a stage, run, read the error
in the console, fix, run again.
```bash
docker compose exec mongo mongosh -u $MONGO_ROOT_USER -p $MONGO_ROOT_PASSWORD --authenticationDatabase admin finsight \
  --eval 'db.prices.findOneAndUpdate({ticker:"AAPL"}, {$set:{close:-1}}, {sort:{date:-1}})'
# console → Run now → the run fails at "Tests, export, validation, Spark ETL"; the
# log tail shows prices.positive_values failed; later steps are "Skipped".
docker compose run --rm ingestion repair      # re-fetches the broken bar (the fix the log suggests)
# console → Run now → succeeds
```

**Triggering without the console** (what Day 8's alerts do):
```bash
curl -s -X POST localhost:8001/retrain -H "Authorization: Bearer $SCHEDULER_TOKEN" \
  -H 'Content-Type: application/json' -d '{"status":"firing","alerts":[{"labels":{"alertname":"Manual"}}]}'
```

**Pushing the scheduler's commits:** `git log --oneline -5` shows
`chore(pipeline): scheduler run <day> [<run_id>]` commits. Push them with your
own work (`git push`). If the scheduler finds a merge or rebase in progress, or a
detached HEAD, it skips the commit and says so in the run's last step.

**Full logs:** `logs/scheduler/<run_id>.log` (every command's complete output);
the console keeps the last 40 lines of a failed step.

---

## Day 8 runbook — monitoring, drift and automated retraining

```
every component ──/metrics──► Prometheus :9090 ──rules──► Alertmanager :9093
  backend :8000 (HTTP, model calls, dependencies)          │ action="retrain"
  scheduler :8001 (runs, steps, drift, live error,         ▼
     data lag, data quality, model versions)        scheduler POST /retrain (token)
  node-exporter (host), cAdvisor (every container,           │ 24 h cooldown
     incl. mlflow + model servers), mongodb-exporter,        ▼
     nginx-exporter (frontend), alertmanager, grafana   retrain run → promotion gate
                         │                                   → promote or keep
                         ▼
                 Grafana :3000 (5 dashboards, provisioned from monitoring/grafana)
browser #/pipeline ─► backend /api/monitoring ─► latest report (Mongo) + active alerts
```

The scheduler computes a **monitoring report** every 2 minutes and at the end
of every run (`params.yaml` → `monitoring`):

| Check | How | Alert |
|---|---|---|
| Returns drift | PSI and KS of each stock's last 60 daily log returns vs the 250 before (5 quantile bins) | `ReturnsDriftDetected` when ≥ 30 % of stocks have PSI > 0.25 → **retrain** |
| Live forecast error | stored forecasts of the last 30 days vs the prices that came in, and the naive forecast on the same points | `ForecastErrorDegraded` when MAPE > 1.25 × naive and > 3 % (≥ 20 points) → **retrain** |
| Sentiment shift | PSI of article scores, last 14 days vs the 90 before | `SentimentShift` (info, no retrain: the model is pretrained) |
| Data freshness | trading days the newest price lags the latest session | `DataStale` (> 2) |
| Data quality | failed checks in `reports/data_quality.json` | `DataQualityFailed` |

Plus availability (`TargetDown`, `ModelServerDown`, `DependencyDown`), API
(`ApiHighErrorRate` > 1 %, `ApiHighLatency` p95 > 300 ms), pipeline
(`PipelineRunFailed`, `PipelineRunPartial`, `PipelineNotRunning`,
`MonitoringStale`) and resources (`HostMemoryLow`, `HostDiskLow`,
`ContainerNearMemoryLimit`, `ContainerOomKilled`). Rules:
`monitoring/prometheus/rules/finsight.yml`, with unit tests in
`monitoring/prometheus/tests/`.

**1. One-time setting in `.env`**
```bash
echo "GRAFANA_ADMIN_PASSWORD=$(python -c 'import secrets; print(secrets.token_urlsafe(16))')" >> .env
```

**2. Update and start**
```bash
pip install -r requirements/dev.txt
pytest tests/unit -q                              # expect all to pass (~225)
docker compose up -d --build backend frontend     # new API endpoint, console card, nginx stub_status
docker compose restart scheduler                  # new code is mounted; the process must restart
docker compose up -d prometheus alertmanager grafana node-exporter cadvisor mongodb-exporter nginx-exporter
docker compose ps                                 # all "Up"; prometheus "healthy"
```
Extra memory: about 0.5 GB in total (Prometheus 384 MB cap, Grafana 256 MB, the rest small).

**3. Look around**

| What | Where |
|---|---|
| Every scrape target is `UP` | http://localhost:9090/targets |
| Alert rules and their state | http://localhost:9090/alerts |
| Grafana (user `admin`, password from `.env`) — FinSight folder: Overview, API, Pipeline, Models & drift, Infrastructure | http://localhost:3000 |
| Alertmanager: what is firing, silences | http://localhost:9093 |
| Monitoring card: alerts, drift per stock, live MAPE vs naive | http://localhost:8080/#/pipeline |
| Recompute the report now | `curl -s -X POST localhost:8001/monitor -H "X-Finsight-Token: $SCHEDULER_TOKEN" \| python -m json.tool \| head -40` |

**4. Drift drill: the whole loop end to end** (about 10 minutes)
```bash
docker compose exec scheduler python -m finsight.monitoring.drill start --minutes 30
```
The drill makes the monitor multiply the recent returns by 4 and shift recent
sentiment down by 0.5. It **never changes stored data** and ends by itself.
- within 2 min: `finsight_drift_detected{feature="returns"} = 1`, the console shows a drill banner and every stock as *Drifted*;
- after 5 min: `ReturnsDriftDetected` turns from *pending* to *firing* (Prometheus → Alerts);
- 1 min later Alertmanager calls `POST /retrain`; a **retrain** run appears in the console with reason `alert: ReturnsDriftDetected`;
- the retrain takes 20–60 min, and the promotion gate decides whether the new LSTM replaces the current one;
- a second alert within 24 h is answered `200 {"ignored": "cooldown"}` and counted in `finsight_retrains_skipped_total`.
```bash
docker compose exec scheduler python -m finsight.monitoring.drill status
docker compose exec scheduler python -m finsight.monitoring.drill stop
```
If memory is tight during the retrain: `docker compose stop sentiment-serve grafana`.

**5. Check the rules and the Alertmanager config yourself**
```bash
docker compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
docker compose exec -w /etc/prometheus/tests prometheus promtool test rules finsight_test.yml
docker compose exec alertmanager amtool check-config /etc/alertmanager/alertmanager.yml
```

**Changing a dashboard:** edit `scripts/build_dashboards.py`, run
`make dashboards`, and Grafana picks up the JSON within 30 s. Edits made in the
Grafana UI are not saved to the repo.

**Changing a rule:** edit `monitoring/prometheus/rules/finsight.yml`, then
`curl -X POST localhost:9090/-/reload`.

### Freeing disk space
Prometheus keeps at most 15 days or 1 GB. If `HostDiskLow` fires:
```bash
docker image prune -f && docker builder prune -af   # old images and build cache
pip cache purge; conda clean -a -y
dvc gc -w -f                                        # data no longer used by the workspace (never -c)
```
Never run `docker volume prune` or `docker compose down -v`: the volumes hold
MongoDB, MLflow, Prometheus and Grafana data.

## Day 9 runbook — testing, security and code quality

Plan and acceptance criteria: [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md).
Latest results: [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) and every test case in
[`docs/TEST_CASES.md`](docs/TEST_CASES.md), both written by `make test-report`.

**1. Secrets: encrypt `.env` (once per machine for the key)**
```bash
make secrets-key        # ~/.config/finsight/secret.key, mode 600; copy it to a USB stick
chmod 600 .env
make secrets-encrypt    # .env -> secrets/.env.enc (commit this file, never .env)
make secrets-check      # every line should say OK
```
After editing `.env`, run `make secrets-encrypt` again. On a new machine: copy
the key to `~/.config/finsight/secret.key` (mode 600), then `make secrets-decrypt`.

**2. Encrypted backups**
```bash
make backup             # MongoDB + reports + config -> backups/finsight-<time>.tar.enc
make backup-models      # the same plus models/
make verify-backup      # decrypt the newest one and check every file's SHA-256
python -m finsight.security.backup restore backups/<file>.tar.enc --to /tmp/finsight-restore \
    --mongo-db finsight_restore     # never the live database
```

**3. Pre-commit hooks**
```bash
pre-commit install      # once per clone: hooks then run on every git commit
make pre-commit         # all hooks on all files
```

**4. Tests against the running stack**
```bash
docker compose up -d
make e2e                # 16 end-to-end smoke tests (each skips if its service is down)
make load-test          # 10 users for 60 s -> reports/load_test.json
make test-report        # all suites + criteria -> docs/TEST_REPORT.md, docs/TEST_CASES.md
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
