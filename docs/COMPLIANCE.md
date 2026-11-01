# FinSight — Guideline compliance

Every question in the *AI Application Evaluation Guideline* and every requirement in the
*AI Application Development Guidelines*, with where FinSight answers it. Use it as the
checklist before the demo and the viva. Paths are relative to the repository root.

Legend: ✅ done · ◐ partly / by design different (explained) · — not applicable

---

## A. Mandatory documents

| # | Document | Where | |
|---|---|---|---|
| 1 | Architecture diagram with an explanation of the blocks | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), diagram [`docs/diagrams/architecture.png`](diagrams/architecture.png) | ✅ |
| 2 | High-level design document narrating the design choices and rationale | [`docs/HLD.md`](HLD.md) (+ [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) for every detailed decision) | ✅ |
| 3 | Low-level design document quoting the endpoint definitions and I/O specification | [`docs/LLD.md`](LLD.md), [`docs/api/openapi.json`](api/openapi.json) | ✅ |
| 4 | Test plan & test cases | [`docs/TEST_PLAN.md`](TEST_PLAN.md), [`docs/TEST_CASES.md`](TEST_CASES.md), [`docs/TEST_REPORT.md`](TEST_REPORT.md) | ✅ |
| 5 | User manual for a non-technical user | [`docs/USER_MANUAL.md`](USER_MANUAL.md); also the in-app **Help** page | ✅ |

## B. Evaluation guideline

### Demonstration — Web application front-end UI/UX

| Question | Answer / evidence | |
|---|---|---|
| Is your UX intuitive for your problem statement? | One page per stock answers "where has it been, where may it go, what does the news say"; overview lists all ten; search by name, ticker or product. `frontend/`, USER_MANUAL §2–5 | ✅ |
| How easy is it for a non-technical user? | Plain-language captions under every chart ("the model expects $X… treat as a rough guide"), help page, glossary, no jargon in the UI | ✅ |
| Is the UX foolproof? | Unknown ticker → "not covered" page; invalid input → clear message; confirmation before Run/Retrain; buttons disabled while a run is busy; every section fails independently with **Try again**; stale forecast clearly labelled | ✅ |
| Is the front end free of UI errors? | 12 frontend unit tests (`tests/unit/test_frontend.py`), e2e portal tests; no console errors in the rehearsal screenshots | ✅ |
| Colours, shapes, look and feel | Consistent palette (green up / red down, never colour alone: arrows and signs too), light and dark themes, Chart.js vendored | ✅ |
| How responsive is your UI? | Mobile layout (day 6/7 screenshots), API p95 17 ms under 10 users (AC-08) | ✅ |
| Is there a user manual? | `docs/USER_MANUAL.md` + Help page | ✅ |

### Demonstration — ML pipeline visualisation

| Question | Answer / evidence | |
|---|---|---|
| A separate UI screen to visualise the ML pipeline, including data ingestion/engineering? | **Pipeline Console** (`#/pipeline`): live DVC DAG of all stages (ingestion → validation → Spark ETL → training → gate → scoring → portfolio) with status colours | ✅ |
| If MLOps tool UIs are used, how are they orchestrated seamlessly? | Console links to MLflow experiments, Grafana dashboards and API docs; the same run IDs appear in the console, MLflow and logs | ✅ |
| A pipeline management console? | Run now / Retrain now (with confirmation), scheduler status, next run | ✅ |
| A console of errors, failures and successful runs? | Run history with status filter; per-run steps, error message and log tail; Grafana Pipeline dashboard; `PipelineRunFailed` alert | ✅ |
| Speed and throughput of the pipeline? | Console charts "Run duration" and "Throughput"; `reports/etl_metrics.json` (rows/s); `make benchmark` → `reports/etl_benchmark.json` (1×/10×/50×); Prometheus `finsight_pipeline_throughput` | ✅ |

### Software engineering — design principle

| Question | Answer / evidence | |
|---|---|---|
| Is there a design document? | HLD, LLD, ARCHITECTURE, DESIGN_DECISIONS | ✅ |
| OO or functional paradigm? | **Both, deliberately**: pure functions for transformations and maths (`lstm/data.py`, `metrics.py`, `portfolio/optimize.py`, `monitoring/drift.py`); classes where there is state or a contract (`PriceProvider` / `NewsProvider` abstract classes and their MarketStack, yfinance and Finnhub implementations, `IngestionService`, `ModelClient`, `SchedulerService`, `EncryptingWriter`, Pydantic schemas, exception hierarchy under `FinSightError`) | ✅ |
| LLD specifies endpoint definitions with I/O? | LLD §2 (every route, parameters, response fields, errors) + OpenAPI generated from the Pydantic models; `tests/unit/test_api_contract.py` fails if they drift | ✅ |
| Is there an architecture diagram? | `docs/diagrams/architecture.png` | ✅ |
| Is there an HLD diagram? | `docs/diagrams/hld_layers.png`, plus sequence diagrams `daily_run.png`, `retrain_loop.png` and the DVC DAG `dvc_dag.png` | ✅ |
| Loose coupling between frontend UI and model inference via configurable REST | Separate containers; browser → backend via `BACKEND_URL` (config.js); backend → models via `LSTM_SERVE_URL` / `SENTIMENT_SERVE_URL`; ARCHITECTURE §4; e2e test that the portal does not proxy the API | ✅ |

### Software engineering — implementation

| Question | Answer / evidence | |
|---|---|---|
| Standardised Python coding style? | PEP 8 via **ruff** (pycodestyle, pyflakes, isort, naming, bugbear, pyupgrade, pydocstyle, bandit) + **black**; `make lint`; pre-commit hooks on every commit | ✅ |
| Is logging implemented? | Structured JSON logs (`finsight.common.logging`) with `run_id` / `request_id` on every line; per-stage log files in `logs/`; log tails in the console | ✅ |
| Comprehensive exception handling? | Exception hierarchy (`finsight.common.exceptions`), retries only on transient errors (`common.retry`), uniform API error body with stable codes (LLD §3), graceful degradation matrix (LLD §7) | ✅ |
| Is the design document strictly adhered to? | OpenAPI generated from code; contract test; DESIGN_DECISIONS records every deviation from the proposal with its reason | ✅ |
| Comprehensive inline documentation? | Google-style docstrings enforced by ruff `D` rules on every public module, class and function; comments explain *why* | ✅ |
| Do the APIs follow the interface specification? | `tests/unit/test_api_contract.py`, `tests/unit/test_api.py`, e2e tests | ✅ |
| Unit tests implemented? | 265 unit tests, 4 integration, 16 e2e (TEST_REPORT) | ✅ |

### Software engineering — testing

| Question | Answer / evidence | |
|---|---|---|
| Is there a test plan? | `docs/TEST_PLAN.md` | ✅ |
| Is there an enlistment of test cases? | `docs/TEST_CASES.md` (IDs UT-/IT-/E2E-, generated) | ✅ |
| Test report with number of test cases passed and failed? | `docs/TEST_REPORT.md` (`make test-report`) | ✅ |
| Definition of acceptance criteria? | TEST_PLAN §5 (AC-01…AC-12, with rationale) | ✅ |
| Did the software meet the acceptance criteria? | TEST_REPORT §2; AC-11 explained in §5 Notes until enough daily runs accumulate | ✅ / ◐ |

### AIOps — data engineering

| Question | Answer / evidence | |
|---|---|---|
| Data ingestion / transformation pipeline? | `finsight.ingestion` (MarketStack, yfinance fallback, Finnhub; quota ledger; idempotent upserts) → DVC `export` → `validate` → `process` | ✅ |
| Should use Ray or Spark or Kubernetes | **Apache Spark** (local mode) in `finsight.etl` (`process` stage) | ✅ |
| Throughput and speed? | `reports/etl_metrics.json`, `reports/etl_benchmark.json`, console Throughput chart | ✅ |
| One of Ray/Spark/K8s or custom pipeline present? | Spark + DVC pipeline | ✅ |

### AIOps — source control & CI

| Question | Answer / evidence | |
|---|---|---|
| How is CI implemented using DVC? | `dvc.yaml`: `unit_tests` is the first stage and every data/model stage depends on it; `validate` gates Spark; `register` gates promotion; the scheduler runs `dvc repro` daily and commits `dvc.lock` | ✅ |
| DVC DAG representing the CI pipeline? | `dvc dag`; `docs/diagrams/dvc_dag.png`; live in the Pipeline Console | ✅ |
| Git, Git LFS and/or DVC for source, data, model versioning? | Git for code and reports, **Git LFS** for images (`.gitattributes`), **DVC** (local remote) for data and models; tags `day1`…`day10` | ✅ |

### AIOps — experiment tracking

| Question | Answer / evidence | |
|---|---|---|
| How are experiments tracked while models are built? | MLflow server; each LSTM training = a parent run + one child run per grid configuration; evaluation and sentiment candidates logged too | ✅ |
| Metrics, parameters and artifacts tracked? | Params: every `lstm` hyper-parameter, seed, window sizes. Metrics: per-epoch train/val loss, val/test MAPE, naive/drift baselines, skill, directional accuracy, per-horizon MAPE, attribution. Artifacts: packaged pyfunc model with signature, learning curves, forecast and attribution plots, full evaluation JSON | ✅ |
| Beyond autolog, other information tracked? | Lineage tags (`finsight.models.tracking`): Git commit + dirty flag, DVC hash of the training data, MongoDB export SHA-256, package versions, platform, seed; registry tags (acceptance, gate decision, eval fingerprint) | ✅ |

### AIOps — exporter instrumentation & visualisation

| Question | Answer / evidence | |
|---|---|---|
| How is Prometheus instrumentation implemented? | `prometheus_client` in the backend (HTTP latency/count by route and status, model-call latency, dependency up) and the scheduler (runs, steps, durations, throughput, drift, live error, data lag, data quality, model versions); exporters for host, containers, MongoDB, nginx | ✅ |
| Which information points are monitored? | LLD §6 lists every metric; alert rules in `monitoring/prometheus/rules/finsight.yml` | ✅ |
| Are all components monitored? | All 15 services: own `/metrics` (backend, scheduler, Prometheus, Alertmanager, Grafana), cAdvisor for every container (including MLflow and model servers), node, MongoDB and nginx exporters; e2e test checks every target is up | ✅ |
| Grafana for near-real-time visualisation? | 5 provisioned dashboards (auto-refresh every 30 s) generated by `scripts/build_dashboards.py` | ✅ |

### AIOps — software packaging

| Question | Answer / evidence | |
|---|---|---|
| MLflow for APIfication of the model for inference? | `mlflow models serve` of `@production` (lstm-serve, sentiment-serve) | ✅ |
| MLprojects for identical dev vs test environments? | `MLproject` with `conda_env: environment.yml` (the developer's own env file, pinning `requirements/*.txt`, also used by the images); reproducibility drill: [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | ✅ |
| FastAPI to expose the APIs the UI needs? | `finsight.api` | ✅ |
| Backend and frontend dockerised? | `docker/Dockerfile.backend`, `docker/Dockerfile.frontend` (+ pipeline, mlflow, serve) | ✅ |
| Docker Compose with them as separate services? | `backend` and `frontend` services in `docker-compose.yml` | ✅ |

### Viva

Preparation material: HLD §5 (choices and alternatives), DESIGN_DECISIONS (D-01…D-13 and the
day-by-day log, each with evidence), TEST_PLAN §5 (why each threshold), the "Viva notes" in
each day's guide. Challenge questions to rehearse are in §D below.

## C. Development guidelines

### I. Core principles

| Principle | How FinSight meets it | |
|---|---|---|
| Automation | Scheduler runs ingest → CI → train/score → forecast → monitor → version daily; alerts trigger retraining; promotion deploys | ✅ |
| Reproducibility (Git commit + MLflow run ID) | Every run tagged with `git_commit`; `reports/lstm_train.json` holds the run ID; `scripts/repro_drill.py` → REPRODUCIBILITY.md | ✅ |
| Continuous integration | DVC DAG with tests first; pre-commit; `make test-report` | ✅ |
| Monitoring and logging | Prometheus/Grafana/Alertmanager; drift + live error; JSON logs | ✅ |
| Collaboration | README runbooks, design docs, decisions log, Pipeline Console shared view, Git history with tags | ✅ |
| Version control of all artifacts | Git (code, configs, reports), DVC (data, models), MLflow registry (model versions), Git LFS (images) | ✅ |
| Environment parity (Docker, compose) | One dependency set for env, MLproject and images; Compose for distribution. Swarm/Kubernetes not applicable on one laptop | ✅ |

### II. Lifecycle stages

| Stage | Requirement | Evidence | |
|---|---|---|---|
| A | Business problem and measurable success metrics | HLD §1 | ✅ |
| A | Data sources, formats, potential biases documented | `docs/DATA_CARD.md` | ✅ |
| A | EDA | `notebooks/01_eda.ipynb`, `reports/figures/`, DATA_CARD | ✅ |
| A | Version-controlled collection scripts; automated ingestion + validation; quality checks and monitoring | `finsight.ingestion`, `validate` stage (21 expectations), `finsight_data_quality_*` gauges, `DataStale` alert | ✅ |
| B | Cleaning, transformation, feature engineering, automated and versioned | Spark `process` stage (calendar alignment, bounded fill, returns, weekly news aggregates); zero-price repair | ✅ |
| B | Track feature importance / impact | Occlusion attribution (`evaluate_lstm`), volatility-feature experiment (`make feature-experiment`) | ✅ |
| C | Model selection, training, hyper-parameter experiments, evaluation vs alternatives | 8-config grid with validation selection; naive/drift baselines; two sentiment candidates compared by the gate | ✅ |
| C | Automated training/evaluation, tracking, containerised training env | DVC stages, MLflow, pipeline image | ✅ |
| D | Deployment strategy, serving, automated deployment, production monitoring, rollback | Online serving via MLflow; promotion gate moves `@production`; servers follow the alias; `make rollback-lstm` | ✅ |
| E | Performance monitoring, drift detection, retraining, versioning | Live error vs naive, PSI/KS drift, alert-driven retrain with cooldown and queue, registry versions | ✅ |

### III. Technology stack

| Recommended | Used | |
|---|---|---|
| Git LFS, DVC | Both | ✅ |
| Spark / Ray / Kubernetes | Spark | ✅ |
| MLflow | Tracking, registry, serving, MLproject | ✅ |
| Docker, Compose, Dockerhub | Docker + Compose; images built locally (no registry needed on one machine) | ◐ |
| CI/CD: DVC | DVC | ✅ |
| Serving: MLflow Serve / Ray Serve / REST | MLflow Serve behind a FastAPI REST API | ✅ |
| Monitoring: Prometheus, Grafana | Both, plus Alertmanager | ✅ |
| Cloud platforms: not allowed | None used (local DVC remote, local MLflow, local everything) | ✅ |

### IV. Best practices

| Practice | Evidence | |
|---|---|---|
| Code quality | ruff + black + pre-commit; docstrings; typed | ✅ |
| Unit, integration and end-to-end tests | `tests/unit`, `tests/integration`, `tests/e2e`; load test | ✅ |
| Security (encryption) for data and models | `secrets/.env.enc`; encrypted, verified backups of MongoDB, reports and models (`make backup-models`); LLD §8b | ✅ |
| Scalability | Spark benchmark at 50× data; stateless backend; separate model servers; load test | ✅ |
| Explainability | LSTM occlusion attribution; headlines behind each sentiment week; portfolio reality check | ✅ |
| Documentation | README runbooks, ARCHITECTURE, HLD, LLD, DATA_CARD, TEST_*, USER_MANUAL, DESIGN_DECISIONS | ✅ |

## D. Questions to rehearse for the viva

1. Why does the LSTM barely beat "price stays the same", and why is that acceptable?
2. Why Spark and not Ray on this data size? What does the benchmark show?
3. How does DVC act as CI here? What happens if a unit test fails at 06:30?
4. Walk through a drift alert from gauge to new model in production. What stops a retrain loop?
5. Why aliases instead of stages in the MLflow registry?
6. Why is FinBERT not judged on Financial PhraseBank?
7. How is the frontend decoupled from the models? What changes if the backend moves?
8. How would you reproduce the model from three days ago?
9. Where is the encryption key, and what happens if it is lost?
10. What would you change to run this for 500 stocks?
