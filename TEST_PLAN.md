# FinSight — Test plan

This plan says what is tested, how, where, and what counts as "good enough to ship".
The results of the latest run are in [TEST_REPORT.md](TEST_REPORT.md), and every
individual test case is listed in [TEST_CASES.md](TEST_CASES.md). Both are written by
`python scripts/test_report.py` (`make test-report`), so they always match the code.

## 1. Scope

**In scope**

| Area | What is checked |
|---|---|
| Data ingestion | Providers (Marketstack, Finnhub, NewsAPI), retries and quota limits, zero-price repair, idempotent upserts into MongoDB |
| ETL and validation | Spark feature build, export, Great-Expectations-style data-quality checks, DVC stages |
| Models | LSTM forecaster (training, windows, metrics vs the naive baseline), sentiment classifiers (macro-F1 vs the majority baseline), portfolio optimiser |
| Registry | MLflow registration, `@production` alias, champion/challenger promotion gate |
| Backend API | Every route, error codes, CORS, caching, degradation when a model server is down, `/metrics` |
| Frontend | Static portal: pages, config, security headers |
| Orchestration | Scheduler (daily run, catch-up, cooldown, `/retrain` webhook and its queue), pipeline run records |
| Monitoring | Drift (PSI) and forecast-error monitor, Prometheus rules (promtool), Alertmanager routing, Grafana provisioning |
| Security | Encryption of secrets and backups at rest, key handling, localhost-only ports, non-root containers, no plaintext secrets, CORS |
| Non-functional | Latency and error rate under concurrent users, live model endpoint latency, scheduled-run reliability |

**Out of scope**: the accuracy of third-party data (Marketstack, Finnhub, NewsAPI);
browser compatibility beyond current Chrome/Firefox; multi-machine or cloud deployment
(the project runs on one laptop by design); trading performance — FinSight is an
analysis tool, not investment advice.

## 2. Test levels and strategy

| Level | Where | How it runs | Needs |
|---|---|---|---|
| **Unit** | `tests/unit/` | `make test` | nothing: in-memory MongoDB (mongomock), stub model servers, tiny synthetic data; Spark/torch/transformers tests skip themselves when the library is missing |
| **Integration** | `tests/integration/` (marker `integration`) | `make test-int` | the real MongoDB container (`docker compose up -d mongo`) |
| **Rule tests** | `monitoring/prometheus/tests/finsight_test.yml` | `make rules-test` or inside the Prometheus container | `promtool` |
| **End-to-end smoke** | `tests/e2e/` (marker `e2e`) | `make e2e` | the full stack (`docker compose up -d`); each test skips if its service is not reachable |
| **Load** | `scripts/load_test.py` | `make load-test` | the full stack; 10 simulated users for 60 s |
| **Security** | `tests/unit/test_security.py` | part of `make test` | nothing; it reads `docker-compose.yml`, the Dockerfiles and `.gitignore` |
| **Static checks** | ruff, black, pre-commit hooks | `make lint`, `pre-commit run --all-files`, on every commit | dev requirements |

Principles:

* **Test behaviour through public interfaces** (HTTP routes, CLI entry points, module
  functions), not private helpers, so refactoring does not break tests.
* **No network in unit tests.** External APIs are replaced with recorded or synthetic
  responses; anything slow (Spark, torch) is kept to tiny inputs.
* **Models are tested twice**: unit tests check the mechanics (shapes, windows, no
  look-ahead), and the acceptance criteria check quality on held-out data against a
  simple baseline.
* **Each bug that reached the running stack gets a regression test** (examples: zero
  prices from the provider, the lost retrain alert, the PSI noise threshold).

## 3. Environments

| Environment | Used for |
|---|---|
| Developer laptop, Python 3.11 virtualenv (`requirements/dev.txt`) | unit, integration, lint |
| Docker Compose stack on the same laptop (12 cores, 8 GB RAM), all ports bound to 127.0.0.1 | e2e, load, rule tests, demo |
| Test data | synthetic frames in the tests; the real 10-stock universe (2 years of daily prices, news) in the stack |

## 4. Entry and exit criteria

**Entry** (before testing a change): the code is committed on a branch, `make lint`
passes, and for e2e/load the stack is up and `/api/health` reports every dependency ok.

**Exit** (the change is ready): all acceptance criteria below are **PASS** in a fresh
`TEST_REPORT.md`, or any that are not are explained in the report's notes. A
skipped test is allowed only for an optional library that is not installed on that
machine (the report lists every skip and its reason).

## 5. Acceptance criteria

| ID | Criterion | Target | Measured by |
|---|---|---|---|
| AC-01 | Unit tests | 100 % of executed tests pass | pytest, `tests/unit` |
| AC-02 | Integration tests | 100 % pass | pytest `-m integration` |
| AC-03 | End-to-end smoke tests | 100 % pass against the running stack | pytest `-m e2e` |
| AC-04 | Alert-rule unit tests | all pass | `promtool test rules` |
| AC-05 | LSTM forecast quality | test-set MAPE ≤ 1.02 × the naive "tomorrow = today" MAPE | `reports/lstm_metrics.json` (held-out windows) |
| AC-06 | Sentiment model quality | macro-F1 of the production model > majority-class baseline | `reports/sentiment_eval_<model>.json` |
| AC-07 | Portfolio optimiser | grid optimum within 0.05 Sharpe of the continuous (SLSQP) optimum; out-of-sample Sharpe vs equal weight is reported | `reports/portfolio.json` |
| AC-08 | API latency | p95 < 300 ms on cached read endpoints with 10 concurrent users | `scripts/load_test.py` |
| AC-09 | API reliability under load | error rate (5xx, 429, timeouts) < 1 % | `scripts/load_test.py` |
| AC-10 | Live model latency | p95 < 2 s for `/api/sentiment/analyze` (CPU inference) | `scripts/load_test.py` |
| AC-11 | Pipeline reliability | ≥ 90 % of scheduled runs in the last 30 days succeed | `/api/pipeline/runs` |
| AC-12 | Data quality | no error-severity check fails in the latest validation | `reports/data_quality.json` |

Why these numbers:

* **AC-05**: daily stock prices are close to a random walk, so "beat the naive
  forecast" is the honest bar; a 2 % tolerance accepts a model that matches it while
  adding the trend and uncertainty the portal shows. The promotion gate uses the same rule.
* **AC-08**: 300 ms is where a page stops feeling instant; cached routes only read
  MongoDB, so this is a fair target on a laptop.
* **AC-10**: the live model runs a transformer on the CPU; 2 s is acceptable for the
  "analyse this headline" box.
* **AC-11**: the free data APIs occasionally fail; the scheduler retries and records
  `partial` runs, so 90 % leaves room for a bad API day without hiding real breakage.

## 6. Test cases

All test cases with their IDs (UT-, IT-, E2E-) and their latest result are generated
into [TEST_CASES.md](TEST_CASES.md). Highlights by area:

| Area | Examples |
|---|---|
| API | unknown ticker → 404 `unknown_ticker`; invalid period → 422; forecast cached once per day; stale forecast served when the model server is down, then 503 |
| Models | the LSTM train/test split is chronological and purged (no look-ahead); an untrained network equals the naive forecast; sentiment models are scored against the majority baseline; the promotion gate only promotes a better, accepted challenger |
| Scheduler | a daily run missed while the laptop slept is caught up; a retrain alert during a daily run is queued, not lost; cooldown blocks repeated retrains |
| Monitoring | PSI of identical distributions ≈ 0; the drift drill multiplies returns ×4 and is detected; rule tests fire `ReturnsDriftDetected`, `ApiHighErrorRate`, `ModelServerDown` |
| Security | wrong key / modified byte / dropped or reordered chunk is refused; key file must be mode 600 and outside the repository; every port is bound to 127.0.0.1; every image we build ends with a non-root `USER` |
| End-to-end | every stock page has history; forecasts have 14 future points; portfolio weights sum to 1; portal sends CSP and `X-Frame-Options: DENY`; Mongo refuses unauthenticated clients; all Prometheus targets are up |

## 7. Defect handling

A failing test blocks the commit (pre-commit runs lint; `make test` before pushing).
A failure found on the running stack is reproduced in a unit or e2e test first, then
fixed; the fix and the test go in the same commit.

## 8. How to run everything

```bash
docker compose up -d                 # the stack (for e2e, load and pipeline figures)
python scripts/load_test.py          # writes reports/load_test.json
python scripts/test_report.py        # runs all suites, writes TEST_REPORT.md and TEST_CASES.md
```
