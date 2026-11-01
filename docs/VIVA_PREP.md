# FinSight — Viva preparation

Short answers to the questions an evaluator is most likely to ask, each with the file or
command that proves it. Say the answer in your own words, then open the evidence. The
demo itself is in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md), and every guideline item is mapped
in [`COMPLIANCE.md`](COMPLIANCE.md).

**Numbers to know by heart** (from `docs/TEST_REPORT.md`):

| | |
|---|---|
| LSTM test MAPE vs naive | **3.21 % vs 3.34 %** (skill 3.9 %), 150 test windows |
| Sentiment in production | **FinBERT**, macro-F1 **0.657** vs majority baseline 0.264 (Twitter financial news) |
| Portfolio, out of sample | Sharpe **0.05** vs equal weight **0.46** (weekly walk-forward over a year) |
| Grid vs continuous optimum | 0.035 Sharpe given up by the 10 % grid |
| Tests | **278** unit, **4** integration, **16** end-to-end, plus the alert-rule tests |
| API under load | p95 **17 ms**, 0 % errors (10 users, 60 s). Live FinBERT call p95 89 ms |
| Data quality | 21 checks on every run |
| Reproducibility | Retrained from the commit: **identical** metrics (`docs/REPRODUCIBILITY.md`) |

---

## 1. The ten questions from COMPLIANCE §D

**1. Why does the LSTM barely beat "price stays the same", and why is that acceptable?**
Daily prices are close to a random walk: tomorrow's best guess is today's price. A model
that beat naive by a wide margin on 14-day horizons would be suspicious (look-ahead
leakage is the usual cause, and the baseline repository had it). FinSight measures honestly:
same test windows, purged split, naive and drift baselines. The acceptance criterion is
relative ("no more than 2 % worse than naive"), and the gate only promotes a model that beats
the champion on the same data. The residual head starts as exactly naive, so training only
moves away from naive where the data supports it.
*Evidence:* `reports/lstm_metrics.json`, D-07, Day 3 log, TEST_PLAN §5.

**2. Why Spark and not Ray on this data size? What does the benchmark show?**
The rubric asks for Ray **or** Spark **or** Kubernetes. Spark fits the job, which is a
batch ETL of tables (joins with a calendar, windows, aggregates), and it runs in `local[4]`
with a 1 GB driver, inside the 8 GB budget. Ray is for parallel Python tasks and training,
which this project doesn't need. On ~10k rows Spark's start-up dominates. The benchmark
shows throughput rising with volume: about 1.7k, 9.5k and 18.7k rows/s at 1×, 10× and 50×.
That growth is the argument for Spark at scale.
*Evidence:* `make benchmark` → `reports/etl_benchmark.json`, D-06, Day 2 log.

**3. How does DVC act as CI here? What happens if a unit test fails at 06:30?**
`unit_tests` is the first stage in `dvc.yaml`. `export` depends on its report, so a failing
suite stops `dvc repro` before any data moves. `validate` gates Spark the same way, and
`register` gates promotion. At 06:30 the scheduler's data step fails, the run is marked
*failed*, later steps are skipped, `PipelineRunFailed` fires, and the console shows the log
tail with the failing test. Yesterday's data and model keep serving. This really happened:
run `5ec75aa8d7f1` on 27 Oct.
*Evidence:* `dvc dag`, Pipeline Console → run history, `docs/TEST_NOTES.md`.

**4. Walk through a drift alert from gauge to a new model in production. What stops a retrain loop?**
The monitor (inside the scheduler, every few minutes) computes PSI and KS of the last 60
days of returns against the year before, per stock. It exports `finsight_drift_detected`.
`ReturnsDriftDetected` fires when at least 30 % of the stocks drift for 5 minutes.
Alertmanager calls `POST /retrain` on the scheduler. The scheduler runs the full DVC
pipeline. The `register` gate compares the new model with the champion on the same test
windows, and moves `@production` only if it is strictly better. The model servers notice
the alias move and restart with the new version. Two things stop a loop: Alertmanager
repeats the notification only every 24 h, and the scheduler ignores `/retrain` within 24 h
of the last retrain (`ignored: cooldown`). A retrain alert during a daily run is queued.
*Evidence:* README Day 8 drift drill, `monitoring/prometheus/rules/finsight.yml`, Day 8 log.

**5. Why aliases instead of stages in the MLflow registry?**
MLflow has deprecated stages in favour of aliases. An alias is a movable pointer
(`@production`) that the serving containers resolve, so promotion and rollback are one API
call and need no rebuild. FinSight sets both, so the UI also shows the
"Staging/Production" view that the proposal describes.
*Evidence:* D-08, `src/finsight/registry/`.

**6. Why is FinBERT not judged on Financial PhraseBank?**
FinBERT was fine-tuned *on* PhraseBank, so scoring it there would be test-set leakage and
would flatter it. The gate compares the candidates on the Twitter financial-news validation
set, which neither model was trained on. PhraseBank is still reported, flagged
`contaminated` for FinBERT.
*Evidence:* D-13, `reports/promotion.json`.

**7. How is the frontend decoupled from the models? What changes if the backend moves?**
Three containers and two REST hops. The browser knows only `BACKEND_URL`, written into
`/config.js` by nginx when the container starts. The backend knows only `LSTM_SERVE_URL`
and `SENTIMENT_SERVE_URL`. The frontend container has no Python and no model. If the
backend moves, change `BACKEND_URL` and restart the frontend: no rebuild. The same variable
updates the Content-Security-Policy.
*Evidence:* ARCHITECTURE §4, Day 6 log, the e2e test that the portal does not proxy the API.

**8. How would you reproduce the model from three days ago?**
Every run is tagged with its Git commit, and `reports/lstm_train.json` links the commit to
the MLflow run ID. Run `python scripts/repro_drill.py --commit <hash or tag>`. It clones,
pulls that commit's data with DVC, retrains with `mlflow run . -e train_lstm` and compares
the metrics. If you only need to *serve* the old model, nothing needs to retrain: every
version is in the registry. `make rollback-lstm` moves `@production` back.
*Evidence:* `docs/REPRODUCIBILITY.md`, `make repro-drill`.

**9. Where is the encryption key, and what happens if it is lost?**
`~/.config/finsight/secret.key`, outside the repository, mode 600, with a copy on a USB
stick. It encrypts `secrets/.env.enc` (committed) and the backups. If it is lost, the
encrypted copies can't be read: that is the point of encryption. Recovery: create a new
key, re-create `.env` from the provider dashboards, re-encrypt, and take a new backup. The
working copy itself is not encrypted, so the running system is not affected.
*Evidence:* `make secrets-check`, Day 9 log, LLD §8b.

**10. What would you change to run this for 500 stocks?**
- **Data.** MarketStack's free plan is 100 requests a month and each ticker costs one, so
  a paid plan, or yfinance as the primary source.
- **Spark.** No change: the benchmark shows it scales.
- **Sentiment.** News volume grows 50×. Truncate FinBERT input to 128 tokens and score in
  parallel, or use a GPU.
- **Portfolio.** The exhaustive grid (92,378 portfolios for 10 stocks) is impossible for
  500. Use the continuous optimiser (already there as SLSQP) or quadratic programming, with
  a longer estimation window.
- **LSTM.** The global model works better with more tickers. Perhaps one model per sector.
- **Orchestration.** Training in parallel is where Ray would start to earn its place.

---

## 2. Tool choices: "why didn't you use X?"

| Asked about | Answer in one breath | Where |
|---|---|---|
| Airflow | Its scheduler, webserver and database need 2–3 GB, which an 8 GB laptop doesn't have next to Spark and the models. The DAG already lives in `dvc.yaml`; a ~600-line scheduler adds catch-up, one-run-at-a-time and a status API. | D-05, Day 7 |
| Cron | No catch-up after sleep, no "never twice", nothing for the console to read. | Day 7 |
| Ray | Not needed for this workload, and it competes for the same memory. Spark covers the rubric item. | D-06 |
| Cloud / S3 | Not allowed by the guidelines. Local DVC remote; MinIO would be the S3-compatible option. | D-04 |
| React / Streamlit | Plain HTML and ES modules served by nginx: no build step, ~80 MB image, ~4 MB RAM, and the UI stays a pure REST client. | Day 6 |
| Great Expectations | Same ideas (suite, severity, report) without the dependency tree. | Day 2 |
| Vault / SOPS / GPG | A server or extra tools for one laptop. Fernet authenticates as well as encrypts and was already a dependency. | Day 9 |
| Locust / JMeter | Ten threads from the standard library measure p95 on a laptop without a new dependency. | Day 9 |
| Kubernetes / Swarm | One machine: Compose gives the same container boundaries. Stateless services would move unchanged. | COMPLIANCE C.I |
| Docker Hub | Images are built locally from pinned Dockerfiles; pushing them adds nothing on one machine. | COMPLIANCE C.III |

## 3. Modelling and evaluation

- **Why one global LSTM?** About 139 windows per ticker is too few for ten models.
  Per-window scaling puts all tickers on one scale, giving ten times the data.
- **Why per-window MinMax scaling?** The scaler is fitted on the 100 input days only, so no
  future value leaks into the past. The baseline repo fitted on the whole series.
- **Why direct 14-step output instead of recursive?** Recursive forecasting feeds its own
  errors back for 14 steps.
- **How did you avoid leakage in the split?** A purged chronological split: train, a
  14-window gap, validation, a gap, test. The test set is touched only by `evaluate_lstm`.
- **Why select on validation MAPE?** It is scale-free. RMSE would be dominated by the most
  expensive stocks.
- **Explainability?** For the LSTM, occlusion attribution: hiding the last 10 days moves the
  forecast ~3 %, older blocks barely at all, as expected for a near-naive model. For
  sentiment, the top headlines behind each week. For the portfolio, the reality check.
- **Why is the portfolio's out-of-sample loss not a failure?** In sample, the optimum can
  never lose to equal weight, because equal weight is a grid point. Out of sample, a
  30-day estimate overfits. Losing to 1/N is a well-known result, and FinSight reports it.
- **Feature engineering?** Volatility as a second input was tested on validation with
  3 seeds, under a rule of "only if the gain exceeds 2 × the seed spread". It made
  validation worse, so it is not used.

## 4. Data engineering and operations

- **The MarketStack quota.** 100 requests a month, one per ticker. A ledger refuses calls
  beyond `budget − reserve`, already-current tickers are skipped using the NYSE calendar,
  and yfinance is the fallback. Every bar records its source.
- **Bad data.** Three zero prices from MarketStack were caught by `validate` before Spark.
  The fix: a sanitiser at the boundary, plus a `repair` command.
- **Idempotency.** Unique indexes and upserts, so re-running a day changes nothing.
- **What does the scheduler commit?** `dvc.lock` and `reports/` locally, after `dvc push`
  to the local remote. It never pushes to GitHub, so no Git credentials enter a container.
- **What happens when a model server is down?** The API serves the last stored forecast
  with `stale: true`, the status dot turns amber, and `ModelServerDown` fires after 5 minutes.
- **Memory on 8 GB.** Every service has a `mem_limit`. Pipeline steps run one at a time.
  The measured MLflow server OOM was fixed by turning off its job runner.

## 5. Testing

- **Why relative acceptance criteria?** "3 % MAPE" means nothing on its own. "No worse than
  naive" and "better than the majority class" are criteria a reader can judge.
- **Why AC-11 is FAIL (or was).** One early run failed at the unit-test gate, because of a
  test-isolation bug, and with five runs one failure weighs 20 %. The target stayed at
  90 % rather than being lowered to fit. `docs/TEST_NOTES.md` explains it.
- **Unit vs integration tests: an example of why both are needed.** mongomock hid an upsert
  bug that counted identical re-upserts as revisions. The integration test against real
  MongoDB found it on Day 1.
- **The test report can't lie.** `scripts/test_report.py` runs the suites and writes the
  report from the JUnit output.

## 6. Defending every proposal change (D-01 … D-13)

| # | Change | One-line defence |
|---|---|---|
| D-01 | 12 months of prices, not 6 | ~137 training windows per ticker instead of ~12, for the same quota. |
| D-02 | Quota ledger + yfinance fallback | 10 tickers × 21 days = 210 requests > the 100 allowed. |
| D-03 | Finnhub, not NewsAPI | NewsAPI's free tier only returns 30 days. |
| D-04 | Local DVC remote | The guidelines rule out cloud platforms. |
| D-05 | Own scheduler, not Airflow | Memory; DVC already holds the DAG. |
| D-06 | Spark, not Ray | Rubric says "or"; Spark fits the job and the RAM. |
| D-07 | Direct, global, per-window-scaled LSTM | No compounding error, no leakage, 10× the data. |
| D-08 | Aliases and stages | Aliases are MLflow's current way; stages keep the proposal's view. |
| D-09 | Walk-forward portfolio backtest | The in-sample comparison with equal weight is meaningless. |
| D-10 | Python 3.11 everywhere | Stable wheels for PySpark, PyTorch-CPU and MLflow; environment parity. |
| D-11 | DistilBERT measured and calibrated first | Movie-review sentiment may not transfer to finance, so it was measured. |
| D-12 | Exhaustive grid + SLSQP check + min-vol fallback | Deterministic; measures what the grid gives up; avoids picking the most volatile portfolio. |
| D-13 | FinBERT as a challenger, gated on unseen data | Champion/challenger is what the registry is for; PhraseBank would leak. |

## 7. Honest limitations (say them before you are asked)

- The LSTM is only slightly better than naive. That is a property of daily prices, and it
  is measured and shown, not hidden.
- The optimised portfolio loses to equal weight out of sample.
- Ten stocks, one laptop: scaling is argued from the Spark benchmark, not demonstrated on a
  cluster.
- Images are not pushed to a registry, and there is no Kubernetes.
- The model artifacts in the working copy are not encrypted in place. Backups are, and the
  reason is in the Day 9 log.
