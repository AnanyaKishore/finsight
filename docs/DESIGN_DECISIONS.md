# FinSight — Design Decisions Log

A running log of choices that differ from (or sharpen) the submitted proposal,
and of baseline-repo behaviour we deliberately do **not** copy. Each entry has
the rationale you can defend in the viva. New entries are appended per day.

---

## Part A — Changes to the proposal (and why)

| # | Proposal said | We do | Why |
|---|---|---|---|
| D-01 | Backfill **6 months** of prices | Backfill **12 months** (`ingestion.price_backfill_days: 365`) | A 100-day lookback + 14-day horizon on ~125 trading days leaves only ~12 training windows per ticker. 12 months (the MarketStack free-plan maximum) gives ~137 windows per ticker for the **same quota cost** (1 request per ticker). Lookback and horizon stay exactly as proposed. |
| D-02 | MarketStack, "runs daily incrementally" | MarketStack stays primary; **quota ledger + yfinance fallback** | Free plan = 100 requests/month and *every ticker counts as one request*. 10 tickers × ~21 trading days = 210 > 100. The ledger refuses calls that would exceed `budget − reserve`; the run then degrades gracefully to yfinance and every bar records its `source` (lineage). |
| D-03 | "News from a news source" | **Finnhub** company news | The baseline's NewsAPI free tier only returns the last ~30 days, so a 6-month backfill is impossible. Finnhub's free tier returns ~1 year of company news at 60 calls/min. |
| D-04 | DVC remote (your assignments used **S3**) | **Local DVC remote** (a directory outside the repo); MinIO optional | Guidelines §III: "Cloud Platforms: Not Allowed"; the proposal also promises "no cloud platforms are used". A local remote is the option shown in the DVC lecture. If you want to keep the S3 *workflow*, a self-hosted MinIO container is S3-compatible and is not a cloud platform. |
| D-05 | "Scheduled ingestion container", "the scheduler" | **APScheduler**-based Python service (Day 7), not Airflow | The baseline's Airflow stack (Postgres + Redis + Celery workers + webserver) needs ~2–3 GB RAM; on an 8 GB laptop it would not fit next to Spark, MLflow, DistilBERT and Grafana. Airflow is not in the guidelines' stack; DVC is the mandated pipeline/CI tool. |
| D-06 | Ray (in guidelines' stack) | **Not used**; Spark (local mode) is the data-engineering engine | The rubric asks for "Ray **or** Spark **or** Kubernetes". Spark in `local[4]` mode with a 1 GB driver fits the 8 GB budget; a Ray cluster would compete for the same memory without adding a capability we need. |
| D-07 | LSTM "predicts the next 14 days" | **Direct multi-output** LSTM (one forward pass → 14 values), one **global** model across tickers, **per-window** MinMax scaling | Recursive 1-step forecasting (baseline) compounds its own errors over 14 steps. Per-window scaling (fit on the 100-day input only) removes look-ahead leakage and lets one model learn from all 10 tickers (10× data). |
| D-08 | Registry "Staging and Production" | Classic **stages** *and* **aliases** (`@staging`, `@production`) | Newer MLflow versions recommend aliases and are deprecating stages (Lecture 3, §4.3). We set both: aliases are what the serving containers load; stages keep the UI view the proposal describes. |
| D-09 | Portfolio Sharpe "against an equal-weight baseline" | Compare on a **walk-forward** (out-of-sample) window | Equal weight (0.1 each) is itself a grid point, so the in-sample optimum can never lose to it — that comparison would be meaningless. We optimise on window *t−60…t−30* and evaluate on *t−30…t*. |
| D-10 | Python version unspecified (q4demo used 3.14) | **Python 3.11** everywhere (mamba env, Docker, MLproject) | PySpark, PyTorch-CPU and MLflow all ship stable 3.11 wheels; one version everywhere is what "environment parity" means. |

## Part B — Baseline behaviour we deliberately do NOT copy

Found while reviewing `MLOps_AI_App` and `MLops_AiApp_rnn_model_tracking`:

1. **`/api/prediction/{ticker}` returns synthetic numbers.** `Web App/app.py` builds the forecast from a seeded random slope plus noise; the stored LSTM predictions are ignored. FinSight serves real model output (or clearly labelled last-known output when degraded).
2. **API keys committed in source** (`stock_data_config.py`, `stock_data_dag.py`). FinSight keeps secrets only in a git-ignored `.env` (encrypted copy on Day 9).
3. **Scaler fit on the whole series** in training (leaks future min/max into the past). Fixed by per-window scaling (D-07).
4. **Inference reads the *oldest* 100 days**: `find(...).sort("date", 1).limit(100)`. FinSight always takes the most recent window.
5. **Validation split yields no windows**: 30 validation days with a 100-day sequence length produces zero validation sequences.
6. **Frontend served by the Flask backend** (`render_template`), i.e. not loosely coupled. FinSight's frontend is a separate container that calls the backend only through a configurable REST base URL (rubric: "strictly follow loose coupling").
7. **Duplicate inserts** in historical backfill (`insert_many` without a unique key). FinSight uses unique indexes and upserts everywhere.
8. **Monitoring stack commented out** in the root compose file, and only infrastructure metrics (node/Mongo). FinSight instruments the application, pipeline and model quality.

## Part C — Day-by-day decisions

### Day 1 (21 Oct)
- **One ingestion code path.** `incremental` gap-fills from each ticker's last stored date and falls back to the backfill window when a ticker has no data, so missed days self-heal.
- **Skip tickers that are already current** using a NYSE trading calendar (`latest_expected_session`), because an empty MarketStack response still costs quota.
- **Empty provider response = failure** when the range must contain a trading session (yfinance returns an empty frame instead of raising when Yahoo is unreachable — found during testing).
- **Retry policy:** exponential backoff with full jitter; only network errors, 429 and 5xx are retried; quota errors are never retried.
- **Idempotency:** unique indexes `(ticker, date)` and `(ticker, news_id)` + bulk upserts.
- **Structured JSON logs** with a `run_id` on every line of a run, to be surfaced in the pipeline console.
- **Bug found by the integration tests (real MongoDB):** upserts stamped `updated_at: now` in `$set`, so an identical re-upsert was counted as a revision (`modified_count` 2 instead of 1). mongomock did not expose it. Fixed by keeping only data fields in `$set` (`ingested_at` remains as first-seen lineage); the unit test now asserts `updated == 0` on an identical rerun. This is the case for keeping both unit tests (fast, mocked) and integration tests (real services) in the test plan.

### Day 2 (22 Oct)
- **pandas pinned to 2.3.3** (was 3.0.6): PySpark 4.2 warns that pandas ≥ 3 is not fully supported.
- **The export writes microsecond timestamps.** pandas defaults to nanoseconds, which Spark rejects (`PARQUET_TYPE_ILLEGAL: TIMESTAMP(NANOS)`). Caught by the Spark end-to-end unit test before it reached the real pipeline.
- **Deterministic outputs everywhere** (sorted rows, no bookkeeping fields, no timestamps in manifests, Spark part files renamed to `part-00000.parquet`, side files removed). Verified in a rehearsal: a second `dvc repro` re-runs `export` but skips `validate` and `process`; changing one `etl` param re-runs only `process`.
- **Tests gate the data.** `export` depends on `reports/unit_tests.json`, so a failing suite stops `dvc repro` before any data moves (verified with a deliberately failing test). `process` depends on `reports/data_quality.json`, so Spark never runs on data that failed a blocking check (verified with an impossible threshold).
- **Custom expectation suite instead of the `great_expectations` package**: same concepts (expectations, suite, severity, persisted report, fails on known-bad data), without a heavy dependency tree. Blocking vs warning severities are separated so a few missing days or a genuine market shock never halt the pipeline, while schema breaks, duplicates or an unadjusted split always do.
- **Model price rule**: `adj_close` per ticker only if complete, else `close`. Never mix the two in one series.
- **Calendar alignment with a broadcast join** (small calendar table shipped to executors), bounded forward-fill (≤ 2 days, flagged `is_filled`), and features that stay null until a full window exists.
- **Spark sizing**: `local[4]`, 1 GB driver, 8 shuffle partitions, adaptive execution off (stable plans and output layout). JVM start-up is reported separately from processing time.
- **Throughput is reported honestly**: on ~10k rows, Spark's fixed overhead dominates (~1.7k rows/s in the rehearsal). `finsight.etl.benchmark` shows throughput rising with volume (rehearsal: about 1.7k → 9.5k → 18.7k rows/s at 1×/10×/50×). This is the argument for Spark at scale, and the reason it is not "faster than pandas" at 10k rows.
- **Run records** (`pipeline_runs`, `pipeline: "dvc"`, one per stage) are written best-effort: if MongoDB is down, the stage still runs and only a warning is logged.
- **Docker base pinned to `python:3.11-slim-bookworm`** for the `openjdk-17-jre-headless` package (newer Debian releases dropped Java 17). The pipeline image runs Spark from Day 7 (scheduler), and the image is rebuilt and checked then.
- **Real-data finding: zero prices from MarketStack.** The first real `dvc repro` stopped at `validate`: 3 stored bars had a price of 0, which produced 4 infinite log returns. The gate did its job, since without it Spark and the LSTM would have trained on infinities. Fixes:
  - (a) a single boundary sanitiser (`make_price_bar`) used by every price provider: it rejects close ≤ 0, nulls adj_close ≤ 0, and repairs non-positive open/high/low and negative volume;
  - (b) `ingestion repair [--dry-run]` re-fetches each invalid stored bar from the unmetered fallback, storing raw close with `adj_close = None` so two providers' adjustment factors never mix; if nothing comes back it deletes the bar, and the ETL forward-fills the gap. Repairs are recorded as `ingestion` runs with `mode = "repair"`;
  - (c) `prices.positive_values` now lists the offending rows plus the fix command, and non-positive prices are masked before return calculations, so one bad bar is reported once, not three times.

### Day 3 (22 Oct, ahead of schedule)
- **One global LSTM, per-window MinMax scaling, direct 14-step output, residual to the last price.** About 139 windows per ticker is too few for 10 separate models; per-window scaling (fitted on the 100 input days only) removes the baseline's look-ahead leakage and puts all tickers on one scale. The zero-initialised residual head makes an untrained model exactly the naive forecast, so training only moves away from naive where the data supports it.
- **Purged chronological split** per ticker: train | 14-window gap | val (15) | 14-window gap | test (15). No training target overlaps a validation or test target. The selected configuration is refit on train+val for the early-stopped number of epochs; test is touched only by `evaluate_lstm`.
- **Selection by validation MAPE** (scale-free across tickers), not pooled RMSE, which the most expensive stocks would dominate.
- **Honest baselines**: naive (last price) and drift. Skill = 1 − MAPE_model / MAPE_naive. Beating naive on daily prices over 14 days is hard; the comparison is reported, never hidden, and the Day 5 promotion gate uses it.
- **Lineage beyond autolog**, as tags on every run: git commit + `git_dirty_code` (DVC-generated paths excluded), DVC hash of `data/processed/prices`, export snapshot SHA-256, seed, library versions, platform. Also logged: the dataset (`mlflow.log_input`), `params.yaml`, `dvc.lock`, the data-quality report, per-configuration learning curves (child runs), the selection table, and the test verdict (`beats_naive` tag). `models/lstm/run.json` plus `reports/lstm_train.json` link the Git commit to the MLflow run ID.
- **Bug found by tests**: `git status --porcelain` output was stripped, cutting the first character of the first path, so runs would have been tagged dirty even when the code was clean.
- **Packaged as MLflow "models from code"** (`pyfunc_entry.py`) instead of a pickled object: loading never unpickles code, and weights load with `torch.load(weights_only=True)`. Preprocessing lives inside the model (same `scale_rows` as training), so callers send raw prices: no training/serving skew. The signature is enforced by MLflow, so malformed requests are rejected before reaching the network.
- **Determinism verified**: a forced retrain (`dvc repro -f train_lstm`) reproduced identical test metrics. This holds for the same thread count on the same machine; across CPUs, floating-point results can differ in the last digits.
- **MLproject uses `conda_env: environment.yml`**: the exact environment file the developer uses (which pins `requirements/*.txt`), so dev and test environments are identical by construction. Entry-point parameters are named `*_override` because MLflow parameters are immutable and the code logs the resolved values itself.
- **MLflow server**: own Dockerfile pinned to the client version (3.17.2); SQLite backend plus `--serve-artifacts` (clients never need a shared filesystem); `--allowed-hosts` allows only localhost and the compose service name (MLflow 3's DNS-rebinding protection); bound to 127.0.0.1.
- **CPU-only PyTorch** from PyTorch's CPU wheel index (`requirements/ml.txt`). The PyPI build pulls about 2+ GB of CUDA libraries that this machine cannot use.
- **pyarrow pinned to 25.0.1** (mlflow 3.17 requires < 26).
- **MLflow server memory (found on the real laptop, Day 3).** The container was OOM-killed (exit 137) at 512 MB. MLflow 3.17's server starts, besides the web worker, a job runner and 7 `huey` consumer processes for GenAI evaluation/trace jobs, about 1.7 GB in total. Fixed with `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` (measured ~510 MB steady, 514 MB peak while a training run uploads artifacts) and `mem_limit: 768m`. The memory budget in PROJECT_PLAN.md was updated (≈ 6.1 GB peak).
