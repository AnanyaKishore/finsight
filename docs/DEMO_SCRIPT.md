# FinSight — 10-minute demo script

What to click, in what order, and what to say. The times are a guide. If the evaluator
interrupts with a question, answer it and skip the matching part later. Questions and answers
for the viva are in [`VIVA_PREP.md`](VIVA_PREP.md).

---

## Before the evaluator arrives (15 minutes before)

```bash
cd ~/finsight && conda activate finsight
docker compose up -d
make preflight            # every line PASS (a WARN is fine if you can explain it)
```

`make preflight` FAILs on anything that would spoil the demo: a container down, a model
without a `@production` version, a drift drill left on, or a Prometheus target down. It
WARNs if the daily run is due within the hour (see "Timing" below), if a run is in progress,
or if memory is low.

Open these tabs in this order, so each act is one tab to the right:

| # | Tab | URL |
|---|---|---|
| 1 | Architecture diagram | `docs/diagrams/architecture.png` on GitHub, or in VS Code |
| 2 | Portal, overview | http://localhost:8080 |
| 3 | Portal, Pipeline Console | http://localhost:8080/#/pipeline |
| 4 | MLflow, experiment `finsight-lstm` | http://localhost:5000 |
| 5 | Grafana, "FinSight · Overview" dashboard | http://localhost:3000/d/finsight-overview |
| 6 | Prometheus, Alerts | http://localhost:9090/alerts |

Also open **one terminal** in `~/finsight`, with a large font. Close other apps: a retrain
or a FinBERT re-score needs the memory.

**Timing.** The daily run starts at 06:30 IST, Tuesday to Saturday, and takes several
minutes. Don't schedule a demo during it. If you must, it is still something to show: the
DAG turns blue stage by stage.

---

## Act 1 — The problem and the architecture (0:00 – 1:00) · tab 1

> "FinSight answers three questions for ten large US stocks: where has the price been,
> where may it go in the next 14 trading days, and what does the news say? It also suggests
> a portfolio. It is built as an AIOps system: everything runs on this laptop in Docker
> Compose, with no cloud."

Point at the diagram from top to bottom:
1. **Browser → frontend (nginx) → backend (FastAPI)**. The UI only knows the backend's URL.
2. **Backend → two model servers** (MLflow serve of the `@production` models) and MongoDB.
3. **Scheduler → DVC pipeline**: ingestion, tests, validation, Spark ETL, training, gate.
4. **MLflow** for tracking and the registry. The model servers follow the `@production` alias.
5. **Prometheus → Alertmanager → `/retrain`**: monitoring closes the loop.

## Act 2 — A user's journey (1:00 – 3:30) · tab 2

1. **Overview.** "Ten stocks, end-of-day prices, green and red with arrows, so colour is
   never the only cue. The status dot top right is green: every dependency is up."
2. Press **/**, type `iphone`, then press **Enter**. "Search works by name, ticker or product."
3. **Apple page → Price history → 3M.** Hover over the line. "Every chart also has a table
   view, for screen readers."
4. **14-day forecast.** Read the caption aloud. "It quotes the model's back-test error
   *next to* the naive 'price stays the same' baseline. LSTM 3.21 % against naive 3.34 %.
   I show that honestly, because beating naive on daily prices is hard."
5. **News sentiment → click the most negative bar.** "Each bar is a week of news. Clicking
   it shows the headlines that made it negative: that is the explainability for sentiment."
6. Back to the overview, **Try the sentiment model.** Paste these two lines and press **Analyse**:
   ```
   Apple beats earnings expectations
   Regulator fines the bank over fraud
   ```
   "This is a live call through the backend to the FinBERT model server."

## Act 3 — Portfolio, honestly (3:30 – 4:30) · tab 2

**Optimised portfolio card.**
> "The weights maximise the Sharpe ratio over the last 30 days. I search all 92,378
> portfolios on a 10 % grid and check against a continuous optimiser. The reality check
> underneath matters more: re-optimising every week over the past year, out of sample, the
> Sharpe was 0.05 against 0.46 for equal weights. That is a well-known result, and the UI
> says so instead of hiding it."

## Act 4 — Break something on purpose (4:30 – 5:00) · terminal, then tab 2

```bash
docker compose stop lstm-serve
```
Reload the Apple page:
> "The forecasting server is down. The status dot is amber, not red. The forecast shows the
> **last saved copy**, clearly labelled, and everything else still works. We'll see the alert
> fire in a few minutes."

## Act 5 — The pipeline (5:00 – 7:00) · tab 3

1. **DAG.** "These are the DVC stages: unit tests run **first** and every data stage depends
   on them, so DVC acts as CI. Then validation (21 checks) gates Spark, training is followed
   by evaluation, and the registry gate decides on promotion. Green succeeded, and an amber
   dot means the inputs changed since the stage last ran." Click one stage to show its details.
2. **Run history.** Filter by *failed* and open run `5ec75aa8d7f1`.
   > "The unit-test gate stopped this run on day 7, before any data changed. The log tail
   > shows the failing test. It was a test-isolation bug, fixed the same afternoon. This is
   > also why AC-11 in the test report is at 80 % until there are ten scheduled runs."
3. **Run duration and Throughput** charts. "Spark rows per second and articles scored per
   second, run by run."
4. Point at **Run now** and **Retrain now** without clicking them. "Both ask for confirmation.
   The browser never sees the scheduler token: the backend holds it."

## Act 6 — Experiments and reproducibility (7:00 – 8:15) · tab 4

1. **MLflow → `finsight-lstm` → newest parent run.** Show the child runs: "one per grid
   configuration, selected on validation MAPE."
2. Open the parent run's **Tags**: the Git commit, the DVC data hash, the export SHA-256 and
   the seed (`git_commit`, `dvc_prices_md5`, `export_prices_sha256`, `seed`). "That is the lineage beyond autolog."
3. **Models → `finsight-lstm`.** "The `@production` alias is what the server loads. The gate
   moves it only if a candidate beats the champion on the *same* test windows."
4. "And I proved reproducibility: `make repro-drill` clones the repository at the training
   commit, pulls that commit's data with DVC and retrains with `mlflow run`. Every metric came
   out identical." Show the `finsight-repro` experiment, or `docs/REPRODUCIBILITY.md`.

## Act 7 — Monitoring closes the loop (8:15 – 9:30) · tabs 5 and 6

1. **Grafana Overview.** "API latency and errors, pipeline runs, model quality and drift, and
   the host and containers. Five dashboards, generated from code."
2. **Prometheus → Alerts.** `ModelServerDown` is *pending* (or already *firing*, if 5 minutes
   have passed). "Rules have a `for:` duration, so a restart doesn't page anyone."
3. "The same path retrains the model: when returns drift in at least 30 % of the stocks, or
   live error is clearly worse than naive, Alertmanager calls the scheduler's `/retrain`.
   There is a 24-hour cooldown, so it can't loop."
4. Repair the server:
   ```bash
   docker compose start lstm-serve
   ```
   "It loads the `@production` version again. Within a minute the dot is green."

## Act 8 — Close (9:30 – 10:00)

> "There are 278 unit tests, plus 4 integration and 16 end-to-end tests. Every acceptance
> criterion passes except the scheduled-run rate, which is waiting for enough runs. API p95
> latency is 17 ms under load. Everything is versioned: Git and Git LFS for code, DVC for data
> and models, the MLflow registry for model versions. Secrets and backups are encrypted.
> With more time I'd add …"

Pick one ending that you mean: more stocks with a per-sector model; FinBERT input truncated
to 128 tokens to halve re-scoring; a proper message queue between the scheduler and the
workers.

---

## If something goes wrong during the demo

| What happens | Say / do |
|---|---|
| A card says "Could not load" | Press **Try again**. "Each section fails on its own, by design." |
| The status dot is amber before Act 4 | Click it, name the part that is down, and run `docker compose start <service>`. This *is* the degradation demo. |
| `lstm-serve` takes long to come back in Act 7 | The watcher loads the model on start-up. Check with `docker compose logs -f lstm-serve`; it normally takes under a minute. |
| The "Analyse" box is slow the first time | That is the first FinBERT call after a restart (model warm-up). The second call is fast. |
| MLflow or Grafana won't load | `docker compose restart mlflow` (or `grafana`). Show `reports/lstm_train.json` and the run ID while it restarts. |
| The laptop is slow or swapping | `docker compose stop grafana sentiment-serve` frees about 2 GB. Skip Act 7's dashboard. |
| A daily run starts mid-demo | Switch to tab 3 and narrate it: the DAG turns blue stage by stage. |

## After the demo

```bash
docker compose ps                 # everything running again?
make preflight
```
