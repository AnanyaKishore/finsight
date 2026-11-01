# FinSight developer shortcuts. Every target is a thin wrapper around a command
# documented in README.md, so nothing here is "magic".
# Load .env (if present) so targets can use the same variables as compose.
-include .env
export
.PHONY: help install lint format test test-int up down mongo-shell ingest-plan backfill incremental status build repro dag eda benchmark train plots sentiment portfolio feature-experiment mlflow-ui register serve api-dev rollback-lstm frontend web-dev scheduler run-now retrain-now monitoring dashboards drill-start drill-stop rules-test e2e load-test test-report secrets-key secrets-encrypt secrets-decrypt secrets-check backup backup-models verify-backup pre-commit audit repro-drill preflight

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-20s %s\n",$$1,$$2}'

install:         ## install dev deps + package in editable mode
	pip install -r requirements/dev.txt && pip install -e .

lint:            ## static checks (PEP 8 via ruff, formatting via black)
	ruff check src tests && black --check src tests

format:          ## auto-format
	ruff check --fix src tests && black src tests

test:            ## unit tests (no external services needed)
	pytest tests/unit -q --cov=finsight --cov-report=term-missing

test-int:        ## integration tests (needs: docker compose up -d mongo)
	pytest tests/integration -q -m integration

up:              ## start the long-running services
	docker compose up -d

down:            ## stop everything (data volumes are kept)
	docker compose down

build:           ## build all images
	docker compose --profile jobs build

mongo-shell:     ## open mongosh inside the mongo container
	docker compose exec mongo mongosh -u $$MONGO_ROOT_USER -p $$MONGO_ROOT_PASSWORD --authenticationDatabase admin finsight

ingest-plan:     ## show how many API calls a backfill would cost (no calls made)
	python -m finsight.ingestion.cli backfill --dry-run

backfill:        ## one-time historical load (prices + news)
	python -m finsight.ingestion.cli backfill

incremental:     ## gap-fill since the last stored date (what the daily job runs)
	python -m finsight.ingestion.cli incremental

status:          ## row counts, date coverage, API quota usage, last runs
	python -m finsight.ingestion.cli status

repro:           ## run the DVC pipeline (tests -> export -> validate -> Spark ETL)
	dvc repro

dag:             ## show the pipeline DAG and current metrics
	dvc dag && dvc metrics show

eda:             ## (re)build and execute the EDA notebook, writing figures to reports/figures
	python scripts/build_eda_notebook.py && jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace

benchmark:       ## Spark ETL throughput at 1x/10x/50x data volume -> reports/etl_benchmark.json
	python -m finsight.etl.benchmark --factors 1 10 50

train:           ## (re)train + evaluate the LSTM through DVC (only if inputs changed)
	dvc repro evaluate_lstm

plots:           ## render all DVC plots -> dvc_plots/index.html
	dvc plots show

sentiment:       ## fetch (once), evaluate + calibrate, then score news sentiment through DVC
	dvc repro score_sentiment

portfolio:       ## optimise today's portfolio + walk-forward backtest through DVC
	dvc repro optimize_portfolio

feature-experiment:  ## LSTM: price-only vs price+volatility input (MLflow only, ~3-5 min)
	python -m finsight.models.lstm.feature_experiment

mlflow-ui:       ## start the MLflow server (UI at http://localhost:5000)
	docker compose up -d mlflow

register:        ## register new model versions + run the promotion gate (through DVC)
	dvc repro register

serve:           ## build + start model servers and the backend (API docs: http://localhost:8000/docs)
	docker compose up -d --build lstm-serve sentiment-serve backend

frontend:        ## build + start the web portal at http://localhost:8080
	docker compose up -d --build frontend

web-dev:         ## serve frontend/ locally without Docker (needs the backend on :8000)
	python -m http.server 8080 --bind 127.0.0.1 -d frontend

api-dev:         ## run the backend locally with auto-reload-free uvicorn (needs serving up)
	python -m finsight.api

rollback-lstm:   ## roll the LSTM back to the previous production version (REASON="...")
	python -m finsight.registry.rollback --model lstm --reason "$(REASON)"

scheduler:       ## start the scheduler (daily runs at 06:30 IST + console triggers) (Day 7)
	docker compose up -d scheduler

run-now:         ## start a daily pipeline run now (same as "Run now" in the console)
	curl -s -X POST localhost:8001/run -H "X-Finsight-Token: $$SCHEDULER_TOKEN" -H 'Content-Type: application/json' -d '{"kind":"daily","reason":"make run-now"}'; echo

retrain-now:     ## start a retrain now (same as "Retrain now" in the console)
	curl -s -X POST localhost:8001/run -H "X-Finsight-Token: $$SCHEDULER_TOKEN" -H 'Content-Type: application/json' -d '{"kind":"retrain","reason":"make retrain-now"}'; echo

monitoring:      ## start Prometheus, Alertmanager, Grafana and the exporters (Day 8)
	docker compose up -d prometheus alertmanager grafana node-exporter cadvisor mongodb-exporter nginx-exporter

dashboards:      ## regenerate the Grafana dashboards from scripts/build_dashboards.py
	python scripts/build_dashboards.py

drill-start:     ## drift drill: the monitor sees drifted data for 30 min (stored data untouched)
	docker compose exec scheduler python -m finsight.monitoring.drill start --minutes 30

drill-stop:      ## end the drift drill now
	docker compose exec scheduler python -m finsight.monitoring.drill stop

rules-test:      ## check the Prometheus config and run the alert-rule unit tests (in the container)
	docker compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
	docker compose exec -w /etc/prometheus/tests prometheus promtool test rules finsight_test.yml

# ---- Day 9: testing, security, code quality ---------------------------------
e2e:             ## end-to-end smoke tests against the running stack (docker compose up -d)
	pytest tests/e2e -q -m e2e

load-test:       ## 10 users for 60 s against the API -> reports/load_test.json
	python scripts/load_test.py

test-report:     ## run every suite -> docs/TEST_REPORT.md + docs/TEST_CASES.md
	python scripts/test_report.py

secrets-key:     ## create the encryption key (once; ~/.config/finsight/secret.key, never overwritten)
	python -m finsight.security.secrets init-key

secrets-encrypt: ## .env -> secrets/.env.enc (safe to commit; re-run after editing .env)
	python -m finsight.security.secrets encrypt

secrets-decrypt: ## secrets/.env.enc -> .env (new machine; refuses to overwrite)
	python -m finsight.security.secrets decrypt

secrets-check:   ## key location/permissions, .env permissions, git, encrypted copy in sync
	python -m finsight.security.secrets check

backup:          ## encrypted snapshot of MongoDB + reports + config -> backups/
	python -m finsight.security.backup create

backup-models:   ## same, plus models/ (larger)
	python -m finsight.security.backup create --include-models

verify-backup:   ## decrypt the newest snapshot and check every file against its manifest
	python -m finsight.security.backup verify "$$(ls -t backups/*.tar.enc | head -1)"

pre-commit:      ## run every pre-commit hook on all files
	pre-commit run --all-files

audit:           ## known vulnerabilities in the installed Python packages (needs internet)
	pip-audit -r requirements/base.txt

# ---- Day 10: reproducibility ------------------------------------------------
repro-drill:     ## fresh clone -> dvc pull -> mlflow run -> same metrics? -> docs/REPRODUCIBILITY.md
	python scripts/repro_drill.py

# ---- Day 11: demo readiness -------------------------------------------------
preflight:       ## is the running stack ready to demo? (containers, models, drill, alerts, next run, AC-11)
	python scripts/preflight.py
