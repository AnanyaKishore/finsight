# FinSight developer shortcuts. Every target is a thin wrapper around a command
# documented in README.md, so nothing here is "magic".
# Load .env (if present) so targets can use the same variables as compose.
-include .env
export
.PHONY: help install lint format test test-int up down mongo-shell ingest-plan backfill incremental status build

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-14s %s\n",$$1,$$2}'

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
