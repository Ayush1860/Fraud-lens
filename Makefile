.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test test-data up down \
        data features train evaluate serve replay load report all

help: ## List the available targets
	@grep -E '^[a-z -]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install dependencies, git hooks and a local .env
	uv sync --locked
	uv run pre-commit install
	@test -f .env || (cp configs/.env.example .env && echo "created .env from configs/.env.example")

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Apply formatting and safe lint fixes
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## mypy --strict on src/
	uv run mypy

test: ## Unit, property and parity tests with the coverage gate
	uv run pytest -m "not integration and not data" --cov --cov-report=term-missing

test-data: ## Checks that need the real datasets (run after `make data features`)
	uv run pytest -m data

up: ## Start Postgres and Redis and wait until both are healthy
	docker compose up -d --wait

down: ## Stop the local stores (data volumes are kept)
	docker compose down

# --- Pipeline and service targets. Each is a stub until its phase lands, and fails
# --- loudly so that nobody mistakes a stub for a successful run.

data: ## Download the datasets, validate them, strip PII, write data/interim/
	uv run python -m fraudlens.data.download
	uv run python -m fraudlens.data.ingest

features: ## Build the feature tables in data/processed/
	uv run python -m fraudlens.features.batch

train evaluate: ## Phase 2: experiments, tuning, calibration, evaluation
	@echo "make $@: not implemented yet (phase 2: modelling and experiments)"; exit 1

serve replay: ## Phase 4: scoring service and replay client
	@echo "make $@: not implemented yet (phase 4: scoring service)"; exit 1

load report: ## Phase 6: load test and generated report
	@echo "make $@: not implemented yet (phase 6: monitoring, load test and documents)"; exit 1

all: data features train evaluate report ## Rebuild data, model and report from raw files
