.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test test-data test-integration up down \
        data features train evaluate migrate serve replay console test-web e2e load report all

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

up: ## Start Postgres, Redis and the service, and wait until all are healthy
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

DEVICE ?= cpu

train: ## Imbalance study, tuning, final fit and ULB benchmark (DEVICE=cuda on a GPU)
	uv run python -m fraudlens.training experiments --device $(DEVICE)
	uv run python -m fraudlens.training tune --device $(DEVICE)
	uv run python -m fraudlens.training release --device $(DEVICE)
	uv run python -m fraudlens.training ulb --device $(DEVICE)

evaluate: ## The one-time test-set evaluation of the newest bundle, then the report
	uv run python -m fraudlens.training evaluate

migrate: ## Apply the audit-database migrations
	uv run alembic upgrade head

serve: migrate ## Run the scoring service locally on port 8000
	uv run uvicorn fraudlens.service.app:create_app --factory --port 8000

REPLAY_ARGS ?= --limit 5000 --rate 50 --warm-days 30

replay: ## Replay the public test file through the running service
	uv run python -m fraudlens.tools.replay $(REPLAY_ARGS)

TEST_DATABASE_URL ?= postgresql+asyncpg://fraudlens:local-dev-only-change-me@localhost:5432/fraudlens
TEST_REDIS_URL ?= redis://localhost:6379/15

console: ## Run the console dev server on http://localhost:5173 (needs Node 22.12+)
	cd web && npm ci && npm run dev

test-web: ## Console lint, types and component tests
	cd web && npm ci && npm run format:check && npm run lint && npm run typecheck && npm test

e2e: ## Console smoke test against the running stack (`make up` first)
	cd web && npx playwright install chromium && npx playwright test

test-integration: ## Service tests against real stores (start them with `make up`)
	FRAUDLENS_TEST_DATABASE_URL=$(TEST_DATABASE_URL) FRAUDLENS_TEST_REDIS_URL=$(TEST_REDIS_URL) \
	uv run pytest -m integration

load report: ## Phase 6: load test and generated report
	@echo "make $@: not implemented yet (phase 6: monitoring, load test and documents)"; exit 1

all: data features train evaluate report ## Rebuild data, model and report from raw files
