# FraudLens

Explainable real-time card-fraud scoring service.

A transaction comes in, the service reads that card's recent history from Redis, computes
velocity and spending-pattern features, scores it with XGBoost and returns **approve**,
**review** or **block**. Every flag carries plain-language reason codes backed by exact SHAP
values, and every decision is written to an append-only Postgres audit trail before the
response leaves.

> **Simulated data.** The primary dataset (Sparkov) is generated, not real card traffic.
> Scores on it demonstrate method and engineering, not real-world accuracy.

## Status

The project is built in phases, one pull request each. No metric appears in this README
until it is generated from an artefact by `make report`.

| Phase | Scope | State |
| --- | --- | --- |
| 0 | Scaffold: tooling, settings, logging, Compose stores, CI | Done |
| 1 | Data checks, PII handling, time splits, the single feature definition | Next |
| 2 | Imbalance experiments, tuning, calibration, thresholds, model bundle | Planned |
| 3 | Contributions and reason codes | Planned |
| 4 | Scoring service: Redis feature store, Postgres audit trail, idempotency | Planned |
| 5 | React analyst console | Planned |
| 6 | Monitoring, load test, documents | Planned |
| 7 | AWS demo from Terraform | Planned |

## Stack

Python 3.12, uv, FastAPI, Pydantic v2, SQLAlchemy 2 (async) with Alembic, Redis, Postgres,
XGBoost, Optuna, pytest, ruff, mypy `--strict`. Console: React, TypeScript, Vite.
Infrastructure: Docker Compose locally, Terraform for the AWS demo.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/), Docker with Compose, and GNU Make.

```bash
make setup   # install locked dependencies, git hooks, and create .env from configs/.env.example
make up      # start Postgres and Redis and wait until both are healthy
make test    # unit tests with the 85% coverage gate
```

Other targets: `make lint`, `make typecheck`, `make format`, `make down`, `make help`.
The pipeline targets (`data`, `features`, `train`, `evaluate`, `serve`, `replay`, `load`,
`report`, `all`) exist as stubs that fail with the phase they arrive in.

Without Make, run the underlying commands directly, for example
`uv sync --locked` and `uv run pytest -m "not integration" --cov`.

## Configuration

Settings are read from `FRAUDLENS_*` environment variables, or from a `.env` file in the
repository root. `configs/.env.example` lists every variable with local-only placeholder
values. Secrets have no defaults: the process refuses to start without them.

## Layout

```text
configs/          .env.example (training and cost configs arrive in phase 2)
data/             gitignored: raw, interim and processed data
models/           gitignored model bundles
docs/             design notes, journal, performance reports
src/fraudlens/
  common/         settings, logging, hashing, shared types
  data/           download, schema checks, PII handling, time splits
  features/       the single feature definition and its batch builder
  training/       experiments, metrics, tuning, calibration, thresholds, bundling
  explain/        contributions and reason codes
  service/        FastAPI app, feature store, audit repository, decision logic
  monitoring/     population stability index and the drift job
  tools/          replay client and load test
migrations/       Alembic
tests/            unit, property, parity, model, integration, contract
web/              React and TypeScript analyst console
infra/            Terraform for the AWS demo
```

## Engineering rules

- Time-ordered splits only. Features use strictly earlier transactions.
- One feature definition, shared by training and serving, with a parity test between them.
- Raw card numbers, names and addresses are never logged. Card ids are keyed HMAC tokens,
  and logs show at most their first six characters.
- The test set is read by exactly one function, once.
- Every public function has type hints and a docstring stating units and time semantics.
