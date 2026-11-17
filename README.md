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
| 1 | Data checks, PII handling, time splits, the single feature definition | Done |
| 2 | Imbalance experiments, tuning, calibration, thresholds, model bundle | In progress |
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
make test    # unit, property and parity tests with the 85% coverage gate
```

Then build the data:

```bash
make data       # download both datasets, validate them, strip PII -> data/interim/
make features   # build the feature tables -> data/processed/
make test-data  # re-run the parity check on 5,000 real transactions
```

Then train and evaluate:

```bash
make train            # imbalance study, tuning, final fit, ULB benchmark (DEVICE=cuda for a GPU)
make evaluate         # the one-time test-set evaluation, then docs/results/report.md
```

Training is heavy on a laptop CPU. `notebooks/colab_training.ipynb` runs the same steps on
a free Google Colab GPU and packages the results to bring back.

Other targets: `make lint`, `make typecheck`, `make format`, `make down`, `make help`.
The remaining pipeline targets (`serve`, `replay`, `load`, `report`) are stubs that fail
with the phase they arrive in.

Without Make, run the underlying commands directly, for example `uv sync --locked`,
`uv run python -m fraudlens.data.download` and
`uv run pytest -m "not integration and not data" --cov`.

## Data

| Dataset | Role | Source |
| --- | --- | --- |
| Sparkov simulated card transactions | Primary: readable fields and two years of per-card history | [kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
| ULB credit card fraud | Benchmark table only: its features are anonymous PCA components | [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

Both are public and download without Kaggle credentials. The first download records a
SHA-256 per file in `data/raw/SHA256SUMS.json`; later runs verify against it and fail if a
raw file has changed.

The Sparkov names, streets and card numbers are synthetic, and are handled as if real:

- Names, street, city, state, zip code, job and date of birth are dropped at ingestion.
- The card number becomes `card_id`, a keyed HMAC-SHA256 token. A plain hash is not
  enough, because the card-number space is small enough to enumerate.
- Age at the transaction is derived from the date of birth before it is dropped.
- `gender` is kept for the offline fairness check only and is never a model feature.

`notebooks/01_eda.ipynb` shows what the training partition looks like, including the
finding that merchant coordinates are noise around the cardholder's home.

## Features

`src/fraudlens/features/definitions.py` holds the only definition of every feature:
`compute_features(txn, history)`, which the scoring service will call per request.
`src/fraudlens/features/batch.py` computes the same values for a whole table at once, for
training. A parity test fails on any row where the two disagree.

- Windows are closed on the left: a window ending at a transaction holds earlier
  transactions only, and never the transaction itself.
- Transactions in the same second are ordered by transaction id, in training and serving.
- Window sums use integer cents, so the batch builder matches the definition exactly
  rather than to a tolerance. Only the five features that pass through `log` or
  trigonometric functions are compared to 1e-9.
- Splits are by time: training, a 7-day gap, then validation cut 60/40 into a tuning
  slice and a calibration slice. The public test partition is not read until the final
  evaluation.

## Training

`python -m fraudlens.training <step>` runs one step; `configs/training.yaml` and
`configs/costs.yaml` hold every setting, with the cost assumptions written beside them.

- **Imbalance study.** Seven arms (logistic regression, then XGBoost with no weighting,
  full and square-root `scale_pos_weight`, `max_delta_step`, undersampling with prior
  correction, and SMOTE) over five seeds, with 95% day-block bootstrap intervals.
  Resampling touches training rows only.
- **Tuning.** Optuna over three prequential folds, maximising mean average precision.
- **Release.** The training slice fits the model, the early validation slice stops it,
  and the late validation slice fits the isotonic calibrator and chooses the review and
  block thresholds by expected cost within the analysts' daily capacity.
- **Bundle.** `models/<version>/` holds the booster, calibrator, thresholds, threshold
  curve, drift reference histograms, metrics and provenance. `manifest.json` lists the
  SHA-256 of every file, and loading fails if any file is missing, extra or changed.
- **Final test.** `fraudlens.training.evaluate.final_test` is the only reader of the test
  partition and refuses to score a bundle twice. It adds a fairness check of
  false-positive rates by gender and age band; the model never sees gender.

## Configuration

Settings are read from `FRAUDLENS_*` environment variables, or from a `.env` file in the
repository root. `configs/.env.example` lists every variable with local-only placeholder
values. Secrets have no defaults: the process refuses to start without them.

## Layout

```text
configs/          .env.example, training.yaml, costs.yaml
data/             gitignored: raw, interim and processed data
models/           gitignored model bundles
docs/             design notes, journal, performance reports, results/ from training
notebooks/        01_eda.ipynb, colab_training.ipynb (no logic lives in notebooks)
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
