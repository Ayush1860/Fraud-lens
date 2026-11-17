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
| 2 | Imbalance experiments, tuning, calibration, thresholds, model bundle | Done |
| 3 | Contributions and reason codes | Done |
| 4 | Scoring service: Redis feature store, Postgres audit trail, idempotency | Done |
| 5 | React analyst console | Done |
| 6 | Monitoring, load test, documents | Next |
| 7 | AWS demo from Terraform | Planned |

## Results

The model report, with the imbalance study, the one-time test evaluation, a fairness
check and the ULB benchmark, is generated from training artefacts:
[docs/results/report.md](docs/results/report.md). Every number in it comes from a JSON
file beside it; none is typed by hand.

## Stack

Python 3.12, uv, FastAPI, Pydantic v2, SQLAlchemy 2 (async) with Alembic, Redis, Postgres,
XGBoost, Optuna, pytest, ruff, mypy `--strict`. Console: React, TypeScript, Vite.
Infrastructure: Docker Compose locally, Terraform for the AWS demo.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/), Docker with Compose, GNU Make, and
Node.js 22.12 or newer for the console outside Docker.

```bash
make setup   # install locked dependencies, git hooks, and create .env from configs/.env.example
make up      # start Postgres, Redis and the service, and wait until all are healthy
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

Then serve and replay:

```bash
make up                # console on http://127.0.0.1:8080, service on :8000, both stores
make replay            # 5,000 test-period transactions at 50 per second, history warmed
make test-integration  # idempotency, outages, concurrency and parity against real stores
make test-web          # console lint, types and component tests
make e2e               # browser smoke test against the running stack
```

The service loads the newest bundle under `models/`, which is gitignored: train one, or
copy a released bundle there. Without one the service refuses to start.
`make serve` runs the service outside Docker instead.

Other targets: `make lint`, `make typecheck`, `make format`, `make down`, `make help`.
The remaining targets (`load`, `report`) are stubs that fail with the phase they arrive in.

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

## Explanations

Every flagged transaction gets up to five reason codes in plain language, worded with its
own values: for example "Amount is 20.9 times this card's 30-day average of $46".

- **Contributions** are XGBoost's own exact TreeSHAP values
  (`Booster.predict(pred_contribs=True)`), in log-odds, plus a bias column; each row sums
  to the model's margin. The `shap` library is never imported by the service.
- **Groups.** Correlated features are summed into six groups with stable ids
  (`R01_AMOUNT_VS_PATTERN` to `R06_CARDHOLDER_CONTEXT`), defined with their wording in
  `src/fraudlens/explain/templates.yaml`. The cardholder-context group is never shown.
- **Share** is each reason's part of the total upward push. Contributions are log-odds,
  so they do not add up in probability units.
- **Independent check.** For XGBoost models, `shap.TreeExplainer` hands path-dependent
  SHAP straight back to XGBoost, so comparing the two would test XGBoost against itself.
  The tests instead compare against a brute-force Shapley calculation written from
  scratch over the dumped trees, including the categorical splits.
- `python -m fraudlens.explain.checks --bundle models/<version>` writes additivity,
  stability, global ranking and monthly explanation drift to
  `docs/results/explainability.json`. `notebooks/02_shap_global.ipynb` plots them.

## Scoring service

`POST /v1/transactions/score` takes one transaction and returns a calibrated risk score,
**approve**, **review** or **block**, and reason codes for anything flagged. OpenAPI docs
are at `/docs` while the service runs; every endpoint except the health checks needs the
`X-API-Key` header.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/transactions/score` | Score and decide; idempotent on `transaction_id` |
| `GET /v1/transactions/{id}` | The stored decision and the feature vector it was made from |
| `GET /v1/transactions/{id}/explanation` | Every feature's contribution, recomputed from stored features |
| `GET /v1/alerts` | Open alerts by score times amount, with cursor paging |
| `POST /v1/alerts/{id}/disposition` | An analyst's label; append-only |
| `GET /v1/model`, `GET /v1/monitoring/summary` | The active release; recent decision mix and latency |
| `GET /healthz`, `/readyz`, `/metrics` | Liveness, readiness, Prometheus metrics |

The hot path (`src/fraudlens/service/decision.py`, free of web-framework code):

1. Claim the `transaction_id` in Redis with `SET NX`. A retry gets the original decision
   back with `Idempotent-Replay: true`, and the card's history is not touched.
2. Under a per-card Redis lock, read the card's history, compute features with the same
   `compute_features` training used, score, calibrate and apply the thresholds.
3. Word reason codes for a review or block only.
4. Insert the decision, with its feature vector, into Postgres. If that fails the request
   fails with 503: no decision leaves without a record.
5. Append the transaction to the card's history, trimmed to 30 days and 500 entries.

Failure behaviour: if Redis is down or slow, the transaction is scored on its own fields,
marked `degraded: true`, and any amount above a configured limit goes to review. If
Postgres is down, the service answers 503 and the caller retries with the same id. A
bundle whose files do not match its manifest stops the service from starting.

The audit tables are append-only, enforced by database triggers, and a stored decision
reproduces its score bit for bit from its feature vector. `docs/design.md` explains the
choices and the accepted limits.

The replay client (`python -m fraudlens.tools.replay`) reads the public test file,
turns card numbers into HMAC tokens before anything is sent, and posts transactions in
time order at a fixed rate or a multiple of real time. `--inject-drift shopping_net=1.5`
inflates one category's amounts part-way through, to trigger the drift alarm on demand.

## Analyst console

`web/` is a React and TypeScript console that answers one question fast: why was this
transaction flagged, and was the model right?

| Screen | What it shows |
| --- | --- |
| Alert queue | Open alerts by dollars at risk, with the leading reason in words; polls every 5 seconds |
| Case view | The reasons in words, the factor pulling the other way, a waterfall from the model's baseline to this transaction, the card's last 20 transactions, and the verdict form |
| Threshold lab | Sliders over the validation curve; every pair is priced by the same function that chose the released thresholds |
| Model and monitoring | The release's validation metrics, the live decision mix against validation, a log-scale score histogram, latency, and realised precision from analyst labels |
| Experiments | The imbalance study's seven arms with bootstrap intervals and precision-recall curves |

- **Types from the API.** `web/src/api/schema.ts` is generated by `openapi-typescript`
  from the contract snapshot `tests/contract/openapi.json`, and CI fails if it is stale.
  A field renamed in the service breaks the console's type check, not its users.
- **One path.** Every call goes to the relative path `/api`. nginx (or the Vite dev
  server) forwards it to the service and adds the API key, so the browser never holds it.
- **Decisions are never colour alone:** each has an icon and a word. Every screen has
  loading, empty and error states, and a permanent banner says the data is simulated.

## Configuration

Settings are read from `FRAUDLENS_*` environment variables, or from a `.env` file in the
repository root. `configs/.env.example` lists every variable with local-only placeholder
values, and the service's optional timeouts and limits with their defaults. Secrets have
no defaults: the process refuses to start without them.

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
web/              React and TypeScript analyst console, its nginx image and smoke test
infra/            Terraform for the AWS demo
```

## Engineering rules

- Time-ordered splits only. Features use strictly earlier transactions.
- One feature definition, shared by training and serving, with a parity test between them.
- Raw card numbers, names and addresses are never logged. Card ids are keyed HMAC tokens,
  and logs show at most their first six characters.
- The test set is read by exactly one function, once.
- Every public function has type hints and a docstring stating units and time semantics.
