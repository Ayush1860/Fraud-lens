# Journal

A dated log of bugs, surprises and decisions.

## 2026-11-16 — Phase 0 scaffold

- **Decision:** secrets (`FRAUDLENS_API_KEY`, `FRAUDLENS_CARD_HMAC_KEY`) and both store URLs
  have no defaults in `Settings`. A missing value fails at start-up instead of the service
  running with a guessable key. Tests build settings from the environment with `.env` disabled.
- **Decision:** Makefile pipeline targets are stubs that exit non-zero and name their phase,
  so `make all` cannot look like a successful run before the pipeline exists.
- **Surprise:** the Postgres 18 image keeps its data under `/var/lib/postgresql/<major>/`,
  so the Compose volume mounts `/var/lib/postgresql`, not the older `/var/lib/postgresql/data`.
- **Decision:** the dev machine has no Docker, so the "both containers report healthy"
  acceptance check runs as its own CI job (`compose`) rather than being taken on trust.
- **Decision:** `astral-sh/setup-uv` publishes no moving major tag, so CI pins the exact
  release; `actions/checkout` is pinned to its major tag.

## 2026-11-17 — Phase 1 data and features

- **Surprise:** Sparkov's `unix_time` column does not match `trans_date_trans_time`. It is
  offset by about seven years, and by two different amounts across a leap day. The
  readable timestamp, read as UTC, is the one source of event time.
- **Decision:** window statistics are computed in integer cents. A first design with
  floating-point prefix sums would have matched the definition only to a tolerance, and
  worse, a card that repeats one amount has a true standard deviation of exactly zero
  while prefix-sum cancellation gives a tiny non-zero one, which turns the z-score into
  an enormous number. With integers, `n * sum(x^2) - sum(x)^2` is exact and never
  negative, and both implementations then run the same IEEE operations in the same order.
- **Decision:** parity is exact equality for every feature except the five that pass
  through `log` or trigonometric functions, where numpy and `math` may differ in the last
  bit. Those are compared to 1e-9. The tolerance is fixed in `tests/helpers.py`.
- **Decision:** the history limit (30 days, 500 entries) is applied inside
  `compute_features`, not only by the feature store, so the batch builder can mirror it
  and extra entries can never change a result.
- **Decision:** implied speed uses a one-second floor on the time gap, so two purchases
  in the same second give a large finite speed instead of infinity.
- **Decision:** features are built over train and test rows together, because the first
  30 days of the test period need history from the end of training. Features read only
  earlier transactions and never a label. Test rows are written to their own file, which
  only the final evaluation will read.
- **Decision:** ingestion also drops city, state, zip code and job. The plan lists names,
  street and date of birth; these four are quasi-identifiers that no feature uses.
- **Decision:** the datasets are fetched with `kagglehub`, Kaggle's own client, which
  downloads public datasets without credentials. No token has to exist on the machine.
- **Bug:** the first version of the reference-path test helper rebuilt 600 history rows
  with `iterrows` for every sampled transaction and took four and a half minutes.
  Converting each card's rows once and slicing the list brought the whole suite to
  under a minute.
- **Bug:** the real-data parity test first failed with a pandas error, not a mismatch:
  `transaction_id` was both the index name and a column, which `sort_values` refuses.
- **Check:** to confirm the parity tests can fail, two bugs were injected into the batch
  builder by hand: the window's old edge made open instead of closed, and the history cap
  off by one. Each was caught by a parity test, then reverted.
- **Check:** the time split reproduces the row counts of the reference paper
  (arXiv 2607.14686): its training rows equal this split's training plus gap rows, and
  its validation rows equal this split's two validation slices.
- **Finding:** merchant coordinates are uniform noise within one degree of the
  cardholder's home, and distance from home has the same distribution for fraud and
  genuine transactions. Distance features will be weak on this data. See
  `notebooks/01_eda.ipynb`.
- **Finding:** no card reaches the 500-entry history cap in Sparkov.

## 2026-11-17 — Phase 2 modelling pipeline

- **Decision:** training runs on Google Colab, because the 7-arm, 5-seed study plus 40
  Optuna trials over three folds is too slow on the laptop CPU. The notebook
  `notebooks/colab_training.ipynb` holds no logic: it clones this branch and runs
  `python -m fraudlens.training <step>`, the same tested code the laptop would run. Every
  step saves to Google Drive, and tuning stores its trials in SQLite so a disconnect
  loses at most one trial.
- **Decision:** the calibrator is stored as isotonic breakpoints and applied with
  `numpy.interp`, which reproduces `IsotonicRegression.predict` exactly. No pickle is
  loaded at serving time.
- **Decision:** the release's calibrator maps the booster's own probability (the logistic
  of its margin) to a calibrated one. For the undersampling arm this absorbs the prior
  correction, because both maps are monotone, so serving never needs to know which arm
  produced the model.
- **Decision:** ranking metrics on the test set use the uncalibrated score. Isotonic
  calibration is a step function, so it would turn distinct scores into ties and make
  the top 1% of a day partly arbitrary.
- **Decision:** threshold search tries every pair of candidate thresholds, using totals
  above each candidate so each pair costs O(1). A test checks it against a brute-force
  search, and the same totals are exported as the threshold lab's curve.
- **Decision:** SMOTE seeds only from frauds with no missing feature, because
  interpolation is undefined for a missing value, and uses SMOTE-NC so the categorical
  column never gets an invented category. The rows with missing values stay in training.
- **Decision:** bootstrap intervals for the alert metrics and the Brier score are
  computed from per-day totals. Alerts are chosen per day, so a resampled day keeps the
  alerts it had, and the result is exact without re-ranking each resample.
- **Decision:** dropped MLflow. Optuna's own storage records every trial, and
  `tuning.json` is the committed record.
- **Decision:** the ULB benchmark has no 7-day gap, because the dataset covers two days.
  Hours replace days as the bootstrap and alert-budget block.
- **Smoke run:** before handing off to Colab, the experiment and release steps ran on the
  real data with one seed and 20 trees, without reading the test set. All seven arms
  trained in under a minute, and the bundle verified and reloaded. With only 20 trees,
  plain XGBoost already reached an average precision near 0.95 on validation. That is
  the risk the plan names, "scores look near-perfect on Sparkov", arriving on time.
- **Bug:** the first Colab run failed while collecting tests, at `import matplotlib`.
  Colab exports `MPLBACKEND=module://matplotlib_inline.backend_inline` to every shell
  command, and matplotlib accepts that name only when `matplotlib-inline` is installed,
  which the project environment does not install. It passed locally only because the EDA
  notebook work had left `matplotlib-inline` in the local `.venv`. The report module now
  imports pyplot only when it draws, forcing the file-only Agg backend.
  `tests/unit/test_report_backend.py` runs the command line under Colab's exact setting.
- **Bug:** a Colab run reached the experiment step without `data/processed/trainval.parquet`.
  The data step had not finished, but `!` shell commands in a notebook do not stop it
  when they fail, so later cells ran on missing files. Every notebook step now goes
  through a `run()` helper that raises on a non-zero exit, and the data step verifies
  its output files and skips work already done, so it is safe to re-run after a reset.
