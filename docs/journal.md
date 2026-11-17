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
- **Bug:** on Colab the download step reported that `fraudTrain.csv` was missing. Colab's
  Kaggle cache ignores the `output_dir` passed to `kagglehub` and returns its own
  directory instead. The download step now copies any missing file from the directory
  the client returns. The row-count and schema checks in ingestion still run on whatever
  arrives, so a different file from the cache would be caught there.

## 2026-11-17 — Phase 2 results from Colab

Read from `docs/results/report.md` and the bundle `2026-11-17.1`.

- **Check:** the bundle was trained from commit `089e8dc`, the branch tip at the time, and
  its raw-data SHA-256 values match the laptop's files exactly. On the laptop the bundle's
  manifest verifies, and the 1,000-row reproduction sample scores identically: zero
  difference in margin and in calibrated score between a Colab GPU fit and a laptop CPU
  load.
- **Finding:** every XGBoost arm lands within about 0.01 average precision of the others,
  with overlapping intervals, far above logistic regression. Weighting and resampling bought
  no ranking gain, which matches the Elor and Averbuch-Elor finding that balancing does not
  help strong classifiers. Tuning picked A3 (square-root weighting) by a hair.
- **Finding:** the tuned A3 model was already close to calibrated, so isotonic calibration
  changed the test Brier score only slightly. Full weighting (A2) and resampling (A5) had
  the worst raw Brier scores of the XGBoost arms, as the plan expected.
- **Finding:** test average precision is below validation, a fall to expect a few months
  later in time. The intervals on the test set do not overlap the validation ones.
- **Finding:** precision at a 1% daily budget is capped near the fraud rate divided by
  the budget (about 0.39 here), so the reported precision is close to the best possible.
- **Finding:** ordering the review queue by score times amount raises dollar recall at
  both budgets while catching slightly fewer frauds by count. That is the trade the plan
  predicted, now measured.
- **Finding:** the false-positive rate is higher for women than for men, peaks for ages 25-39
  and is lowest for 70 and over. The model never sees gender, so the gap comes through
  correlated features such
  as category and amount. It is a finding to discuss in the model card, not to hide.
- **Finding:** the model leans on amount, 24-hour spend, category and night-time, the same
  signals the EDA showed. No identifier-like feature exists to leak, and none ranks high.

## 2026-11-17 — Phase 3 explainability

- **Surprise:** the plan's "library agreement" check, comparing XGBoost's contributions with
  `shap.TreeExplainer(feature_perturbation="tree_path_dependent")`, cannot fail. For an
  XGBoost model in that mode, shap calls `Booster.predict(pred_contribs=True)` itself; the
  two agreed to exactly 0.0, which is what gave it away. Shap's own implementation cannot
  read XGBoost's categorical splits at all.
- **Decision:** the independent check is a brute-force Shapley calculation written from
  scratch in `tests/model/shapley_oracle.py`. It walks the trees from the model's JSON dump,
  evaluates each tree's path-dependent expectation for every subset of the features the
  tree uses, and applies the Shapley formula. It is exponential in features per tree, so
  it runs on a small depth-3 model with categorical splits. It catches the mistakes the
  plan wanted caught: column order and category encoding.
- **Bug (in the oracle):** its first margins disagreed with XGBoost by up to 0.3. The
  categorical direction was right; the difference was precision. XGBoost compares feature
  values as 32-bit floats, so a value just below a threshold in 64 bits can round onto it.
  Comparing in 32 bits brought agreement to about 2e-7.
- **Decision:** reasons are grouped into six groups with stable ids. The cardholder-context
  group (age, city population, history length) is computed but never worded, because it
  says nothing an analyst can act on.
- **Decision:** the time-and-category template states the UTC hour and the category, not
  "outside this card's usual hours" as the plan's example does. The model has no feature
  for a card's usual hours, so that wording would claim something the model never saw.
- **Review:** all 20 proposed golden cases were reviewed and approved on 2026-11-17. Their
  leading reason ids are frozen in `tests/model/golden_cases.json`. Nine lead with the
  amount group, five with purchase pace, four with time and category, and one each with
  travel and novelty.

## 2026-11-18 — Phase 4 scoring service

- **Decision:** with Redis down, history features are sent to the model as missing (NaN),
  not as the zeros an empty history gives. Zero would tell the model "first purchase on
  this card", which is a strong signal in its own right. The rule lives in
  `definitions.py` as `transaction_only_features`, beside the one feature definition.
- **Bug (found by the degraded-path test):** wording a reason for a degraded decision
  crashed, because the pace template turned a missing count into an integer. Pace and
  novelty are now not worded when history is unknown; the golden cases are unaffected.
- **Decision:** idempotency has two layers. Redis answers ordinary retries quickly; the
  Postgres primary key catches retries after the Redis key expires or while Redis is down.
- **Decision:** append-only is enforced by triggers rather than grants. Locally a single
  database user owns the tables, and an owner can always UPDATE its own table; a trigger
  stops it anyway.
- **Decision:** the replay client reads the raw test CSV and tokenises cards itself, as
  the plan asks, instead of reading `data/processed/test.parquet`, which stays reserved
  for the one-time evaluation.
- **Testing without Docker:** this laptop has no Docker, so the integration tests ran
  locally against an embedded Postgres 16 and an in-process fakeredis, and run in CI
  against Postgres 18 and Redis 8 containers. Fakeredis's TCP server mode was unusable
  here: it dropped connections around Lua script loads and rejected valid lock releases.
- **Check:** 2,000 synthetic transactions replayed through the service produce stored
  features equal to the batch builder's, and 1,000 stored decisions re-score bit for bit.

## 2026-11-18 — Phase 5 analyst console

- **Decision:** TypeScript is pinned at 5.9.3, not the newest release (7.0). The plan's type
  generator, `openapi-typescript`, declares TypeScript 5 as a peer, and `typescript-eslint`
  supports nothing past 6.0. Every other console dependency is at its latest version.
- **Decision:** the threshold lab prices pairs on the server with `pair_costs`, the function
  that chose the released thresholds, instead of repeating the cost formula in TypeScript.
  A test checks that the released pair costs exactly what training recorded.
- **Finding:** the curve has 21 points, not 400: isotonic calibration produces 21 distinct
  scores, so those are the only operating points that exist. The sliders step through them.
- **Finding:** the bundle's score reference histogram is degenerate (about 99% of validation
  scores are 0, so the decile edges collapse). The monitoring page compares the live
  decision mix with the validation decision mix instead, and plots the live score histogram
  on a log scale, where the tail near 1 is visible.
- **Bug (found by building the queue):** a review forced by the degraded-mode amount limit
  carried no reason codes when the model itself had nothing positive to say, so an analyst
  saw a flag with no explanation. It now leads with a rule reason,
  `R00_HISTORY_UNAVAILABLE`, which has no contribution or share because it is not part of
  the model's output.
- **Check:** with the released bundle served locally, a verdict recorded in the browser was
  read back from Postgres, and the Playwright smoke test passed against the local stack in
  Edge. CI runs the same test in Chromium against the Docker Compose stack.

## 2026-11-18 — Phase 6 monitoring, load test and documents

- **Finding:** the plan's drift demonstration, shopping_net amounts times 1.5, moves the
  overall amount PSI from 0.004 to at most 0.016, far below any alarm. Within the category
  the same shift measured 0.34, against at most 0.12 for clean window pairs. The drift
  report therefore compares amount within each category, newest window against the one
  before. Measured on validation rows; the test file stays unread.
- **Finding:** day of week shows a PSI of 2.09 with no drift at all, because 10,000
  decisions cover about a day. Calendar features are reported but never alarm. This is the
  clearest example of where PSI misleads: a window shorter than the feature's cycle.
- **Decision:** score drift is measured as the decision mix against the validation mix.
  The bundle's score histogram is degenerate (about 99% of validation scores are near 0, so
  the decile edges collapse), and the release cannot be changed after the fact.
- **Bug (found while looking at the alarm in the console):** categories too small to judge
  sorted to the top of the segment table, because a missing PSI was keyed as -1.
- **Surprise:** after a restart with an empty Redis, history features alarm loudly
  (`n_prior_30d` measured 8.3): every card looks new. It is a real signal, written into the
  runbook, and the replay client gained `--warm-only` to refill history without traffic.
- **Decision:** load tests run on a GitHub Actions runner, against the Compose stack, with
  a model of the released model's size (398 trees, depth 6) trained on synthetic data. The
  released bundle is not in the repository, and timing depends on tree count and depth,
  not on the data. A release keeps only the trees up to its best validation round, so the
  load-test bundle is refitted at full size with the same calibration and threshold code.
- **Decision:** `make report` copies the release's result documents into
  `docs/results/release/`, then writes the README block, the model card and the validation
  report from committed files only. A test regenerates them and fails on any difference.
