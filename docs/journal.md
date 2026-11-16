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
