# Runbook

What to do when something goes wrong with the scoring service. Each section says how you
notice, what it means, and what to do, in that order.

## Roll back a model

**When:** a new release scores worse than the last one, or its decision mix jumps.

1. Find the previous release: `ls models/`. Releases are `YYYY-MM-DD.N` and never
   overwritten.
2. Point the service at it: set `FRAUDLENS_BUNDLE_DIR=models/<previous version>` in `.env`,
   then `docker compose up -d api`. The service verifies the bundle's manifest at start-up
   and refuses to start if any file has changed.
3. Check `GET /v1/model`: the version and manifest hash must be the previous release's.
4. Decisions already made stay as they were. Each audit row records the model version
   that made it, and `/v1/transactions/{id}/explanation` answers 409 for a decision made
   by a model that is no longer active, rather than explaining it with the wrong one.

## Drift alarm

**How you notice:** the red banner on the console's Model and monitoring page, or
`fraudlens_drift_alarms` above 0 in `/metrics`. The report is recomputed every minute.

PSI over 0.25 means a distribution has moved. It does not mean the model is wrong.

1. **Read which comparison alarmed.**
   - *Amount in a category* (newest window against the previous one): a sudden shift in
     one category. Check whether the volume or the merchant mix in that category changed.
   - *A feature against training:* traffic now looks different from what the model learned
     on. If it is a history feature (`n_prior_30d`, `txn_count_7d`, `amt_*_30d`), first check
     the next section: an empty or recently restarted Redis makes every card look new.
   - *Decision mix against validation:* the share of reviews or blocks has moved. This is
     the one that reaches analysts' workload directly.
2. **Calendar features never alarm.** Day of week and weekend are listed with their PSI but
   cannot raise an alarm: a window of 10,000 decisions covers about a day, so they always
   look drifted.
3. **Check outcomes before acting.** Realised precision of labelled alerts is on the same
   page. If precision holds, the shift is in the traffic, not in the model's quality; note
   it and keep watching. If precision falls, plan a retrain on recent data.
4. **Do not change thresholds to make an alarm go away.** Use the threshold lab to see
   what a change would cost, and record the decision in `docs/journal.md`.

## Redis outage

**How you notice:** `degraded: true` in responses, `fraudlens_degraded_responses_total`
rising, and `redis: false` on `/readyz`. Readiness stays true: the service keeps scoring.

1. Scoring continues from each transaction's own fields; amounts over
   `FRAUDLENS_DEGRADED_REVIEW_AMOUNT` (default $500) go to review with the rule reason
   `R00_HISTORY_UNAVAILABLE`. Expect more reviews.
2. Restart Redis: `docker compose up -d redis`. The data volume keeps card history.
3. If Redis comes back empty, card history rebuilds as traffic arrives, over 30 days. Until
   then history features read low, and the drift report will alarm on them. In the local
   demo, `python -m fraudlens.tools.replay --warm-days 30 --warm-only` refills it from the
   ingested transactions without scoring anything. A production system would rebuild it
   from its own transaction store.
4. Idempotency still holds while Redis is down: the Postgres primary key catches retries.

## Postgres outage

**How you notice:** scoring returns 503 with `Retry-After: 1`; `/readyz` returns 503.

1. This is by design: no decision leaves the service without an audit row. Callers retry
   with the same `transaction_id`, and the retry is decided afresh.
2. Restart Postgres: `docker compose up -d postgres`. The service reconnects on its own.
3. Nothing to replay: requests that got a 503 were never decided.

## The service will not start

1. `docker compose logs api`. A `BundleIntegrityError` names the file that fails its
   manifest: the bundle was changed or copied incompletely. Copy it again; never edit a
   bundle in place.
2. `no model bundle under models` means `models/` is empty: train one, or copy a release in.
3. A missing setting (`FRAUDLENS_API_KEY` and so on) fails at start-up by design. Copy
   `configs/.env.example` to `.env`.
