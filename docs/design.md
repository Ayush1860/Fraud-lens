# Scoring service design

How the scoring service decides, what it does when a dependency fails, and the limits it
accepts. The code is in `src/fraudlens/service/`; the hot path is `decision.py`.

## The hot path

| Step | What happens | Where |
| --- | --- | --- |
| 1 | Check the API key; validate the body (positive amount, coordinates in range, `event_time` with a time zone and no more than 5 minutes ahead, `card_id` an HMAC token) | `state.py`, `schemas.py` |
| 2 | Claim the id: `SET idem:{transaction_id} pending NX EX 10 GET`. An existing key means a retry | `feature_store.py` |
| 3 | Take the card's lock, then read `hist:{card_id}` in one round trip | `feature_store.py` |
| 4 | `compute_features(txn, history)`, the function training used | `features/definitions.py` |
| 5 | `inplace_predict` margin, logistic, isotonic calibrator, the two thresholds | `training/bundle.py` |
| 6 | For review and block only: `pred_contribs` contributions and reason codes | `explain/` |
| 7 | Insert the decision, and an alert if flagged, in one Postgres transaction | `repository.py` |
| 8 | Append the transaction to the card's history and trim it, atomically in Lua | `feature_store.py` |
| 9 | Store the response under the idempotency key for 24 hours, release the lock, answer | `decision.py` |

`latency_ms` in the response and the audit row is measured from the start of the
decision to the audit write. Prometheus records the full request and every stage
separately.

## Idempotency

- The claim is `pending` for 10 seconds, then becomes the response body for 24 hours. A
  request that dies mid-flight frees its id within 10 seconds.
- A retry that finds `pending` polls for up to one second, then falls back to Postgres,
  then answers 409 with `Retry-After`.
- Postgres is the final guard. The decision insert is `ON CONFLICT DO NOTHING`; if the row
  already exists, the stored decision is returned and the history is not touched. This
  covers retries after the Redis key has expired and retries made while Redis was down.
- A retry returns the original body, including its original `latency_ms`, with the
  header `Idempotent-Replay: true`.

## One card, two requests

Steps 3 to 8 run under `lock:{card_id}`, a Redis lock with a 2-second lease. The second
request on a card reads the history only after the first has appended to it. The
integration test fires 30 requests on one card at once and checks that each one saw
exactly the transactions recorded before it.

A request that waits more than 150 ms for the lock scores degraded rather than queueing:
latency stays bounded and the busy card's next transaction sees the full history again.

## Failure modes

| Failure | Behaviour |
| --- | --- |
| Redis down, slow (50 ms per call), or the card lock busy | Score from the transaction alone: every feature that reads history is missing (NaN), not zero, so the model takes the missing-value branch it learned rather than reading "first purchase on this card". Set `degraded: true`. An amount above `degraded_review_amount` (default $500) goes to review at least. Pace and novelty reasons are not worded, since nothing about them was observed. |
| Postgres down or slower than 2 s | 503 with `Retry-After: 1`. The idempotency claim is released, so the retry is decided afresh. Readiness turns false. |
| Bundle fails its manifest at start-up | The lifespan handler raises and the process exits. It never serves. |
| History append fails after the audit insert | The decision stands and is answered; the card's history misses that transaction. Logged as a warning. |
| Missing optional field | Passed to the model as NaN. |

Failing open on Redis and closed on Postgres is deliberate. Without history the model is
weaker but still useful, and a person reviews anything large. Without the audit row there
is no record of why money moved or did not, which a bank cannot accept.

## Audit trail

- `decisions` stores the feature vector as JSONB, with the margin, score, thresholds and
  model version. Floats are written with Python's shortest round-trip form, and a missing
  feature as `null`. The integration test re-scores 1,000 stored rows and gets the same
  margin, score and decision bit for bit.
- `decisions`, `dispositions` and `model_versions` reject UPDATE and DELETE through
  triggers, whatever role connects. In `alerts` only `status` may change. Locally one
  database user owns everything, so the triggers, not grants, are what enforce this.
- `model_versions` gets one row per release, with the manifest hash, on first start.

## Accepted limits

- **Out-of-order events.** History is sorted by time on every read, and the feature
  definition ignores anything not strictly earlier, so a late event never leaks future
  information into a decision. But a transaction scored before a late-arriving earlier one
  is not rescored: its features miss that event. Card networks deliver authorisations in
  near real time, so this is rare.
- **Lost appends.** If the append fails after the audit insert, that transaction is
  missing from the card's history until it ages out. The alternative, rolling back the
  decision, would turn a Redis blip into a failed payment.
- **The degraded limit is a flat amount.** A per-category or per-card limit would be
  better and needs history, which is exactly what is missing in degraded mode.
- **Inference runs inline on the event loop.** One-row `inplace_predict` and
  `pred_contribs` block the loop while they run. Inline against a thread pool is
  measured in phase 6 with the load test, and the faster one is kept.

## The console's access to the API

The console calls `/api` on its own origin. nginx forwards those calls to the service and
adds the `X-API-Key` header, so the key never reaches the browser. The consequence is that
anyone who can reach the console's port can use the API through it: in Compose it binds to
loopback, and the public demo in phase 7 has to put its own access control in front.

The alert queue polls every 5 seconds. At a few hundred alerts a day that is a handful of
small requests per analyst per minute. Polling stops being enough when analysts need to
see an alert within a second or two, or when enough analysts are polling that the queue
query itself becomes load; then the service would push new alerts over server-sent events.

## The replay client

`python -m fraudlens.tools.replay` reads `data/raw/sparkov/fraudTest.csv`, validates it,
and tokenises card numbers with the HMAC key before anything is sent. It never opens
`data/processed/test.parquet`, which only the one-time evaluation reads.
`--warm-days 30` loads each card's preceding 30 days from the ingested transactions
straight into Redis, so the first replayed decisions see the same history the offline
features did.
