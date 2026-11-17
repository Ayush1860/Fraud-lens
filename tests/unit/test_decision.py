"""The hot path, with in-memory stores: idempotency, degraded mode, failing closed."""

import asyncio
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from fraudlens.common.types import Transaction
from fraudlens.features.definitions import HISTORY_FEATURES
from fraudlens.service.decision import (
    PENDING,
    AuditUnavailable,
    DecisionEngine,
    StillProcessing,
)
from fraudlens.training.bundle import ModelBundle
from service_fakes import MemoryAudit, MemoryStore, RecordingObserver

pytestmark = pytest.mark.anyio

CONTRACT_KEYS = {
    "transaction_id",
    "risk_score",
    "decision",
    "model_version",
    "degraded",
    "reason_codes",
    "latency_ms",
}

TXN = Transaction(
    transaction_id="t0001",
    card_id="c_0123456789abcdef",
    event_ts=1_560_000_000,
    amount=42.10,
    category="grocery_pos",
    merchant_id="m_007",
    merchant_lat=40.1,
    merchant_long=-74.2,
    home_lat=40.3,
    home_long=-74.0,
    city_pop=12_000.0,
    cardholder_age=47.0,
)


@pytest.fixture(scope="module")
def bundle(toy_bundle_dir: Path) -> ModelBundle:
    return ModelBundle.load(toy_bundle_dir)


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def audit() -> MemoryAudit:
    return MemoryAudit()


@pytest.fixture
def observer() -> RecordingObserver:
    return RecordingObserver()


@pytest.fixture
def engine(
    bundle: ModelBundle, store: MemoryStore, audit: MemoryAudit, observer: RecordingObserver
) -> DecisionEngine:
    return DecisionEngine(
        bundle=bundle,
        store=store,
        audit=audit,
        degraded_review_amount=500.0,
        observer=observer,
    )


def txn(i: int, **changes: object) -> Transaction:
    fields: dict[str, object] = {"event_ts": TXN.event_ts + 60 * i, **changes}
    return replace(TXN, transaction_id=f"t{i:04d}", **fields)  # type: ignore[arg-type]


async def test_a_new_transaction_is_recorded_and_remembered(
    engine: DecisionEngine, store: MemoryStore, audit: MemoryAudit, observer: RecordingObserver
) -> None:
    outcome = await engine.score(TXN)

    assert set(outcome.body) == CONTRACT_KEYS
    assert outcome.replayed is False
    assert outcome.body["degraded"] is False
    assert outcome.body["decision"] in {"approve", "review", "block"}
    assert audit.decisions[TXN.transaction_id].response() == outcome.body
    assert [e.transaction_id for e in store.histories[TXN.card_id]] == [TXN.transaction_id]
    assert json.loads(store.keys[TXN.transaction_id]) == outcome.body
    assert {"redis_claim", "redis_lock", "redis_read", "features", "model", "database"} <= set(
        observer.stages
    )
    assert observer.outcomes == [(outcome.body["decision"], False, False)]


async def test_history_feeds_the_next_transaction_on_the_card(
    engine: DecisionEngine, audit: MemoryAudit
) -> None:
    for i in range(3):
        await engine.score(txn(i))

    third = audit.decisions["t0002"].features
    assert third["n_prior_30d"] == 2
    assert third["txn_count_1h"] == 2


async def test_a_retry_replays_the_decision_without_touching_history(
    engine: DecisionEngine, store: MemoryStore, audit: MemoryAudit, observer: RecordingObserver
) -> None:
    first = await engine.score(TXN)
    second = await engine.score(TXN)

    assert second.replayed is True
    assert second.body == first.body
    assert len(audit.decisions) == 1
    assert len(store.histories[TXN.card_id]) == 1
    assert observer.outcomes[-1][2] is True


async def test_a_retry_waits_for_the_first_request_to_finish(
    engine: DecisionEngine, store: MemoryStore
) -> None:
    body = {
        "transaction_id": TXN.transaction_id,
        "decision": "approve",
        "degraded": False,
        "risk_score": 0.01,
    }
    store.keys[TXN.transaction_id] = PENDING

    async def finish_later() -> None:
        await asyncio.sleep(0.05)
        store.keys[TXN.transaction_id] = json.dumps(body)

    task = asyncio.create_task(finish_later())
    outcome = await engine.score(TXN)
    await task

    assert outcome.replayed is True
    assert outcome.body == body


async def test_a_retry_of_a_stuck_request_is_refused(
    engine: DecisionEngine, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fraudlens.service.decision.DUPLICATE_WAIT_S", 0.02)
    store.keys[TXN.transaction_id] = PENDING

    with pytest.raises(StillProcessing):
        await engine.score(TXN)


async def test_a_retry_whose_redis_key_expired_is_answered_from_the_audit_log(
    engine: DecisionEngine, store: MemoryStore, audit: MemoryAudit
) -> None:
    first = await engine.score(TXN)
    store.keys.clear()

    again = await engine.score(TXN)

    assert again.replayed is True
    assert again.body == first.body
    assert len(store.histories[TXN.card_id]) == 1


async def test_redis_down_scores_from_the_transaction_alone(
    engine: DecisionEngine, store: MemoryStore, audit: MemoryAudit
) -> None:
    store.down = True

    small = await engine.score(txn(1, amount=20.0))
    large = await engine.score(txn(2, amount=900.0))

    assert small.body["degraded"] is True
    assert large.body["degraded"] is True
    assert large.body["decision"] in {"review", "block"}
    features = audit.decisions["t0002"].features
    assert all(math.isnan(float(features[name])) for name in HISTORY_FEATURES)
    assert all("nan" not in reason["text"].lower() for reason in large.body["reason_codes"])


async def test_degraded_retries_are_answered_from_the_audit_log(
    engine: DecisionEngine, store: MemoryStore, audit: MemoryAudit
) -> None:
    store.down = True
    first = await engine.score(TXN)
    second = await engine.score(TXN)

    assert second.replayed is True
    assert second.body == first.body
    assert len(audit.decisions) == 1


@pytest.mark.parametrize("failing", ["card_lock", "history"])
async def test_a_failure_after_the_claim_degrades_and_keeps_idempotency(
    engine: DecisionEngine, store: MemoryStore, failing: str
) -> None:
    store.failing.add(failing)

    outcome = await engine.score(TXN)

    assert outcome.body["degraded"] is True
    assert json.loads(store.keys[TXN.transaction_id]) == outcome.body
    assert store.histories[TXN.card_id] == []


async def test_a_busy_card_lock_degrades(engine: DecisionEngine, store: MemoryStore) -> None:
    store.lock_wait_s = 0.01
    await store.locks[TXN.card_id].acquire()

    outcome = await engine.score(TXN)

    assert outcome.body["degraded"] is True


async def test_postgres_down_fails_closed_and_frees_the_id(
    engine: DecisionEngine, store: MemoryStore, audit: MemoryAudit
) -> None:
    audit.down = True
    with pytest.raises(AuditUnavailable):
        await engine.score(TXN)

    assert TXN.transaction_id not in store.keys
    assert store.histories[TXN.card_id] == []

    audit.down = False
    outcome = await engine.score(TXN)
    assert outcome.replayed is False


async def test_postgres_and_redis_down_fails_closed(
    engine: DecisionEngine, store: MemoryStore, audit: MemoryAudit
) -> None:
    store.down = audit.down = True

    with pytest.raises(AuditUnavailable):
        await engine.score(TXN)


async def test_a_failed_history_append_still_answers(
    engine: DecisionEngine, store: MemoryStore, audit: MemoryAudit
) -> None:
    store.failing.add("append")

    outcome = await engine.score(TXN)

    assert outcome.body["degraded"] is False
    assert TXN.transaction_id in audit.decisions
    assert store.histories[TXN.card_id] == []


async def test_failures_to_complete_or_release_are_not_fatal(
    engine: DecisionEngine, store: MemoryStore, audit: MemoryAudit
) -> None:
    store.failing.add("complete")
    assert (await engine.score(txn(1))).replayed is False

    store.failing = {"release"}
    audit.down = True
    with pytest.raises(AuditUnavailable):
        await engine.score(txn(2))


async def test_flagged_transactions_carry_worded_reasons(
    engine: DecisionEngine, audit: MemoryAudit, observer: RecordingObserver
) -> None:
    flagged = None
    for i, amount in enumerate((300.0, 900.0, 2_500.0, 6_000.0, 12_000.0)):
        outcome = await engine.score(txn(10 + i, amount=amount, event_ts=1_560_045_600 + i))
        if outcome.body["decision"] != "approve":
            flagged = outcome
            break
    assert flagged is not None, "the toy model flagged none of the large night purchases"

    reasons = flagged.body["reason_codes"]
    assert reasons
    assert all(0.0 < r["share"] <= 1.0 for r in reasons)
    assert sum(r["share"] for r in reasons) <= 1.0 + 1e-9
    stored = audit.decisions[flagged.body["transaction_id"]]
    assert [r.id for r in stored.reason_codes] == [r["id"] for r in reasons]
    assert "explanation" in observer.stages


async def test_a_review_forced_by_the_degraded_limit_says_so(
    engine: DecisionEngine, store: MemoryStore, bundle: ModelBundle
) -> None:
    store.down = True
    outcome = None
    for i, amount in enumerate((600.0, 700.0, 800.0, 950.0)):
        candidate = await engine.score(txn(40 + i, amount=amount))
        model_score = candidate.body["risk_score"]
        if model_score < bundle.thresholds.t_review:  # the model alone would approve
            outcome = candidate
            break
    assert outcome is not None, "the toy model flagged every probe on its own"

    assert outcome.body["decision"] == "review"
    first = outcome.body["reason_codes"][0]
    assert first["id"] == "R00_HISTORY_UNAVAILABLE"
    assert first["share"] == 0.0
    assert "$500" in first["text"]


async def test_a_degraded_model_flag_carries_no_rule_reason(
    engine: DecisionEngine, store: MemoryStore
) -> None:
    store.down = True

    small = await engine.score(txn(50, amount=20.0))

    assert all(r["id"] != "R00_HISTORY_UNAVAILABLE" for r in small.body["reason_codes"])
