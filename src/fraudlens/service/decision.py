"""The scoring hot path: one transaction in, one recorded decision out.

This module has no web-framework imports, so the whole path can be tested with in-memory
stores. The steps, for one request:

1. Claim the idempotency key. A key that already exists means a retry: return the stored
   decision and stop.
2. Take the card's lock, then read the card's history (steps 3 to 8 of the plan run under
   it, so two requests on one card never interleave their read and append).
3. Compute features with ``compute_features``, the single feature definition.
4. Score, calibrate and apply the two thresholds.
5. For review and block only, compute contributions and word the reason codes.
6. Insert the audit row. If the insert fails the request fails: no decision leaves the
   service without a record.
7. Append the transaction to the card's history.

Failure policy:

* Redis unavailable, slow or the card lock busy: score from the transaction alone
  (``degraded``), and send amounts above a limit to review at least. Small payments keep
  flowing; large ones get a human.
* Postgres unavailable: raise :class:`AuditUnavailable`; the caller retries with the
  same ``transaction_id``.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from fraudlens.common.logging import mask_card_id
from fraudlens.common.types import HistoryEntry, Transaction
from fraudlens.explain.reason_codes import (
    MAX_REASONS,
    ReasonCode,
    Templates,
    explain,
    load_templates,
)
from fraudlens.features.definitions import (
    FEATURE_NAMES,
    FeatureVector,
    compute_features,
    transaction_only_features,
)
from fraudlens.features.encoding import FEATURE_TYPES, encode_vector
from fraudlens.features.records import history_entry
from fraudlens.training.bundle import ModelBundle, logistic

logger = logging.getLogger(__name__)

DECISIONS: tuple[str, str, str] = ("approve", "review", "block")
REVIEW = 1

PENDING = "pending"
"""Value of an idempotency key whose first request is still being processed."""

HISTORY_UNAVAILABLE = "R00_HISTORY_UNAVAILABLE"
"""Id of the rule reason given when the degraded limit, not the model, forced a review."""

DUPLICATE_WAIT_S = 1.0
"""How long a retry waits for the first request to finish before answering 409."""


class StoreUnavailable(Exception):  # noqa: N818 - a state, not a programming error
    """Redis failed, timed out, or the card's lock could not be taken in time."""


class AuditUnavailable(Exception):  # noqa: N818
    """The audit store could not record the decision."""


class StillProcessing(Exception):  # noqa: N818
    """A request with this ``transaction_id`` is still being decided."""


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One decision as the audit table stores it.

    Attributes:
        transaction_id: Primary key and idempotency key.
        card_id: HMAC token of the card.
        event_ts: Transaction time, whole Unix seconds (UTC).
        amount: Dollars.
        category: Normalised merchant category.
        features: Every model feature, as ``compute_features`` returned it.
        margin: Model output in log-odds.
        score: Calibrated fraud probability, the ``risk_score``.
        decision: ``approve``, ``review`` or ``block``.
        t_review: Review threshold in force, on the calibrated probability.
        t_block: Block threshold in force.
        model_version: Bundle version that scored it.
        reason_codes: Reasons for a review or block; empty for an approval.
        degraded: True when the card's history could not be read.
        latency_ms: Milliseconds from the start of the decision to the audit write.
        created_at: When the row was written; set by the database.
    """

    transaction_id: str
    card_id: str
    event_ts: int
    amount: float
    category: str
    features: FeatureVector
    margin: float
    score: float
    decision: str
    t_review: float
    t_block: float
    model_version: str
    reason_codes: tuple[ReasonCode, ...]
    degraded: bool
    latency_ms: int
    created_at: datetime | None = None

    def response(self) -> dict[str, Any]:
        """The scoring endpoint's response body for this decision (section 7.2)."""
        return {
            "transaction_id": self.transaction_id,
            "risk_score": self.score,
            "decision": self.decision,
            "model_version": self.model_version,
            "degraded": self.degraded,
            "reason_codes": [
                {"id": r.id, "text": r.text, "share": r.share} for r in self.reason_codes
            ],
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class Outcome:
    """The response body, and whether it is a replay of an earlier decision."""

    body: dict[str, Any]
    replayed: bool


class FeatureStore(Protocol):
    """Card history and idempotency keys. Every method raises ``StoreUnavailable``."""

    async def claim(self, transaction_id: str) -> str | None:
        """Claim the key; ``None`` if this call claimed it, else its current value."""

    async def peek(self, transaction_id: str) -> str | None:
        """Current value of the key, or ``None`` if it does not exist."""

    async def complete(self, transaction_id: str, response: str) -> None:
        """Store the response under the claimed key, for later replays."""

    async def release(self, transaction_id: str) -> None:
        """Drop the claim, so a retry is decided afresh."""

    def card_lock(self, card_id: str) -> AbstractAsyncContextManager[None]:
        """Hold the card's lock for the duration of the block."""

    async def history(self, card_id: str) -> list[HistoryEntry]:
        """The card's history, sorted by ``(event_ts, transaction_id)``."""

    async def append(
        self, card_id: str, entry: HistoryEntry, history: Sequence[HistoryEntry]
    ) -> None:
        """Add ``entry`` and trim to the history window; ``history`` is what was read."""


class AuditLog(Protocol):
    """The decision table. Every method raises ``AuditUnavailable``."""

    async def insert_decision(self, record: AuditRecord) -> bool:
        """Insert, opening an alert if flagged; ``False`` if the id was already recorded."""

    async def get_decision(self, transaction_id: str) -> AuditRecord | None:
        """The recorded decision, or ``None``."""


class Observer(Protocol):
    """Receives per-stage timings and outcomes, for metrics."""

    def stage(self, name: str, seconds: float) -> None:
        """One stage of one request took ``seconds``."""

    def decided(self, decision: str, *, degraded: bool, replayed: bool, score: float) -> None:
        """One request finished with this outcome."""


class _NoObserver:
    def stage(self, name: str, seconds: float) -> None:
        pass

    def decided(self, decision: str, *, degraded: bool, replayed: bool, score: float) -> None:
        pass


@dataclass
class DecisionEngine:
    """Decides transactions with one bundle, one feature store and one audit log.

    Attributes:
        bundle: The verified model release.
        store: Card history and idempotency keys.
        audit: The decision table.
        degraded_review_amount: Dollars; without history, larger amounts go to review.
        observer: Receives timings and outcomes.
        templates: Reason-code wording.
        clock: Monotonic clock in seconds.
    """

    bundle: ModelBundle
    store: FeatureStore
    audit: AuditLog
    degraded_review_amount: float
    observer: Observer = field(default_factory=_NoObserver)
    templates: Templates = field(default_factory=load_templates)
    clock: Callable[[], float] = time.perf_counter

    async def score(self, txn: Transaction) -> Outcome:
        """Decide one transaction, or return the decision already recorded for it.

        Raises:
            AuditUnavailable: the decision could not be recorded; nothing was decided.
            StillProcessing: an earlier request with this id has not finished yet.
        """
        started = self.clock()
        claimed = False
        try:
            with self._timed("redis_claim"):
                existing = await self.store.claim(txn.transaction_id)
        except StoreUnavailable:
            existing = None
        else:
            if existing is not None:
                return self._observe(await self._duplicate(txn.transaction_id, existing))
            claimed = True

        try:
            outcome = await self._decide(txn, started, use_history=claimed)
        except BaseException:
            if claimed:
                await self._release(txn.transaction_id)
            raise
        if claimed:
            try:
                await self.store.complete(txn.transaction_id, json.dumps(outcome.body))
            except StoreUnavailable:
                logger.warning("idempotency key not completed", exc_info=True)
        return self._observe(outcome)

    async def _decide(self, txn: Transaction, started: float, *, use_history: bool) -> Outcome:
        async with AsyncExitStack() as held:
            history: list[HistoryEntry] | None = None
            if use_history:
                try:
                    with self._timed("redis_lock"):
                        await held.enter_async_context(self.store.card_lock(txn.card_id))
                    with self._timed("redis_read"):
                        history = await self.store.history(txn.card_id)
                except StoreUnavailable:
                    logger.warning(
                        "card history unavailable, scoring degraded",
                        extra={"card": mask_card_id(txn.card_id)},
                    )
            return await self._finish(txn, history, started)
        raise AssertionError("unreachable")  # pragma: no cover - keeps mypy's flow analysis

    async def _finish(
        self, txn: Transaction, history: list[HistoryEntry] | None, started: float
    ) -> Outcome:
        """Score, record and remember one transaction. ``history`` is ``None`` if degraded."""
        degraded = history is None
        with self._timed("features"):
            features = (
                transaction_only_features(txn)
                if history is None
                else compute_features(txn, history)
            )
            X = encode_vector(features)
        with self._timed("model"):
            margin = self.bundle.margin(X)
            score = float(self.bundle.calibrator(logistic(margin))[0])
            action = int(self.bundle.decide(np.array([score]))[0])
        forced = degraded and txn.amount > self.degraded_review_amount and action < REVIEW
        if forced:
            action = REVIEW

        reasons: tuple[ReasonCode, ...] = ()
        if action >= REVIEW:
            with self._timed("explanation"):
                explanation = explain(
                    self.bundle.booster, X, [features], FEATURE_NAMES, FEATURE_TYPES, self.templates
                )[0]
                reasons = explanation.reasons
            if forced:
                reasons = (self._limit_reason(txn.amount), *reasons)[:MAX_REASONS]

        record = AuditRecord(
            transaction_id=txn.transaction_id,
            card_id=txn.card_id,
            event_ts=txn.event_ts,
            amount=txn.amount,
            category=txn.category,
            features=features,
            margin=float(margin[0]),
            score=score,
            decision=DECISIONS[action],
            t_review=self.bundle.thresholds.t_review,
            t_block=self.bundle.thresholds.t_block,
            model_version=self.bundle.version,
            reason_codes=reasons,
            degraded=degraded,
            latency_ms=round((self.clock() - started) * 1000),
        )
        with self._timed("database"):
            inserted = await self.audit.insert_decision(record)
        if not inserted:
            # Recorded by an earlier request whose idempotency key has since expired.
            return await self._recorded(txn.transaction_id)

        if history is not None:
            try:
                with self._timed("redis_write"):
                    await self.store.append(txn.card_id, history_entry(txn), history)
            except StoreUnavailable:
                logger.warning(
                    "history not updated", extra={"card": mask_card_id(txn.card_id)}, exc_info=True
                )
        return Outcome(record.response(), replayed=False)

    def _limit_reason(self, amount: float) -> ReasonCode:
        """The rule reason for a review the degraded limit forced, not the model.

        It has no contribution and no share: it is a rule, not part of the model's margin.
        """
        return ReasonCode(
            HISTORY_UNAVAILABLE,
            f"Card history unavailable, and the amount of ${amount:,.2f} is over the "
            f"${self.degraded_review_amount:,.0f} review limit",
            0.0,
            0.0,
        )

    async def _duplicate(self, transaction_id: str, value: str | None) -> Outcome:
        """Answer a retry: wait briefly for an in-flight first request, then replay it."""
        deadline = self.clock() + DUPLICATE_WAIT_S
        while value == PENDING and self.clock() < deadline:
            await asyncio.sleep(0.005)
            try:
                value = await self.store.peek(transaction_id)
            except StoreUnavailable:
                value = None
        if value is not None and value != PENDING:
            return Outcome(json.loads(value), replayed=True)
        return await self._recorded(transaction_id)

    async def _recorded(self, transaction_id: str) -> Outcome:
        record = await self.audit.get_decision(transaction_id)
        if record is None:
            raise StillProcessing(transaction_id)
        return Outcome(record.response(), replayed=True)

    async def _release(self, transaction_id: str) -> None:
        try:
            await self.store.release(transaction_id)
        except StoreUnavailable:
            logger.warning("idempotency claim not released", exc_info=True)

    def _observe(self, outcome: Outcome) -> Outcome:
        body = outcome.body
        self.observer.decided(
            body["decision"],
            degraded=bool(body["degraded"]),
            replayed=outcome.replayed,
            score=float(body["risk_score"]),
        )
        return outcome

    @contextmanager
    def _timed(self, stage: str) -> Iterator[None]:
        start = self.clock()
        try:
            yield
        finally:
            self.observer.stage(stage, self.clock() - start)
