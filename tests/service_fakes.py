"""In-memory stand-ins for Redis and Postgres, for fast tests of the service.

They follow the same contracts as ``RedisFeatureStore`` and ``Repository``, including the
failure exceptions, and can be switched to failing on demand.
"""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from fraudlens.common.types import HistoryEntry
from fraudlens.features.definitions import HISTORY_MAX_ENTRIES, HISTORY_WINDOW_S
from fraudlens.service.decision import (
    PENDING,
    AuditRecord,
    AuditUnavailable,
    StoreUnavailable,
)
from fraudlens.service.repository import (
    AlertRow,
    CardTransaction,
    DispositionRow,
    RecentDecision,
    decode_cursor,
    encode_cursor,
)


class MemoryStore:
    """A feature store in a dict. ``down`` fails every call; ``failing`` fails named ones."""

    def __init__(self) -> None:
        self.keys: dict[str, str] = {}
        self.histories: dict[str, list[HistoryEntry]] = defaultdict(list)
        self.locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.cache: dict[str, Any] = {}
        self.down = False
        self.failing: set[str] = set()
        self.lock_wait_s = 1.0

    def _check(self, name: str) -> None:
        if self.down or name in self.failing:
            raise StoreUnavailable(name)

    async def ping(self) -> bool:
        return not self.down

    async def claim(self, transaction_id: str) -> str | None:
        self._check("claim")
        previous = self.keys.get(transaction_id)
        if previous is None:
            self.keys[transaction_id] = PENDING
        return previous

    async def peek(self, transaction_id: str) -> str | None:
        self._check("peek")
        return self.keys.get(transaction_id)

    async def complete(self, transaction_id: str, response: str) -> None:
        self._check("complete")
        self.keys[transaction_id] = response

    async def release(self, transaction_id: str) -> None:
        self._check("release")
        self.keys.pop(transaction_id, None)

    @asynccontextmanager
    async def card_lock(self, card_id: str) -> AsyncIterator[None]:
        self._check("card_lock")
        lock = self.locks[card_id]
        try:
            await asyncio.wait_for(lock.acquire(), self.lock_wait_s)
        except TimeoutError as error:
            raise StoreUnavailable("card lock busy") from error
        try:
            yield
        finally:
            lock.release()

    async def history(self, card_id: str) -> list[HistoryEntry]:
        self._check("history")
        return sorted(self.histories[card_id], key=lambda e: (e.event_ts, e.transaction_id))

    async def append(
        self, card_id: str, entry: HistoryEntry, history: Sequence[HistoryEntry]
    ) -> None:
        self._check("append")
        stored = [*self.histories[card_id], entry]
        newest = max(e.event_ts for e in stored)
        kept = [e for e in stored if e.event_ts >= newest - HISTORY_WINDOW_S]
        self.histories[card_id] = kept[-HISTORY_MAX_ENTRIES:]

    async def cache_get(self, key: str) -> Any | None:
        return None if self.down else self.cache.get(key)

    async def cache_set(self, key: str, value: Any, ttl_s: int) -> None:
        if not self.down:
            self.cache[key] = value


class MemoryAudit:
    """The audit tables in lists. ``down`` makes every call raise ``AuditUnavailable``."""

    def __init__(self) -> None:
        self.decisions: dict[str, AuditRecord] = {}
        self.alerts: list[dict[str, Any]] = []
        self.dispositions: list[DispositionRow] = []
        self.versions: dict[str, dict[str, Any]] = {}
        self.down = False

    def _check(self) -> None:
        if self.down:
            raise AuditUnavailable("down")

    async def ping(self) -> bool:
        return not self.down

    async def insert_decision(self, record: AuditRecord) -> bool:
        self._check()
        if record.transaction_id in self.decisions:
            return False
        now = datetime.now(UTC)
        self.decisions[record.transaction_id] = replace(record, created_at=now)
        if record.decision != "approve":
            self.alerts.append(
                {
                    "id": len(self.alerts) + 1,
                    "transaction_id": record.transaction_id,
                    "priority": record.score * record.amount,
                    "status": "open",
                    "created_at": now,
                }
            )
        return True

    async def get_decision(self, transaction_id: str) -> AuditRecord | None:
        self._check()
        return self.decisions.get(transaction_id)

    async def list_alerts(
        self, *, status: str | None, decision: str | None, limit: int, cursor: str | None
    ) -> tuple[list[AlertRow], str | None]:
        self._check()
        rows = []
        for alert in sorted(self.alerts, key=lambda a: (-a["priority"], -a["id"])):
            record = self.decisions[alert["transaction_id"]]
            if status is not None and alert["status"] != status:
                continue
            if decision is not None and record.decision != decision:
                continue
            rows.append(
                AlertRow(
                    **alert,
                    decision=record.decision,
                    risk_score=record.score,
                    amount=record.amount,
                    category=record.category,
                    event_time=datetime.fromtimestamp(record.event_ts, tz=UTC),
                    top_reason=record.reason_codes[0].text if record.reason_codes else None,
                )
            )
        if cursor is not None:
            priority, alert_id = decode_cursor(cursor)
            rows = [r for r in rows if (r.priority, r.id) < (priority, alert_id)]
        page = rows[:limit]
        more = len(rows) > limit
        return page, encode_cursor(page[-1].priority, page[-1].id) if more else None

    async def add_disposition(
        self, alert_id: int, *, label: str, analyst: str, note: str | None
    ) -> DispositionRow | None:
        self._check()
        alert = next((a for a in self.alerts if a["id"] == alert_id), None)
        if alert is None:
            return None
        row = DispositionRow(
            len(self.dispositions) + 1, alert_id, label, analyst, note, datetime.now(UTC)
        )
        self.dispositions.append(row)
        alert["status"] = "closed"
        return row

    async def record_model_version(
        self, version: str, *, manifest_hash: str, metrics: Mapping[str, Any]
    ) -> None:
        self._check()
        self.versions.setdefault(version, {"manifest_hash": manifest_hash, **metrics})

    async def recent_decisions(self, limit: int) -> list[RecentDecision]:
        self._check()
        newest = sorted(self.decisions.values(), key=lambda r: r.created_at or 0, reverse=True)
        return [
            RecentDecision(r.score, r.decision, r.degraded, r.latency_ms) for r in newest[:limit]
        ]

    async def card_history(
        self, card_id: str, *, until: datetime, limit: int
    ) -> list[CardTransaction]:
        self._check()
        cutoff = until.timestamp()
        rows = sorted(
            (r for r in self.decisions.values() if r.card_id == card_id and r.event_ts <= cutoff),
            key=lambda r: (r.event_ts, r.transaction_id),
            reverse=True,
        )
        return [
            CardTransaction(
                r.transaction_id,
                datetime.fromtimestamp(r.event_ts, tz=UTC),
                r.amount,
                r.category,
                r.decision,
                r.score,
                r.degraded,
            )
            for r in rows[:limit]
        ]

    async def disposition_counts(self) -> dict[str, int]:
        self._check()
        latest: dict[int, str] = {}
        for row in self.dispositions:
            latest[row.alert_id] = row.label
        counts = {"fraud": 0, "genuine": 0}
        for label in latest.values():
            counts[label] += 1
        return counts

    async def recent_features(self, limit: int) -> list[tuple[dict[str, float | int | str], str]]:
        self._check()
        newest = sorted(self.decisions.values(), key=lambda r: r.created_at or 0, reverse=True)
        return [(dict(r.features), r.decision) for r in newest[:limit]]


class RecordingObserver:
    """Collects what the decision engine reports."""

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.outcomes: list[tuple[str, bool, bool]] = []

    def stage(self, name: str, seconds: float) -> None:
        assert seconds >= 0
        self.stages.append(name)

    def decided(self, decision: str, *, degraded: bool, replayed: bool, score: float) -> None:
        assert 0.0 <= score <= 1.0
        self.outcomes.append((decision, degraded, replayed))
