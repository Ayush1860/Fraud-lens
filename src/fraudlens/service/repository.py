"""Postgres: the audit trail of decisions, the alert queue, dispositions and releases.

``decisions``, ``dispositions`` and ``model_versions`` are append-only: database triggers
reject any UPDATE or DELETE, whatever role connects. In ``alerts`` only ``status`` may
change. The schema itself is created by the Alembic migrations in ``migrations/``; the
table objects here describe it for queries.

Feature vectors are stored as JSONB. JSON has no NaN, so a missing feature is stored as
``null`` and read back as NaN. Floats are written with Python's shortest round-trip
representation, so a stored vector reproduces the model input bit for bit.
"""

import asyncio
import base64
import json
import math
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from fraudlens.explain.reason_codes import ReasonCode
from fraudlens.service.decision import AuditRecord, AuditUnavailable

T = TypeVar("T")

metadata = sa.MetaData()

decisions = sa.Table(
    "decisions",
    metadata,
    sa.Column("transaction_id", sa.Text, primary_key=True),
    sa.Column("card_id", sa.Text, nullable=False),
    sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("amount", sa.Float, nullable=False),
    sa.Column("category", sa.Text, nullable=False),
    sa.Column("features", JSONB, nullable=False),
    sa.Column("margin", sa.Float, nullable=False),
    sa.Column("score", sa.Float, nullable=False),
    sa.Column("decision", sa.Text, nullable=False),
    sa.Column("t_review", sa.Float, nullable=False),
    sa.Column("t_block", sa.Float, nullable=False),
    sa.Column("model_version", sa.Text, nullable=False),
    sa.Column("reason_codes", JSONB, nullable=False),
    sa.Column("degraded", sa.Boolean, nullable=False),
    sa.Column("latency_ms", sa.Integer, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

alerts = sa.Table(
    "alerts",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True),
    sa.Column("transaction_id", sa.Text, nullable=False, unique=True),
    sa.Column("priority", sa.Float, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

dispositions = sa.Table(
    "dispositions",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True),
    sa.Column("alert_id", sa.BigInteger, nullable=False),
    sa.Column("label", sa.Text, nullable=False),
    sa.Column("analyst", sa.Text, nullable=False),
    sa.Column("note", sa.Text),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

model_versions = sa.Table(
    "model_versions",
    metadata,
    sa.Column("version", sa.Text, primary_key=True),
    sa.Column("manifest_hash", sa.Text, nullable=False),
    sa.Column("metrics", JSONB, nullable=False),
    sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
)


def features_to_json(features: Mapping[str, float | int | str]) -> dict[str, Any]:
    """Make a feature vector JSON-safe: NaN becomes ``None``.

    Raises:
        ValueError: on an infinite value, which no feature should ever produce.
    """
    out: dict[str, Any] = {}
    for name, value in features.items():
        if isinstance(value, float) and not math.isfinite(value):
            if not math.isnan(value):
                raise ValueError(f"feature {name} is infinite")
            out[name] = None
        else:
            out[name] = value
    return out


def features_from_json(stored: Mapping[str, Any]) -> dict[str, float | int | str]:
    """Inverse of :func:`features_to_json`: ``None`` becomes NaN."""
    return {name: math.nan if value is None else value for name, value in stored.items()}


def encode_cursor(priority: float, alert_id: int) -> str:
    """Opaque keyset cursor for the alert queue."""
    return base64.urlsafe_b64encode(json.dumps([priority, alert_id]).encode()).decode()


def decode_cursor(cursor: str) -> tuple[float, int]:
    """Inverse of :func:`encode_cursor`.

    Raises:
        ValueError: if the cursor was not produced by :func:`encode_cursor`.
    """
    try:
        priority, alert_id = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return float(priority), int(alert_id)
    except (ValueError, TypeError) as error:
        raise ValueError("invalid cursor") from error


@dataclass(frozen=True, slots=True)
class AlertRow:
    """One alert with the decision it was raised for."""

    id: int
    transaction_id: str
    priority: float
    status: str
    created_at: datetime
    decision: str
    risk_score: float
    amount: float
    category: str
    event_time: datetime


@dataclass(frozen=True, slots=True)
class DispositionRow:
    """One analyst label."""

    id: int
    alert_id: int
    label: str
    analyst: str
    note: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RecentDecision:
    """The fields the monitoring summary reads from one recent decision."""

    score: float
    decision: str
    degraded: bool
    latency_ms: int


def _record(row: Mapping[Any, Any]) -> AuditRecord:
    return AuditRecord(
        transaction_id=row["transaction_id"],
        card_id=row["card_id"],
        event_ts=int(row["event_time"].timestamp()),
        amount=float(row["amount"]),
        category=row["category"],
        features=features_from_json(row["features"]),
        margin=float(row["margin"]),
        score=float(row["score"]),
        decision=row["decision"],
        t_review=float(row["t_review"]),
        t_block=float(row["t_block"]),
        model_version=row["model_version"],
        reason_codes=tuple(ReasonCode(**reason) for reason in row["reason_codes"]),
        degraded=bool(row["degraded"]),
        latency_ms=int(row["latency_ms"]),
        created_at=row["created_at"],
    )


class Repository:
    """Every query the service makes, on one async engine.

    Args:
        engine: SQLAlchemy async engine on the asyncpg driver.
        timeout_s: Seconds allowed for each operation before it counts as unavailable.
    """

    def __init__(self, engine: AsyncEngine, *, timeout_s: float) -> None:
        """Use ``engine`` for every query."""
        self.engine = engine
        self.timeout_s = timeout_s

    async def _guard(self, awaitable: Awaitable[T]) -> T:
        try:
            async with asyncio.timeout(self.timeout_s):
                return await awaitable
        except (SQLAlchemyError, OSError, TimeoutError) as error:
            raise AuditUnavailable(f"postgres: {type(error).__name__}") from error

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncConnection]:
        async with self.engine.begin() as conn:
            yield conn

    async def ping(self) -> bool:
        """True if Postgres answers within the timeout."""

        async def run() -> None:
            async with self.engine.connect() as conn:
                await conn.execute(sa.text("SELECT 1"))

        try:
            await self._guard(run())
        except AuditUnavailable:
            return False
        return True

    async def insert_decision(self, record: AuditRecord) -> bool:
        """Insert a decision, and an alert for a review or block, in one transaction.

        Returns:
            ``False`` if a decision with this ``transaction_id`` already exists; nothing
            is written then.
        """

        async def run() -> bool:
            async with self._transaction() as conn:
                inserted = await conn.execute(
                    insert(decisions)
                    .values(
                        transaction_id=record.transaction_id,
                        card_id=record.card_id,
                        event_time=datetime.fromtimestamp(record.event_ts, tz=UTC),
                        amount=record.amount,
                        category=record.category,
                        features=features_to_json(record.features),
                        margin=record.margin,
                        score=record.score,
                        decision=record.decision,
                        t_review=record.t_review,
                        t_block=record.t_block,
                        model_version=record.model_version,
                        reason_codes=[
                            {
                                "id": r.id,
                                "text": r.text,
                                "contribution": r.contribution,
                                "share": r.share,
                            }
                            for r in record.reason_codes
                        ],
                        degraded=record.degraded,
                        latency_ms=record.latency_ms,
                    )
                    .on_conflict_do_nothing(index_elements=["transaction_id"])
                    .returning(decisions.c.transaction_id)
                )
                if inserted.first() is None:
                    return False
                if record.decision != "approve":
                    await conn.execute(
                        alerts.insert().values(
                            transaction_id=record.transaction_id,
                            priority=record.score * record.amount,
                        )
                    )
                return True

        return await self._guard(run())

    async def get_decision(self, transaction_id: str) -> AuditRecord | None:
        """The recorded decision for ``transaction_id``, or ``None``."""

        async def run() -> AuditRecord | None:
            async with self.engine.connect() as conn:
                row = (
                    (
                        await conn.execute(
                            sa.select(decisions).where(decisions.c.transaction_id == transaction_id)
                        )
                    )
                    .mappings()
                    .first()
                )
            return None if row is None else _record(row)

        return await self._guard(run())

    async def list_alerts(
        self,
        *,
        status: str | None,
        decision: str | None,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[AlertRow], str | None]:
        """Alerts by priority (score times amount), highest first, one page at a time.

        Returns:
            The page and the cursor for the next one, ``None`` on the last page.

        Raises:
            ValueError: if ``cursor`` is malformed.
        """
        query = (
            sa.select(
                alerts,
                decisions.c.decision,
                decisions.c.score.label("risk_score"),
                decisions.c.amount,
                decisions.c.category,
                decisions.c.event_time,
            )
            .join(decisions, decisions.c.transaction_id == alerts.c.transaction_id)
            .order_by(alerts.c.priority.desc(), alerts.c.id.desc())
            .limit(limit + 1)
        )
        if status is not None:
            query = query.where(alerts.c.status == status)
        if decision is not None:
            query = query.where(decisions.c.decision == decision)
        if cursor is not None:
            priority, alert_id = decode_cursor(cursor)
            query = query.where(
                sa.or_(
                    alerts.c.priority < priority,
                    sa.and_(alerts.c.priority == priority, alerts.c.id < alert_id),
                )
            )

        async def run() -> list[AlertRow]:
            async with self.engine.connect() as conn:
                rows = (await conn.execute(query)).mappings().all()
            return [AlertRow(**dict(row)) for row in rows]

        rows = await self._guard(run())
        page = rows[:limit]
        next_cursor = encode_cursor(page[-1].priority, page[-1].id) if len(rows) > limit else None
        return page, next_cursor

    async def add_disposition(
        self, alert_id: int, *, label: str, analyst: str, note: str | None
    ) -> DispositionRow | None:
        """Record an analyst's label and close the alert; ``None`` if no such alert."""

        async def run() -> DispositionRow | None:
            async with self._transaction() as conn:
                found = (
                    await conn.execute(
                        sa.select(alerts.c.id).where(alerts.c.id == alert_id).with_for_update()
                    )
                ).first()
                if found is None:
                    return None
                row = (
                    (
                        await conn.execute(
                            dispositions.insert()
                            .values(alert_id=alert_id, label=label, analyst=analyst, note=note)
                            .returning(*dispositions.c)
                        )
                    )
                    .mappings()
                    .one()
                )
                await conn.execute(
                    alerts.update().where(alerts.c.id == alert_id).values(status="closed")
                )
                return DispositionRow(**dict(row))

        return await self._guard(run())

    async def record_model_version(
        self, version: str, *, manifest_hash: str, metrics: Mapping[str, Any]
    ) -> None:
        """Record a release the first time it is activated; later activations are no-ops."""

        async def run() -> None:
            async with self._transaction() as conn:
                await conn.execute(
                    insert(model_versions)
                    .values(version=version, manifest_hash=manifest_hash, metrics=dict(metrics))
                    .on_conflict_do_nothing(index_elements=["version"])
                )

        await self._guard(run())

    async def recent_decisions(self, limit: int) -> list[RecentDecision]:
        """The newest ``limit`` decisions, newest first."""

        async def run() -> list[RecentDecision]:
            async with self.engine.connect() as conn:
                rows = (
                    await conn.execute(
                        sa.select(
                            decisions.c.score,
                            decisions.c.decision,
                            decisions.c.degraded,
                            decisions.c.latency_ms,
                        )
                        .order_by(decisions.c.created_at.desc())
                        .limit(limit)
                    )
                ).all()
            return [RecentDecision(float(s), d, bool(g), int(ms)) for s, d, g, ms in rows]

        return await self._guard(run())
