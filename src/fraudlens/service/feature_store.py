"""Redis: each card's recent history, idempotency keys and cached explanations.

Keys:

* ``hist:{card_id}``, a list of the card's past transactions in arrival order, each
  ``"{event_ts}|{json}"``. It is trimmed to the 30-day window and the 500-entry cap on
  every append, by a Lua script so the trim is atomic.
* ``idem:{transaction_id}``, ``"pending"`` while the first request is being decided, then
  the response body. Claimed with ``SET NX``.
* ``lock:{card_id}``, the per-card lock held from the history read to the append.
* ``expl:{transaction_id}:{model_version}``, a cached explanation.

Every Redis failure, and every call slower than the configured timeout, surfaces as
:class:`~fraudlens.service.decision.StoreUnavailable`, which sends the request down the
degraded path.
"""

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import LockError, RedisError

from fraudlens.common.types import HistoryEntry
from fraudlens.features.definitions import DAY_S, HISTORY_MAX_ENTRIES, HISTORY_WINDOW_S
from fraudlens.service.decision import PENDING, StoreUnavailable

logger = logging.getLogger(__name__)

LOCK_LEASE_S = 2.0
"""A held card lock expires after this long, so a crashed request cannot block a card."""

PENDING_TTL_S = 10
"""Seconds a ``pending`` claim lives; a request that dies mid-flight frees its id soon."""

HISTORY_TTL_S = HISTORY_WINDOW_S + DAY_S
"""Seconds of inactivity after which a card's whole history is dropped."""

# Append one entry, pop entries older than the window from the head, cap the length.
_APPEND = """
redis.call('RPUSH', KEYS[1], ARGV[1])
local cutoff = tonumber(ARGV[2])
while true do
  local head = redis.call('LINDEX', KEYS[1], 0)
  if not head then break end
  local ts = tonumber(string.match(head, '^(%-?%d+)|'))
  if ts == nil or ts >= cutoff then break end
  redis.call('LPOP', KEYS[1])
end
redis.call('LTRIM', KEYS[1], -tonumber(ARGV[3]), -1)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[4]))
return redis.call('LLEN', KEYS[1])
"""


def _finite_or_none(value: float) -> float | None:
    return None if math.isnan(value) else value


def encode_entry(entry: HistoryEntry) -> str:
    """Serialise a history entry as ``"{event_ts}|[id, cents, category, merchant, lat, long]"``."""
    body = [
        entry.transaction_id,
        entry.amount_cents,
        entry.category,
        entry.merchant_id,
        _finite_or_none(entry.merchant_lat),
        _finite_or_none(entry.merchant_long),
    ]
    return f"{entry.event_ts}|{json.dumps(body, separators=(',', ':'))}"


def decode_entry(raw: str) -> HistoryEntry:
    """Inverse of :func:`encode_entry`. Missing coordinates come back as NaN."""
    ts, _, body = raw.partition("|")
    tid, cents, category, merchant, lat, long = json.loads(body)
    return HistoryEntry(
        transaction_id=tid,
        event_ts=int(ts),
        amount_cents=int(cents),
        category=category,
        merchant_id=merchant,
        merchant_lat=math.nan if lat is None else float(lat),
        merchant_long=math.nan if long is None else float(long),
    )


class RedisFeatureStore:
    """The online feature store and idempotency keys, on one Redis database.

    Args:
        client: A ``redis.asyncio`` client created with ``decode_responses=True``.
        timeout_s: Seconds allowed for each call before it counts as unavailable.
        lock_wait_s: Seconds to wait for a card's lock before giving up.
        idempotency_ttl_s: Seconds a completed decision stays replayable from Redis.
    """

    def __init__(
        self,
        client: "Redis",
        *,
        timeout_s: float,
        lock_wait_s: float,
        idempotency_ttl_s: int,
    ) -> None:
        """Wrap ``client``; the append script is registered once here."""
        self.client = client
        self.timeout_s = timeout_s
        self.lock_wait_s = lock_wait_s
        self.idempotency_ttl_s = idempotency_ttl_s
        self._append = client.register_script(_APPEND)

    async def _call(self, awaitable: Awaitable[Any]) -> Any:
        try:
            async with asyncio.timeout(self.timeout_s):
                return await awaitable
        except (RedisError, OSError, TimeoutError) as error:
            raise StoreUnavailable(f"redis: {type(error).__name__}") from error

    async def ping(self) -> bool:
        """True if Redis answers within the timeout."""
        try:
            return bool(await self._call(self.client.ping()))
        except StoreUnavailable:
            return False

    async def claim(self, transaction_id: str) -> str | None:
        """Claim ``transaction_id``; ``None`` if this call claimed it, else the stored value."""
        previous: str | None = await self._call(
            self.client.set(f"idem:{transaction_id}", PENDING, nx=True, ex=PENDING_TTL_S, get=True)
        )
        return previous

    async def peek(self, transaction_id: str) -> str | None:
        """The stored value for ``transaction_id``, or ``None``."""
        value: str | None = await self._call(self.client.get(f"idem:{transaction_id}"))
        return value

    async def complete(self, transaction_id: str, response: str) -> None:
        """Replace the claim with the response body, kept for ``idempotency_ttl_s``."""
        await self._call(
            self.client.set(f"idem:{transaction_id}", response, ex=self.idempotency_ttl_s)
        )

    async def release(self, transaction_id: str) -> None:
        """Delete the claim so a retry is decided afresh."""
        await self._call(self.client.delete(f"idem:{transaction_id}"))

    @asynccontextmanager
    async def card_lock(self, card_id: str) -> AsyncIterator[None]:
        """Hold ``lock:{card_id}`` for the block.

        Raises:
            StoreUnavailable: if Redis fails or the lock is not free within ``lock_wait_s``.
        """
        lock = self.client.lock(
            f"lock:{card_id}",
            timeout=LOCK_LEASE_S,
            sleep=0.002,
            blocking_timeout=self.lock_wait_s,
        )
        try:
            async with asyncio.timeout(self.lock_wait_s + self.timeout_s):
                acquired = await lock.acquire()
        except (RedisError, OSError, TimeoutError) as error:
            raise StoreUnavailable(f"card lock: {type(error).__name__}") from error
        if not acquired:
            raise StoreUnavailable("card lock busy")
        try:
            yield
        finally:
            try:
                await self._call(lock.release())
            except (StoreUnavailable, LockError):
                # The lease expires on its own; a failed release only delays the next request.
                logger.warning("card lock not released", exc_info=True)

    async def history(self, card_id: str) -> list[HistoryEntry]:
        """The card's history, sorted by ``(event_ts, transaction_id)``."""
        raw: list[str] = await self._call(self.client.lrange(f"hist:{card_id}", 0, -1))
        entries = [decode_entry(item) for item in raw]
        entries.sort(key=lambda entry: (entry.event_ts, entry.transaction_id))
        return entries

    async def append(
        self, card_id: str, entry: HistoryEntry, history: Sequence[HistoryEntry]
    ) -> None:
        """Append ``entry`` and trim the list to the window and the cap.

        The window is measured back from the newest event on the card, so an event that
        arrives late never shortens the history of a card that has moved on.
        """
        newest = max([entry.event_ts, *(e.event_ts for e in history[-1:])])
        await self._call(
            self._append(
                keys=[f"hist:{card_id}"],
                args=[
                    encode_entry(entry),
                    newest - HISTORY_WINDOW_S,
                    HISTORY_MAX_ENTRIES,
                    HISTORY_TTL_S,
                ],
            )
        )

    async def seed(self, histories: Mapping[str, Iterable[HistoryEntry]]) -> int:
        """Load whole histories at once, replacing any stored for those cards.

        For warming the store before a replay. Each card keeps its newest
        ``HISTORY_MAX_ENTRIES`` entries in time order. Returns the number of cards written.
        """
        pipe = self.client.pipeline(transaction=False)
        for card_id, entries in histories.items():
            ordered = sorted(entries, key=lambda e: (e.event_ts, e.transaction_id))
            key = f"hist:{card_id}"
            pipe.delete(key)
            if ordered:
                pipe.rpush(key, *[encode_entry(e) for e in ordered[-HISTORY_MAX_ENTRIES:]])
                pipe.expire(key, HISTORY_TTL_S)
        try:
            await pipe.execute()
        except (RedisError, OSError) as error:
            raise StoreUnavailable(f"redis: {type(error).__name__}") from error
        return len(histories)

    async def cache_get(self, key: str) -> Any | None:
        """A cached JSON document, or ``None`` if absent or Redis is unavailable."""
        try:
            raw: str | None = await self._call(self.client.get(key))
        except StoreUnavailable:
            return None
        return None if raw is None else json.loads(raw)

    async def cache_set(self, key: str, value: Any, ttl_s: int) -> None:
        """Cache a JSON document; a Redis failure is ignored, since it is only a cache."""
        try:
            await self._call(self.client.set(key, json.dumps(value), ex=ttl_s))
        except StoreUnavailable:
            logger.warning("cache write skipped", exc_info=True)
