"""The Redis feature store, against in-process fakeredis (which runs the Lua scripts)."""

import asyncio
import math
from collections.abc import AsyncIterator

import fakeredis
import pytest
from redis.asyncio import Redis

from fraudlens.common.types import HistoryEntry
from fraudlens.features.definitions import DAY_S, HISTORY_MAX_ENTRIES, HISTORY_WINDOW_S
from fraudlens.service.decision import PENDING, StoreUnavailable
from fraudlens.service.feature_store import RedisFeatureStore, decode_entry, encode_entry

pytestmark = pytest.mark.anyio

CARD = "c_0123456789abcdef"


def entry(i: int, ts: int, *, lat: float = 40.0) -> HistoryEntry:
    return HistoryEntry(f"t{i:04d}", ts, 1_000 + i, "grocery_pos", f"m_{i % 3}", lat, -74.0)


@pytest.fixture
async def client() -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield redis
    await redis.aclose()


@pytest.fixture
def store(client: fakeredis.FakeAsyncRedis) -> RedisFeatureStore:
    return RedisFeatureStore(client, timeout_s=1.0, lock_wait_s=0.05, idempotency_ttl_s=60)


@pytest.fixture
async def dead_store() -> AsyncIterator[RedisFeatureStore]:
    client = Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2, decode_responses=True)
    yield RedisFeatureStore(client, timeout_s=0.5, lock_wait_s=0.05, idempotency_ttl_s=60)
    await client.aclose()


def test_entries_round_trip_including_missing_coordinates() -> None:
    original = entry(1, 1_600_000_000, lat=math.nan)

    decoded = decode_entry(encode_entry(original))

    assert encode_entry(original).startswith("1600000000|")
    assert decoded.transaction_id == original.transaction_id
    assert decoded.amount_cents == original.amount_cents
    assert math.isnan(decoded.merchant_lat)
    assert decoded.merchant_long == -74.0


async def test_claim_complete_and_release(store: RedisFeatureStore) -> None:
    assert await store.claim("t1") is None
    assert await store.claim("t1") == PENDING
    await store.complete("t1", '{"decision": "approve"}')
    assert await store.claim("t1") == '{"decision": "approve"}'
    assert await store.peek("t1") == '{"decision": "approve"}'
    await store.release("t1")
    assert await store.peek("t1") is None


async def test_history_comes_back_in_time_order(store: RedisFeatureStore) -> None:
    late, early = entry(2, 1_000_200), entry(1, 1_000_100)
    await store.append(CARD, late, [])
    await store.append(CARD, early, [late])  # arrived out of order

    assert [e.transaction_id for e in await store.history(CARD)] == ["t0001", "t0002"]


async def test_append_drops_entries_older_than_the_window(
    store: RedisFeatureStore, client: fakeredis.FakeAsyncRedis
) -> None:
    start = 1_600_000_000
    old = [entry(i, start + i) for i in range(3)]
    await store.seed({CARD: old})
    new = entry(9, start + HISTORY_WINDOW_S + 2)

    await store.append(CARD, new, old)

    assert [e.transaction_id for e in await store.history(CARD)] == ["t0002", "t0009"]
    assert await client.ttl(f"hist:{CARD}") > HISTORY_WINDOW_S


async def test_append_caps_the_history(store: RedisFeatureStore) -> None:
    full = [entry(i, 1_600_000_000 + i) for i in range(HISTORY_MAX_ENTRIES + 20)]
    await store.seed({CARD: full})
    history = await store.history(CARD)
    assert len(history) == HISTORY_MAX_ENTRIES

    await store.append(CARD, entry(9_999, 1_600_000_000 + DAY_S), history)

    stored = await store.history(CARD)
    assert len(stored) == HISTORY_MAX_ENTRIES
    assert stored[-1].transaction_id == "t9999"
    assert stored[0].transaction_id == "t0021"


async def test_card_lock_serialises_and_gives_up_when_busy(store: RedisFeatureStore) -> None:
    order: list[str] = []

    async def hold(name: str, seconds: float) -> None:
        async with store.card_lock(CARD):
            order.append(f"{name} in")
            await asyncio.sleep(seconds)
            order.append(f"{name} out")

    first = asyncio.create_task(hold("a", 0.2))
    await asyncio.sleep(0.02)
    with pytest.raises(StoreUnavailable, match="busy"):
        await hold("b", 0)
    await first
    await hold("c", 0)

    assert order == ["a in", "a out", "c in", "c out"]


async def test_cache_round_trip(store: RedisFeatureStore) -> None:
    await store.cache_set("expl:t1:v1", {"margin": -3.5}, 60)

    assert await store.cache_get("expl:t1:v1") == {"margin": -3.5}
    assert await store.cache_get("expl:missing") is None
    assert await store.ping() is True


async def test_an_unreachable_redis_is_reported_as_unavailable(
    dead_store: RedisFeatureStore,
) -> None:
    assert await dead_store.ping() is False
    assert await dead_store.cache_get("anything") is None
    await dead_store.cache_set("anything", 1, 60)  # ignored, not raised
    for call in (
        dead_store.claim("t1"),
        dead_store.history(CARD),
        dead_store.append(CARD, entry(1, 1), []),
        dead_store.seed({CARD: [entry(1, 1)]}),
    ):
        with pytest.raises(StoreUnavailable):
            await call
    with pytest.raises(StoreUnavailable):
        async with dead_store.card_lock(CARD):
            pass


async def test_a_slow_call_counts_as_unavailable(store: RedisFeatureStore) -> None:
    store.timeout_s = 0.01

    with pytest.raises(StoreUnavailable, match="TimeoutError"):
        await store._call(asyncio.sleep(1))
