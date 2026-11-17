"""Replay the public test file through the running service, as a card network would.

The client reads ``data/raw/sparkov/fraudTest.csv`` and turns each card number into its
HMAC token before anything is sent, so raw card numbers never reach the service. It does
not read ``data/processed/test.parquet``, which only the one-time evaluation may open.

Transactions are posted in time order, either at a fixed rate or at a multiple of real
time. Requests on one card are sent one after another, as an issuer would see them;
different cards run concurrently.

``--warm-days N`` first loads each card's last N days of history from the ingested
transactions straight into Redis, so the first replayed transactions see the same history
the offline features did. Add ``--warm-only`` to load the history and send nothing.

``--inject-drift CATEGORY=FACTOR`` multiplies amounts in one category from part-way
through the stream, to show the drift alarm on demand.

Usage::

    python -m fraudlens.tools.replay --limit 5000 --rate 50 --warm-days 30
"""

import argparse
import asyncio
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import numpy.typing as npt
import pandas as pd

from fraudlens.common.logging import configure_logging
from fraudlens.common.paths import RAW_DIR, TRANSACTIONS_PATH
from fraudlens.common.settings import get_settings
from fraudlens.common.types import HistoryEntry
from fraudlens.data.pii import clean_sparkov
from fraudlens.data.schema import SPARKOV_ROWS, validate_raw_sparkov
from fraudlens.features.definitions import DAY_S
from fraudlens.features.records import to_history_entry

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

TEST_CSV = RAW_DIR / "sparkov" / "fraudTest.csv"
OPTIONAL = ("merchant_lat", "merchant_long", "home_lat", "home_long", "city_pop", "cardholder_age")
INTEGER_FIELDS = frozenset({"city_pop", "cardholder_age"})


def to_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the scoring request body for one ingested transaction row.

    ``event_ts`` (Unix seconds, UTC) becomes an ISO 8601 ``event_time``. Optional fields
    that are missing (NaN or ``None``) are left out of the body.
    """
    body: dict[str, Any] = {
        "transaction_id": str(row["transaction_id"]),
        "card_id": str(row["card_id"]),
        "event_time": datetime.fromtimestamp(int(row["event_ts"]), tz=UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "amount": float(row["amount"]),
        "category": str(row["category"]),
        "merchant_id": str(row["merchant_id"]),
    }
    for name in OPTIONAL:
        value = row.get(name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        body[name] = int(value) if name in INTEGER_FIELDS else float(value)
    return body


def parse_drift(text: str) -> tuple[str, float]:
    """Parse ``CATEGORY=FACTOR``, for example ``shopping_net=1.5``.

    Raises:
        argparse.ArgumentTypeError: if the text is malformed or the factor is not positive.
    """
    category, sep, factor = text.partition("=")
    try:
        value = float(factor)
    except ValueError:
        value = math.nan
    if not sep or not category or not value > 0:
        raise argparse.ArgumentTypeError(f"expected CATEGORY=FACTOR, got {text!r}")
    return category, value


def inject_drift(
    stream: pd.DataFrame, category: str, factor: float, *, after: float
) -> pd.DataFrame:
    """Multiply amounts in ``category`` by ``factor`` for rows past fraction ``after``.

    Amounts are rounded to cents, as a card network would send them.
    """
    out = stream.copy()
    late = np.arange(len(out)) >= int(after * len(out))
    hit = late & (out["category"] == category).to_numpy()
    out.loc[hit, "amount"] = (out.loc[hit, "amount"] * factor).round(2)
    return out


def send_offsets(
    event_ts: npt.NDArray[np.int64], *, rate: float | None, speed: float | None
) -> FloatArray:
    """Seconds after the start at which each transaction is sent.

    Args:
        event_ts: Transaction times, ascending, in Unix seconds.
        rate: Fixed requests per second; takes precedence over ``speed``.
        speed: Multiple of real time; 60 replays one hour in one minute.

    Returns:
        Offsets in seconds, all zero when neither is given (as fast as possible).
    """
    offsets: FloatArray = np.zeros(len(event_ts), dtype=np.float64)
    if rate:
        offsets = np.arange(len(event_ts), dtype=np.float64) / rate
    elif speed:
        offsets = (event_ts - event_ts[0]).astype(np.float64) / speed
    return offsets


def load_stream(
    path: Path, *, hmac_key: bytes, start: int | None, limit: int | None
) -> pd.DataFrame:
    """Read the raw test file, tokenise cards and return rows in time order.

    The file is validated against the raw schema and its known row count first.
    """
    raw = pd.read_csv(path)
    validate_raw_sparkov(raw, expected_rows=SPARKOV_ROWS["test"])
    stream = clean_sparkov(raw, partition="test", hmac_key=hmac_key)
    stream = stream.sort_values(["event_ts", "transaction_id"], kind="stable")
    if start is not None:
        stream = stream[stream["event_ts"] >= start]
    return stream.head(limit).reset_index(drop=True) if limit else stream.reset_index(drop=True)


def warm_histories(
    transactions: pd.DataFrame, *, before: int, days: int
) -> dict[str, list[HistoryEntry]]:
    """Each card's transactions in the ``days`` before ``before``, as history entries."""
    window = transactions[
        (transactions["event_ts"] >= before - days * DAY_S) & (transactions["event_ts"] < before)
    ]
    histories: dict[str, list[HistoryEntry]] = {}
    for row in cast(list[dict[str, Any]], window.to_dict(orient="records")):
        histories.setdefault(str(row["card_id"]), []).append(to_history_entry(row))
    return histories


@dataclass
class ReplayStats:
    """What came back from the service."""

    sent: int = 0
    statuses: Counter[int] = field(default_factory=Counter)
    decisions: Counter[str] = field(default_factory=Counter)
    degraded: int = 0
    replayed: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """A JSON-ready summary with client-side latency percentiles in milliseconds."""
        latency = np.array(self.latency_ms) if self.latency_ms else np.array([math.nan])
        return {
            "sent": self.sent,
            "statuses": {str(k): v for k, v in sorted(self.statuses.items())},
            "decisions": dict(self.decisions),
            "degraded": self.degraded,
            "replayed": self.replayed,
            "client_latency_ms": {
                f"p{q}": round(float(np.percentile(latency, q)), 2) for q in (50, 95, 99)
            },
        }


async def replay(
    client: httpx.AsyncClient,
    stream: pd.DataFrame,
    offsets: Sequence[float],
    *,
    concurrency: int = 32,
) -> ReplayStats:
    """Post every row of ``stream`` at its offset; requests on one card never overlap."""
    stats = ReplayStats()
    gate = asyncio.Semaphore(concurrency)
    previous: dict[str, asyncio.Task[None]] = {}
    started = time.perf_counter()

    async def send(body: dict[str, Any], before: asyncio.Task[None] | None) -> None:
        if before is not None:
            await before
        async with gate:
            sent_at = time.perf_counter()
            try:
                response = await client.post("/v1/transactions/score", json=body)
            except httpx.HTTPError:
                stats.statuses[0] += 1
                return
            stats.latency_ms.append((time.perf_counter() - sent_at) * 1000)
        stats.statuses[response.status_code] += 1
        if response.status_code == 200:
            result = response.json()
            stats.decisions[result["decision"]] += 1
            stats.degraded += bool(result["degraded"])
            stats.replayed += response.headers.get("Idempotent-Replay") == "true"

    rows = cast(list[dict[str, Any]], stream.to_dict(orient="records"))
    for row, offset in zip(rows, offsets, strict=True):
        delay = started + offset - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        card = str(row["card_id"])
        previous[card] = asyncio.create_task(send(to_payload(row), previous.get(card)))
        stats.sent += 1
    await asyncio.gather(*previous.values())
    return stats


async def _seed(redis_url: str, histories: Mapping[str, list[HistoryEntry]]) -> int:
    from redis.asyncio import Redis

    from fraudlens.service.feature_store import RedisFeatureStore

    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        store = RedisFeatureStore(client, timeout_s=30.0, lock_wait_s=1.0, idempotency_ttl_s=1)
        return await store.seed(histories)
    finally:
        await client.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m fraudlens.tools.replay``."""
    parser = argparse.ArgumentParser(prog="python -m fraudlens.tools.replay")
    parser.add_argument("--source", type=Path, default=TEST_CSV)
    parser.add_argument("--limit", type=int, default=None, help="transactions to send")
    parser.add_argument("--start", type=str, default=None, help="first event date, YYYY-MM-DD")
    pace = parser.add_mutually_exclusive_group()
    pace.add_argument("--rate", type=float, default=None, help="requests per second")
    pace.add_argument("--speed", type=float, default=None, help="multiple of real time")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warm-days", type=int, default=0)
    parser.add_argument(
        "--warm-only", action="store_true", help="seed card history, then exit without sending"
    )
    parser.add_argument("--inject-drift", type=parse_drift, default=None)
    parser.add_argument("--drift-after", type=float, default=0.5, help="fraction of the stream")
    parser.add_argument("--url", type=str, default=None)
    args = parser.parse_args(argv)
    if args.warm_only and not args.warm_days:
        parser.error("--warm-only needs --warm-days")

    settings = get_settings()
    configure_logging(settings.log_level)
    start = (
        int(datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp())
        if args.start
        else None
    )
    stream = load_stream(
        args.source,
        hmac_key=settings.card_hmac_key.get_secret_value().encode("utf-8"),
        start=start,
        limit=args.limit,
    )
    if args.inject_drift is not None:
        category, factor = args.inject_drift
        stream = inject_drift(stream, category, factor, after=args.drift_after)
    if args.warm_days:
        histories = warm_histories(
            pd.read_parquet(TRANSACTIONS_PATH),
            before=int(stream["event_ts"].iloc[0]),
            days=args.warm_days,
        )
        cards = asyncio.run(_seed(settings.redis_url, histories))
        logger.info("history warmed", extra={"cards": cards, "days": args.warm_days})
        if args.warm_only:
            print(f"warmed {cards:,} cards with {args.warm_days} days of history")
            return 0

    offsets = send_offsets(stream["event_ts"].to_numpy(), rate=args.rate, speed=args.speed)

    async def run() -> ReplayStats:
        async with httpx.AsyncClient(
            base_url=args.url or settings.service_url,
            headers={"X-API-Key": settings.api_key.get_secret_value()},
            timeout=10.0,
        ) as client:
            return await replay(client, stream, offsets.tolist(), concurrency=args.concurrency)

    stats = asyncio.run(run())
    print(json.dumps(stats.summary(), indent=2))
    return 0 if set(stats.statuses) <= {200} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
