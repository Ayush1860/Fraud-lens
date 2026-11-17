"""Synthetic scoring requests for load tests.

A fixed pool of cards, each with a home and a typical spend, makes purchases at a steady
pace in simulated time. Cards repeat, so each card's history grows during a run and the
service does its full work: history read, per-card lock, features, append. Amounts are
log-normal around each card's typical spend, so a small share of requests are flagged and
exercise the explanation path too.

No real data is used: a GitHub runner has none.
"""

import itertools
import threading
from datetime import UTC, datetime
from typing import Any

import numpy as np

from fraudlens.features.definitions import CATEGORIES

START = datetime(2019, 6, 1, tzinfo=UTC)
"""Simulated time starts here and only moves forward."""


class SyntheticTraffic:
    """Thread-safe generator of request bodies for ``POST /v1/transactions/score``.

    Args:
        n_cards: Size of the card pool.
        seed: Seed for the card pool and the purchases.
        run: Goes into every transaction id, so separate runs never collide.
        seconds_per_request: Average simulated seconds between purchases, across all cards.
    """

    def __init__(
        self,
        n_cards: int = 5_000,
        *,
        seed: int = 0,
        run: str = "load",
        seconds_per_request: float = 2.0,
    ) -> None:
        """Draw the card pool."""
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._counter = itertools.count()
        self._clock = START.timestamp()
        self._run = run
        self._step = seconds_per_request
        rng = self._rng
        self._cards = [f"c_{i:016x}" for i in range(n_cards)]
        self._home = np.column_stack(
            [rng.uniform(26.0, 47.0, n_cards), rng.uniform(-122.0, -72.0, n_cards)]
        )
        self._typical = rng.lognormal(3.6, 0.6, n_cards)
        self._age = rng.integers(18, 90, n_cards)
        self._pop = rng.integers(200, 2_000_000, n_cards)
        self._categories = [c for c in CATEGORIES if c != "unknown"]

    def next(self) -> dict[str, Any]:
        """The next request body."""
        with self._lock:
            i = next(self._counter)
            rng = self._rng
            self._clock += float(rng.exponential(self._step))
            card = int(rng.integers(0, len(self._cards)))
            amount = round(float(self._typical[card] * rng.lognormal(0.0, 0.8)), 2)
            lat, long = self._home[card] + rng.normal(0.0, 0.5, 2)
            category = self._categories[int(rng.integers(0, len(self._categories)))]
            merchant = int(rng.integers(0, 2_000))
            when = datetime.fromtimestamp(int(self._clock), tz=UTC)
        return {
            "transaction_id": f"{self._run}-{i:09d}",
            "card_id": self._cards[card],
            "event_time": when.isoformat().replace("+00:00", "Z"),
            "amount": max(amount, 1.0),
            "category": category,
            "merchant_id": f"m_{merchant:04d}",
            "merchant_lat": round(float(lat), 6),
            "merchant_long": round(float(long), 6),
            "home_lat": round(float(self._home[card][0]), 4),
            "home_long": round(float(self._home[card][1]), 4),
            "city_pop": int(self._pop[card]),
            "cardholder_age": int(self._age[card]),
        }
