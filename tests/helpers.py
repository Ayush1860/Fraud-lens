"""Shared test helpers: a synthetic transaction generator and the slow reference path.

Nothing here re-implements a feature. The reference path only replays transactions in
order and calls ``compute_features``, which is the single definition.
"""

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from fraudlens.common.types import HistoryEntry, Transaction
from fraudlens.features.definitions import (
    CATEGORIES,
    FEATURE_NAMES,
    FeatureVector,
    amount_to_cents,
    compute_features,
)

# Features that go through log / trigonometric functions. numpy and ``math`` may differ in
# the last bit for these, so they are compared to 1e-9. Every other feature must be equal.
TRANSCENDENTAL = frozenset(
    {"log_amount", "log_city_pop", "dist_home_km", "dist_prev_km", "speed_prev_kmh"}
)
REL_TOL = 1e-9
ABS_TOL = 1e-9

# More prior rows than the history cap, so slicing never changes what the definition sees.
_PRIOR_ROWS = 600


def make_transactions(
    n_cards: int = 20,
    n_txns: int = 2_000,
    *,
    seed: int = 0,
    start_ts: int = 1_546_300_800,  # 2019-01-01T00:00:00Z
    span_days: int = 120,
) -> pd.DataFrame:
    """Build a Sparkov-shaped frame that deliberately hits the awkward cases.

    Besides ordinary traffic it contains bursts inside one hour, several transactions in
    the same second on one card, and gaps of exactly 1 hour, 24 hours, 7 days and 30 days,
    which sit on the closed edge of each window.
    """
    rng = np.random.default_rng(seed)
    card = rng.integers(0, n_cards, n_txns)
    ts = start_ts + rng.integers(0, span_days * 86_400, n_txns)

    # Bursts, same-second ties and exact window-edge gaps, each copied from another row.
    for share, offsets in (
        (0.10, np.array([1, 30, 600, 3_599])),
        (0.05, np.array([0])),
        (0.05, np.array([3_600, 86_400, 7 * 86_400, 30 * 86_400])),
    ):
        k = int(n_txns * share)
        src = rng.integers(0, n_txns, k)
        dst = rng.integers(0, n_txns, k)
        card[dst] = card[src]
        ts[dst] = ts[src] + rng.choice(offsets, k)

    home_lat = rng.uniform(25, 48, n_cards).round(4)
    home_long = rng.uniform(-122, -70, n_cards).round(4)
    city_pop = rng.integers(50, 3_000_000, n_cards)
    age = rng.integers(18, 90, n_cards)

    amount = np.round(np.exp(rng.normal(3.5, 1.2, n_txns)), 2).clip(1.0, 25_000.0)
    # A few cards repeat one amount, which makes the 30-day standard deviation exactly zero.
    flat = card < max(1, n_cards // 10)
    amount[flat] = 42.50

    return pd.DataFrame(
        {
            "transaction_id": [f"{value:032x}" for value in rng.permutation(n_txns)],
            "card_id": [f"c_{value:04d}" for value in card],
            "event_ts": ts.astype(np.int64),
            "amount": amount,
            "category": rng.choice(CATEGORIES[:-1], n_txns),
            "merchant_id": [f"m_{value:03d}" for value in rng.integers(0, 40, n_txns)],
            "merchant_lat": (home_lat[card] + rng.normal(0, 0.7, n_txns)).round(6),
            "merchant_long": (home_long[card] + rng.normal(0, 0.7, n_txns)).round(6),
            "home_lat": home_lat[card],
            "home_long": home_long[card],
            "city_pop": city_pop[card].astype(np.int64),
            "cardholder_age": age[card].astype(np.int64),
        }
    )


def to_transactions(rows: pd.DataFrame) -> list[Transaction]:
    """Convert frame rows into serving-side ``Transaction`` objects, in row order."""
    return [
        Transaction(
            transaction_id=str(tid),
            card_id=str(card),
            event_ts=int(ts),
            amount=float(amount),
            category=str(category),
            merchant_id=str(merchant),
            merchant_lat=float(mlat),
            merchant_long=float(mlong),
            home_lat=float(hlat),
            home_long=float(hlong),
            city_pop=float(pop),
            cardholder_age=float(age),
        )
        for tid, card, ts, amount, category, merchant, mlat, mlong, hlat, hlong, pop, age in zip(
            rows["transaction_id"],
            rows["card_id"],
            rows["event_ts"],
            rows["amount"],
            rows["category"],
            rows["merchant_id"],
            rows["merchant_lat"],
            rows["merchant_long"],
            rows["home_lat"],
            rows["home_long"],
            rows["city_pop"],
            rows["cardholder_age"],
            strict=True,
        )
    ]


def to_entry(txn: Transaction) -> HistoryEntry:
    """Convert a transaction into the ``HistoryEntry`` the feature store would keep."""
    return HistoryEntry(
        transaction_id=txn.transaction_id,
        event_ts=txn.event_ts,
        amount_cents=amount_to_cents(txn.amount),
        category=txn.category,
        merchant_id=txn.merchant_id,
        merchant_lat=txn.merchant_lat,
        merchant_long=txn.merchant_long,
    )


def reference_features(
    txns: pd.DataFrame, index: Iterable[object] | None = None
) -> dict[object, FeatureVector]:
    """Compute features one transaction at a time through ``compute_features``.

    For each requested row the history handed to the definition is that card's rows in
    ``(event_ts, transaction_id)`` order: the 600 before it and, on purpose, the row itself
    and the few after it, which the definition must ignore.

    Args:
        txns: Transactions with the batch builder's input columns.
        index: Index labels to compute; all rows when ``None``.

    Returns:
        ``{index label: feature vector}``.
    """
    wanted = set(txns.index if index is None else index)
    out: dict[object, FeatureVector] = {}
    ordered = txns.sort_values(["card_id", "event_ts", "transaction_id"], kind="stable")
    for _, card_rows in ordered.groupby("card_id", sort=False):
        card_txns = to_transactions(card_rows)
        entries = [to_entry(txn) for txn in card_txns]
        for position, label in enumerate(card_rows.index):
            if label in wanted:
                history = entries[max(0, position - _PRIOR_ROWS) : position + 4]
                out[label] = compute_features(card_txns[position], history)
    return out


def mismatches(
    reference: dict[object, FeatureVector], batch: pd.DataFrame
) -> list[tuple[object, str, object, object]]:
    """List every ``(row, feature, reference value, batch value)`` that disagrees."""
    found: list[tuple[object, str, object, object]] = []
    for label, expected in reference.items():
        actual = batch.loc[label]
        for name in FEATURE_NAMES:
            if not values_match(name, expected[name], actual[name]):
                found.append((label, name, expected[name], actual[name]))
    return found


def values_match(name: str, expected: object, actual: object) -> bool:
    """Exact equality, except NaN equals NaN and transcendental features allow 1e-9."""
    if isinstance(expected, str):
        return expected == actual
    left, right = float(expected), float(actual)  # type: ignore[arg-type]
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if name in TRANSCENDENTAL:
        return math.isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return left == right


def history_of(entries: Sequence[HistoryEntry]) -> list[HistoryEntry]:
    """Sort entries the way the feature store keeps them: by time, then transaction id."""
    return sorted(entries, key=lambda entry: (entry.event_ts, entry.transaction_id))


def make_raw_sparkov(
    n: int = 200, *, seed: int = 0, start: str = "2019-01-01", span_days: int = 400
) -> pd.DataFrame:
    """Build a frame with the raw Sparkov columns, including its fake PII."""
    rng = np.random.default_rng(seed)
    cards = rng.choice([4_000_000_000_000_000 + value for value in range(9)], n)
    when = pd.Timestamp(start) + pd.to_timedelta(
        np.sort(rng.integers(0, span_days * 86_400, n)), unit="s"
    )
    return pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n),
            "trans_date_trans_time": when.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": cards,
            "merchant": [f"fraud_Shop {value}" for value in rng.integers(0, 30, n)],
            "category": rng.choice(CATEGORIES[:-1], n),
            "amt": np.round(rng.uniform(1, 500, n), 2),
            "first": "Jane",
            "last": "Doe",
            "gender": rng.choice(["F", "M"], n),
            "street": "1 Main Street",
            "city": "Springfield",
            "state": "NC",
            "zip": 28_654,
            "lat": 36.0788,
            "long": -81.1781,
            "city_pop": 3_495,
            "job": "Engineer",
            "dob": "1988-03-09",
            "trans_num": [f"{value:032x}" for value in rng.permutation(n) + seed * 1_000_000],
            "unix_time": 1_325_376_018 + np.arange(n),
            "merch_lat": np.round(rng.uniform(35, 37, n), 6),
            "merch_long": np.round(rng.uniform(-82, -80, n), 6),
            "is_fraud": (rng.random(n) < 0.05).astype(np.int64),
        }
    )
