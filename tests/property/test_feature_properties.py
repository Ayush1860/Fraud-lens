"""Properties that must hold for any card history, not just the examples we thought of."""

import math

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fraudlens.common.types import HistoryEntry, Transaction
from fraudlens.features.batch import build_features
from fraudlens.features.definitions import (
    CATEGORIES,
    FEATURE_NAMES,
    HISTORY_MAX_ENTRIES,
    HISTORY_WINDOW_S,
    FeatureVector,
    compute_features,
)
from helpers import history_of, mismatches, reference_features

NOW = 1_577_836_800  # 2020-01-01T00:00:00Z
DAY = 86_400

# Offsets cluster on the window edges, where off-by-one mistakes live.
EDGES = [0, 1, 3_599, 3_600, 3_601, DAY - 1, DAY, DAY + 1, 7 * DAY, 7 * DAY + 1, 30 * DAY]
seconds_ago = st.one_of(st.sampled_from(EDGES), st.integers(0, 45 * DAY))
ids = st.text(alphabet="abcdef0123456789", min_size=1, max_size=6)
cents = st.integers(1, 2_000_000)
categories = st.sampled_from(CATEGORIES[:4])
merchants = st.sampled_from(["m_1", "m_2", "m_3", "m_4"])
lats = st.floats(25, 48, allow_nan=False)
longs = st.floats(-122, -70, allow_nan=False)


@st.composite
def entries(draw: st.DrawFn, ago: st.SearchStrategy[int] = seconds_ago) -> HistoryEntry:
    return HistoryEntry(
        transaction_id=draw(ids),
        event_ts=NOW - draw(ago),
        amount_cents=draw(cents),
        category=draw(categories),
        merchant_id=draw(merchants),
        merchant_lat=draw(lats),
        merchant_long=draw(longs),
    )


@st.composite
def transactions(draw: st.DrawFn) -> Transaction:
    return Transaction(
        transaction_id=draw(ids),
        card_id="c_1",
        event_ts=NOW,
        amount=draw(cents) / 100.0,
        category=draw(categories),
        merchant_id=draw(merchants),
        merchant_lat=draw(lats),
        merchant_long=draw(longs),
        home_lat=draw(lats),
        home_long=draw(longs),
        city_pop=float(draw(st.integers(0, 5_000_000))),
        cardholder_age=float(draw(st.integers(18, 95))),
    )


def unique_by_id(items: list[HistoryEntry]) -> list[HistoryEntry]:
    return list({item.transaction_id: item for item in items}.values())


def same(left: FeatureVector, right: FeatureVector) -> bool:
    """Bitwise-equal feature vectors, treating NaN as equal to NaN."""
    return all(
        left[name] == right[name]
        or (
            isinstance(left[name], float)
            and isinstance(right[name], float)
            and math.isnan(float(left[name]))
            and math.isnan(float(right[name]))
        )
        for name in FEATURE_NAMES
    )


def is_earlier(entry: HistoryEntry, txn: Transaction) -> bool:
    return (entry.event_ts, entry.transaction_id) < (txn.event_ts, txn.transaction_id)


@given(txn=transactions(), past=st.lists(entries(), max_size=40), extra=st.lists(entries()))
def test_a_later_transaction_never_changes_earlier_features(
    txn: Transaction, past: list[HistoryEntry], extra: list[HistoryEntry]
) -> None:
    earlier = [entry for entry in unique_by_id(past) if is_earlier(entry, txn)]
    later = [
        HistoryEntry(
            transaction_id=f"later-{position}",
            event_ts=NOW + (NOW - item.event_ts),
            amount_cents=item.amount_cents,
            category=item.category,
            merchant_id=item.merchant_id,
            merchant_lat=item.merchant_lat,
            merchant_long=item.merchant_long,
        )
        for position, item in enumerate(extra)
    ]

    without = compute_features(txn, history_of(earlier))
    with_later = compute_features(txn, history_of(earlier + later))

    assert same(without, with_later)


@given(txn=transactions(), past=st.lists(entries(), max_size=40))
def test_features_read_only_strictly_earlier_rows(
    txn: Transaction, past: list[HistoryEntry]
) -> None:
    everything = unique_by_id(past)
    earlier = [entry for entry in everything if is_earlier(entry, txn)]

    assert same(
        compute_features(txn, history_of(everything)),
        compute_features(txn, history_of(earlier)),
    )


@given(
    txn=transactions(),
    recent=st.lists(entries(), max_size=40),
    stale=st.lists(entries(ago=st.integers(HISTORY_WINDOW_S + 1, 400 * DAY)), max_size=20),
)
def test_history_older_than_thirty_days_never_matters(
    txn: Transaction, recent: list[HistoryEntry], stale: list[HistoryEntry]
) -> None:
    kept = [entry for entry in unique_by_id(recent) if entry.event_ts >= NOW - HISTORY_WINDOW_S]
    old = [
        HistoryEntry(
            transaction_id=f"stale-{position}",
            event_ts=item.event_ts,
            amount_cents=item.amount_cents,
            category=item.category,
            merchant_id=item.merchant_id,
            merchant_lat=item.merchant_lat,
            merchant_long=item.merchant_long,
        )
        for position, item in enumerate(stale)
    ]

    assert same(
        compute_features(txn, history_of(kept + old)),
        compute_features(txn, history_of(kept)),
    )


@given(txn=transactions(), past=st.lists(entries(), max_size=60))
def test_counts_never_shrink_as_the_window_grows(
    txn: Transaction, past: list[HistoryEntry]
) -> None:
    features = compute_features(txn, history_of(unique_by_id(past)))

    assert (
        0
        <= int(features["txn_count_1h"])
        <= int(features["txn_count_24h"])
        <= int(features["txn_count_7d"])
        <= int(features["n_prior_30d"])
        <= HISTORY_MAX_ENTRIES
    )
    assert int(features["n_merchants_24h"]) <= int(features["txn_count_24h"])
    assert int(features["n_categories_24h"]) <= int(features["txn_count_24h"])


@given(txn=transactions(), past=st.lists(entries(), max_size=60))
def test_window_statistics_are_internally_consistent(
    txn: Transaction, past: list[HistoryEntry]
) -> None:
    features = compute_features(txn, history_of(unique_by_id(past)))

    if int(features["n_prior_30d"]) >= 1:
        assert float(features["amt_max_30d"]) >= float(features["amt_mean_30d"]) > 0
        assert float(features["secs_since_prev"]) >= 0
    if int(features["n_prior_30d"]) >= 2:
        assert float(features["amt_std_30d"]) >= 0
    assert float(features["amt_sum_24h"]) >= 0
    for name in ("dist_home_km",):
        assert 0 <= float(features[name]) <= 20_040  # half the Earth's circumference


@st.composite
def frames(draw: st.DrawFn) -> pd.DataFrame:
    n = draw(st.integers(1, 40))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    offsets = np.array([draw(seconds_ago) for _ in range(n)])
    cards = rng.integers(0, 3, n)
    return pd.DataFrame(
        {
            "transaction_id": [f"{value:04x}" for value in rng.permutation(n)],
            "card_id": [f"c_{value}" for value in cards],
            "event_ts": (NOW - offsets).astype(np.int64),
            "amount": rng.integers(1, 500_000, n) / 100.0,
            "category": rng.choice(CATEGORIES[:4], n),
            "merchant_id": rng.choice(["m_1", "m_2", "m_3"], n),
            "merchant_lat": rng.uniform(25, 48, n),
            "merchant_long": rng.uniform(-122, -70, n),
            "home_lat": rng.uniform(25, 48, n),
            "home_long": rng.uniform(-122, -70, n),
            "city_pop": rng.integers(0, 5_000_000, n),
            "cardholder_age": rng.integers(18, 95, n),
        }
    )


@settings(max_examples=150, deadline=None)
@given(txns=frames())
def test_batch_builder_equals_the_definition_on_any_frame(txns: pd.DataFrame) -> None:
    assert mismatches(reference_features(txns), build_features(txns)) == []
