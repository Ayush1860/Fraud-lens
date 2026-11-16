"""The feature definition, checked against values worked out by hand."""

import math
from dataclasses import replace

import pytest

from fraudlens.common.types import HistoryEntry, Transaction
from fraudlens.features.definitions import (
    CATEGORIES,
    FEATURE_NAMES,
    HISTORY_MAX_ENTRIES,
    amount_to_cents,
    compute_features,
    normalise_category,
)
from helpers import history_of

HOUR = 3_600
DAY = 86_400
T0 = 1_597_374_764  # 2020-08-14T03:12:44Z, a Friday

TXN = Transaction(
    transaction_id="t-now",
    card_id="c_1",
    event_ts=T0,
    amount=100.0,
    category="shopping_net",
    merchant_id="m_1",
    merchant_lat=41.0,
    merchant_long=-74.0,
    home_lat=40.0,
    home_long=-74.0,
    city_pop=12_335,
    cardholder_age=47,
)


def entry(
    seconds_ago: int,
    amount: float,
    *,
    tid: str | None = None,
    category: str = "grocery_pos",
    merchant_id: str = "m_2",
    lat: float = 40.0,
    long: float = -74.0,
) -> HistoryEntry:
    return HistoryEntry(
        transaction_id=tid or f"t-{seconds_ago}-{amount}",
        event_ts=T0 - seconds_ago,
        amount_cents=amount_to_cents(amount),
        category=category,
        merchant_id=merchant_id,
        merchant_lat=lat,
        merchant_long=long,
    )


def test_feature_names_are_unique_and_match_the_output_keys() -> None:
    features = compute_features(TXN, [])

    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert tuple(features) == FEATURE_NAMES


def test_excluded_fields_never_become_features() -> None:
    banned = {"gender", "first", "last", "street", "job", "card_id", "merchant_id", "dob"}

    assert banned.isdisjoint(FEATURE_NAMES)


def test_transaction_features() -> None:
    features = compute_features(TXN, [])

    assert features["amount"] == 100.0
    assert features["log_amount"] == pytest.approx(math.log(101.0))
    assert features["category"] == "shopping_net"
    assert features["hour"] == 3
    assert features["day_of_week"] == 4  # Monday is 0, so Friday is 4
    assert features["is_night"] == 1
    assert features["is_weekend"] == 0


@pytest.mark.parametrize(
    ("event_ts", "hour", "day_of_week", "is_night", "is_weekend"),
    [
        (0, 0, 3, 1, 0),  # 1970-01-01T00:00:00Z, Thursday
        (1_546_300_800 + 5 * DAY + 6 * HOUR, 6, 6, 0, 1),  # Sunday 2019-01-06 06:00
        (1_546_300_800 + 4 * DAY + 22 * HOUR, 22, 5, 1, 1),  # Saturday 2019-01-05 22:00
        (1_546_300_800 + 21 * HOUR + 3_599, 21, 1, 0, 0),  # Tuesday 2019-01-01 21:59:59
    ],
)
def test_calendar_features_use_utc(
    event_ts: int, hour: int, day_of_week: int, is_night: int, is_weekend: int
) -> None:
    features = compute_features(replace(TXN, event_ts=event_ts), [])

    assert features["hour"] == hour
    assert features["day_of_week"] == day_of_week
    assert features["is_night"] == is_night
    assert features["is_weekend"] == is_weekend


def test_empty_history_gives_zero_counts_and_missing_pattern() -> None:
    features = compute_features(TXN, [])

    for name in ("txn_count_1h", "txn_count_24h", "txn_count_7d", "n_prior_30d"):
        assert features[name] == 0
    assert features["amt_sum_24h"] == 0.0
    assert features["n_merchants_24h"] == 0
    assert features["n_categories_24h"] == 0
    assert features["is_new_merchant_30d"] == 1
    assert features["is_new_category_30d"] == 1
    for name in (
        "amt_mean_30d",
        "amt_std_30d",
        "amt_max_30d",
        "amt_ratio_30d",
        "amt_zscore_30d",
        "secs_since_prev",
        "dist_prev_km",
        "speed_prev_kmh",
    ):
        assert math.isnan(float(features[name])), name


def test_spending_pattern_over_thirty_days() -> None:
    history = history_of([entry(2 * DAY, 10.0), entry(5 * DAY, 20.0), entry(20 * DAY, 30.0)])

    features = compute_features(TXN, history)

    assert features["n_prior_30d"] == 3
    assert features["amt_mean_30d"] == 20.0
    assert features["amt_std_30d"] == 10.0  # sample standard deviation of 10, 20, 30
    assert features["amt_max_30d"] == 30.0
    assert features["amt_ratio_30d"] == 5.0  # 100 / 20
    assert features["amt_zscore_30d"] == 8.0  # (100 - 20) / 10


def test_single_prior_transaction_has_a_mean_but_no_spread() -> None:
    features = compute_features(TXN, [entry(DAY, 25.0)])

    assert features["amt_mean_30d"] == 25.0
    assert features["amt_ratio_30d"] == 4.0
    assert math.isnan(float(features["amt_std_30d"]))
    assert math.isnan(float(features["amt_zscore_30d"]))


def test_identical_prior_amounts_give_zero_spread_and_no_zscore() -> None:
    history = history_of([entry(day * DAY, 42.50) for day in range(1, 8)])

    features = compute_features(TXN, history)

    assert features["amt_std_30d"] == 0.0
    assert math.isnan(float(features["amt_zscore_30d"]))


def test_velocity_counts_and_sum() -> None:
    history = history_of(
        [
            entry(10 * 60, 5.0),
            entry(50 * 60, 7.0),
            entry(3 * HOUR, 11.0),
            entry(2 * DAY, 13.0),
            entry(10 * DAY, 17.0),
        ]
    )

    features = compute_features(TXN, history)

    assert features["txn_count_1h"] == 2
    assert features["txn_count_24h"] == 3
    assert features["txn_count_7d"] == 4
    assert features["n_prior_30d"] == 5
    assert features["amt_sum_24h"] == 23.0
    assert features["secs_since_prev"] == 600.0


@pytest.mark.parametrize(
    ("window", "feature"),
    [
        (HOUR, "txn_count_1h"),
        (DAY, "txn_count_24h"),
        (7 * DAY, "txn_count_7d"),
        (30 * DAY, "n_prior_30d"),
    ],
)
def test_windows_are_closed_on_the_left(window: int, feature: str) -> None:
    on_the_edge = compute_features(TXN, [entry(window, 5.0)])
    one_second_older = compute_features(TXN, [entry(window + 1, 5.0)])

    assert on_the_edge[feature] == 1
    assert one_second_older[feature] == 0


def test_current_transaction_is_never_in_its_own_history() -> None:
    itself = HistoryEntry(
        transaction_id=TXN.transaction_id,
        event_ts=TXN.event_ts,
        amount_cents=amount_to_cents(TXN.amount),
        category=TXN.category,
        merchant_id=TXN.merchant_id,
        merchant_lat=TXN.merchant_lat,
        merchant_long=TXN.merchant_long,
    )

    assert compute_features(TXN, [itself]) == pytest.approx(compute_features(TXN, []), nan_ok=True)


def test_same_second_transactions_are_ordered_by_transaction_id() -> None:
    earlier_id = entry(0, 5.0, tid="t-aaa")  # "t-aaa" < "t-now"
    later_id = entry(0, 7.0, tid="t-zzz")  # "t-zzz" > "t-now"

    features = compute_features(TXN, history_of([earlier_id, later_id]))

    assert features["n_prior_30d"] == 1
    assert features["amt_sum_24h"] == 5.0
    assert features["secs_since_prev"] == 0.0


def test_later_transactions_in_history_are_ignored() -> None:
    past = [entry(DAY, 10.0), entry(2 * DAY, 20.0)]
    future = [entry(-60, 999.0), entry(-DAY, 999.0)]

    assert compute_features(TXN, history_of(past + future)) == pytest.approx(
        compute_features(TXN, history_of(past)), nan_ok=True
    )


def test_geography() -> None:
    previous = entry(40 * 60, 5.0, lat=41.0, long=-75.0)

    features = compute_features(TXN, [previous])

    # One degree of latitude is 111.195 km on a sphere of radius 6371.0088 km.
    assert features["dist_home_km"] == pytest.approx(111.195, abs=0.001)
    # One degree of longitude at latitude 41 is 111.195 * cos(41 deg) = 83.92 km.
    assert features["dist_prev_km"] == pytest.approx(83.92, abs=0.01)
    assert features["speed_prev_kmh"] == pytest.approx(83.92 / (40 / 60), abs=0.02)


def test_speed_uses_a_one_second_floor_for_same_second_purchases() -> None:
    previous = entry(0, 5.0, tid="t-aaa", lat=41.0, long=-75.0)

    features = compute_features(TXN, [previous])

    assert features["speed_prev_kmh"] == pytest.approx(float(features["dist_prev_km"]) * 3_600)


def test_novelty() -> None:
    history = history_of(
        [
            entry(HOUR, 5.0, merchant_id="m_1", category="grocery_pos"),
            entry(2 * HOUR, 5.0, merchant_id="m_2", category="grocery_pos"),
            entry(3 * HOUR, 5.0, merchant_id="m_2", category="travel"),
            entry(3 * DAY, 5.0, merchant_id="m_9", category="shopping_net"),
        ]
    )

    features = compute_features(TXN, history)

    assert features["is_new_merchant_30d"] == 0  # m_1 seen an hour ago
    assert features["is_new_category_30d"] == 0  # shopping_net seen three days ago
    assert features["n_merchants_24h"] == 2  # m_1 and m_2; m_9 is older than 24 hours
    assert features["n_categories_24h"] == 2  # grocery_pos and travel


def test_merchant_seen_only_before_the_window_counts_as_new() -> None:
    features = compute_features(TXN, [entry(31 * DAY, 5.0, merchant_id="m_1")])

    assert features["is_new_merchant_30d"] == 1


def test_context_features() -> None:
    features = compute_features(TXN, [])

    assert features["cardholder_age"] == 47.0
    assert features["log_city_pop"] == pytest.approx(math.log(12_336))


def test_missing_optional_fields_become_nan_not_errors() -> None:
    nan = float("nan")
    txn = replace(TXN, merchant_lat=nan, home_lat=nan, city_pop=nan, cardholder_age=nan)

    features = compute_features(txn, [entry(HOUR, 5.0)])

    for name in ("dist_home_km", "dist_prev_km", "speed_prev_kmh", "log_city_pop"):
        assert math.isnan(float(features[name])), name
    assert math.isnan(float(features["cardholder_age"]))
    assert features["txn_count_1h"] == 1


def test_history_is_capped_at_the_most_recent_entries() -> None:
    history = history_of(
        [entry(second, 1.0, tid=f"t-{second:05d}") for second in range(1, 601)]
        + [entry(700, 5_000.0, tid="t-big")]
    )

    features = compute_features(TXN, history)

    assert features["n_prior_30d"] == HISTORY_MAX_ENTRIES
    assert features["txn_count_1h"] == HISTORY_MAX_ENTRIES
    assert features["amt_max_30d"] == 1.0  # the large purchase fell outside the cap


@pytest.mark.parametrize(
    ("amount", "cents"),
    [(1.0, 100), (336.10, 33_610), (0.29, 29), (1.005, 100), (28_948.90, 2_894_890)],
)
def test_amount_to_cents_rounds_to_the_nearest_cent(amount: float, cents: int) -> None:
    assert amount_to_cents(amount) == cents


def test_unknown_categories_map_to_unknown() -> None:
    assert normalise_category("grocery_pos") == "grocery_pos"
    assert normalise_category("crypto_atm") == "unknown"
    assert CATEGORIES[-1] == "unknown"
