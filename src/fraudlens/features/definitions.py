"""The single definition of every model feature.

``compute_features(txn, history)`` is what the scoring service calls for each request.
The vectorised training builder in ``batch.py`` must reproduce it exactly, and a parity
test holds the two together. No other module may define feature logic.

Time semantics, for every windowed feature:

* A window of length ``W`` ending at a transaction with time ``t`` holds the card's
  *earlier* transactions whose time is at least ``t - W``. The old edge is included
  (closed on the left); the transaction itself never is.
* "Earlier" means earlier in ``(event_ts, transaction_id)`` order, so two transactions in
  the same second are ordered by transaction id, identically in training and serving.
* History is the card's most recent ``HISTORY_MAX_ENTRIES`` earlier transactions within
  ``HISTORY_WINDOW_S``. Anything older, or beyond the cap, does not exist for any feature.

Amounts inside a window are summed as integer cents, so sums, means and variances are
exact and do not depend on the order of addition. That is what lets the batch builder
match this function bit for bit instead of to a tolerance.
"""

import math
from collections.abc import Sequence

from fraudlens.common.types import HistoryEntry, Transaction
from fraudlens.features.geo import haversine_km

FeatureVector = dict[str, float | int | str]

HOUR_S = 3_600
DAY_S = 86_400

WINDOW_1H_S = HOUR_S
WINDOW_24H_S = DAY_S
WINDOW_7D_S = 7 * DAY_S
HISTORY_WINDOW_S = 30 * DAY_S
"""Length of the longest window, in seconds. History older than this is never read."""

HISTORY_MAX_ENTRIES = 500
"""Most earlier transactions any feature may read; the feature store trims to this."""

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 6
"""Night is 22:00 up to, but not including, 06:00 UTC."""

UNKNOWN_CATEGORY = "unknown"

CATEGORIES: tuple[str, ...] = (
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
    UNKNOWN_CATEGORY,
)
"""Every category the model knows, in the fixed order used for its categorical encoding."""

_KNOWN_CATEGORIES = frozenset(CATEGORIES)

FEATURE_NAMES: tuple[str, ...] = (
    # Transaction
    "amount",
    "log_amount",
    "category",
    "hour",
    "day_of_week",
    "is_night",
    "is_weekend",
    # Spending pattern over 30 days
    "amt_mean_30d",
    "amt_std_30d",
    "amt_max_30d",
    "amt_ratio_30d",
    "amt_zscore_30d",
    # Velocity
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "amt_sum_24h",
    "secs_since_prev",
    # Geography
    "dist_home_km",
    "dist_prev_km",
    "speed_prev_kmh",
    # Novelty
    "is_new_merchant_30d",
    "is_new_category_30d",
    "n_merchants_24h",
    "n_categories_24h",
    # Context
    "cardholder_age",
    "log_city_pop",
    "n_prior_30d",
)
"""Model features in column order. ``category`` is a string; all others are numeric."""


HISTORY_FEATURES: tuple[str, ...] = (
    "amt_mean_30d",
    "amt_std_30d",
    "amt_max_30d",
    "amt_ratio_30d",
    "amt_zscore_30d",
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "amt_sum_24h",
    "secs_since_prev",
    "dist_prev_km",
    "speed_prev_kmh",
    "is_new_merchant_30d",
    "is_new_category_30d",
    "n_merchants_24h",
    "n_categories_24h",
    "n_prior_30d",
)
"""Features that read the card's history. All others need only the transaction itself."""


def normalise_category(category: str) -> str:
    """Return ``category`` if the model knows it, otherwise ``"unknown"``."""
    return category if category in _KNOWN_CATEGORIES else UNKNOWN_CATEGORY


def amount_to_cents(amount: float) -> int:
    """Convert an amount in dollars to integer cents, rounding half to even."""
    return round(amount * 100.0)


def compute_features(txn: Transaction, history: Sequence[HistoryEntry]) -> FeatureVector:
    """Compute every model feature for one transaction.

    Args:
        txn: The transaction being scored. Its category must already be normalised.
        history: Transactions on the same card, sorted ascending by
            ``(event_ts, transaction_id)``. Entries that are not strictly earlier than
            ``txn`` in that order are ignored, as is anything outside the history window
            or beyond the history cap, so passing extra entries never changes the result.

    Returns:
        A dict keyed by ``FEATURE_NAMES``, in that order. Amounts are in dollars, times in
        seconds, distances in kilometres and speeds in km/h. Counts and flags are ints.
        A feature that cannot be computed (no history, a missing coordinate) is NaN.
    """
    now = txn.event_ts
    this = (now, txn.transaction_id)
    oldest_allowed = now - HISTORY_WINDOW_S

    # The card's usable history, newest first.
    window: list[HistoryEntry] = []
    for entry in reversed(history):
        if (entry.event_ts, entry.transaction_id) >= this:
            continue
        if entry.event_ts < oldest_allowed or len(window) == HISTORY_MAX_ENTRIES:
            break
        window.append(entry)

    n_prior = len(window)
    sum_cents = 0
    sum_sq_cents = 0
    max_cents = 0
    count_1h = count_24h = count_7d = 0
    sum_24h_cents = 0
    merchants_24h: set[str] = set()
    categories_24h: set[str] = set()
    new_merchant = 1
    new_category = 1

    for entry in window:
        cents = entry.amount_cents
        sum_cents += cents
        sum_sq_cents += cents * cents
        if cents > max_cents:
            max_cents = cents
        age_s = now - entry.event_ts
        if age_s <= WINDOW_7D_S:
            count_7d += 1
            if age_s <= WINDOW_24H_S:
                count_24h += 1
                sum_24h_cents += cents
                merchants_24h.add(entry.merchant_id)
                categories_24h.add(entry.category)
                if age_s <= WINDOW_1H_S:
                    count_1h += 1
        if entry.merchant_id == txn.merchant_id:
            new_merchant = 0
        if entry.category == txn.category:
            new_category = 0

    # Spending pattern. Every operand is converted to float before it is used, so the
    # arithmetic is the same sequence of IEEE operations as the numpy version in batch.py.
    this_cents = float(amount_to_cents(txn.amount))
    mean_30d = std_30d = max_30d = ratio_30d = zscore_30d = math.nan
    if n_prior >= 1:
        mean_cents = float(sum_cents) / float(n_prior)
        mean_30d = mean_cents / 100.0
        max_30d = float(max_cents) / 100.0
        if mean_cents > 0.0:
            ratio_30d = this_cents / mean_cents
    if n_prior >= 2:
        # n * sum(x^2) - sum(x)^2 is an exact, non-negative integer.
        numerator = n_prior * sum_sq_cents - sum_cents * sum_cents
        std_cents = math.sqrt(float(numerator) / float(n_prior * (n_prior - 1)))
        std_30d = std_cents / 100.0
        if std_cents > 0.0:
            zscore_30d = (this_cents - mean_cents) / std_cents

    # Previous transaction: recency, distance and implied speed.
    secs_since_prev = dist_prev_km = speed_prev_kmh = math.nan
    if n_prior >= 1:
        previous = window[0]
        secs_since_prev = float(now - previous.event_ts)
        dist_prev_km = haversine_km(
            previous.merchant_lat, previous.merchant_long, txn.merchant_lat, txn.merchant_long
        )
        # A one-second floor keeps same-second purchases finite: 10 km apart is 36,000 km/h.
        speed_prev_kmh = dist_prev_km / (max(secs_since_prev, 1.0) / 3600.0)

    hour = (now % DAY_S) // HOUR_S
    day_of_week = (now // DAY_S + 3) % 7  # 1970-01-01 was a Thursday; Monday is 0

    return {
        "amount": txn.amount,
        "log_amount": math.log1p(txn.amount),
        "category": txn.category,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_night": int(hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR),
        "is_weekend": int(day_of_week >= 5),
        "amt_mean_30d": mean_30d,
        "amt_std_30d": std_30d,
        "amt_max_30d": max_30d,
        "amt_ratio_30d": ratio_30d,
        "amt_zscore_30d": zscore_30d,
        "txn_count_1h": count_1h,
        "txn_count_24h": count_24h,
        "txn_count_7d": count_7d,
        "amt_sum_24h": float(sum_24h_cents) / 100.0,
        "secs_since_prev": secs_since_prev,
        "dist_home_km": haversine_km(
            txn.home_lat, txn.home_long, txn.merchant_lat, txn.merchant_long
        ),
        "dist_prev_km": dist_prev_km,
        "speed_prev_kmh": speed_prev_kmh,
        "is_new_merchant_30d": new_merchant,
        "is_new_category_30d": new_category,
        "n_merchants_24h": len(merchants_24h),
        "n_categories_24h": len(categories_24h),
        "cardholder_age": float(txn.cardholder_age),
        "log_city_pop": math.log1p(txn.city_pop),
        "n_prior_30d": n_prior,
    }


def transaction_only_features(txn: Transaction) -> FeatureVector:
    """Compute features when the card's history cannot be read (the degraded path).

    Args:
        txn: The transaction being scored. Its category must already be normalised.

    Returns:
        The same keys as ``compute_features``. Every feature in ``HISTORY_FEATURES`` is NaN:
        the history is unknown, which is not the same as empty. A count of zero would tell
        the model "first purchase on this card"; NaN sends it down the missing-value branch
        each tree learned in training instead. Transaction-only features are unchanged.
    """
    features = compute_features(txn, ())
    for name in HISTORY_FEATURES:
        features[name] = math.nan
    return features
