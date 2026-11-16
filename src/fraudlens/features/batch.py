"""Vectorised feature builder for training.

This module computes, for every row of a transaction table at once, exactly what
``definitions.compute_features`` computes for one transaction. It defines no feature of
its own: every rule here (closed-left windows, ordering by ``(event_ts, transaction_id)``,
the 30-day and 500-entry history cap, integer-cent arithmetic) is the definition's rule,
and ``tests/parity`` fails on any row where the two disagree.

How it works: rows are sorted by ``(card, event_ts, transaction_id)``, so a card's history
is the contiguous run of rows just before each row. Each window is then a half-open range
of row positions ``[lo, i)``, found by binary search on a combined ``(card, time)`` key.
Sums come from differences of integer prefix sums, which are exact.
"""

import logging
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraudlens.common.logging import configure_logging
from fraudlens.common.paths import TEST_FEATURES_PATH, TRAINVAL_FEATURES_PATH, TRANSACTIONS_PATH
from fraudlens.features.definitions import (
    CATEGORIES,
    DAY_S,
    FEATURE_NAMES,
    HISTORY_MAX_ENTRIES,
    HISTORY_WINDOW_S,
    HOUR_S,
    NIGHT_END_HOUR,
    NIGHT_START_HOUR,
    WINDOW_1H_S,
    WINDOW_7D_S,
    WINDOW_24H_S,
)
from fraudlens.features.geo import haversine_km_array

logger = logging.getLogger(__name__)

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

INPUT_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "card_id",
    "event_ts",
    "amount",
    "category",
    "merchant_id",
    "merchant_lat",
    "merchant_long",
    "home_lat",
    "home_long",
    "city_pop",
    "cardholder_age",
)
"""Columns ``build_features`` reads. ``event_ts`` is whole Unix seconds (UTC)."""

KEY_COLUMNS: tuple[str, ...] = ("transaction_id", "card_id", "event_ts", "is_fraud")
"""Non-feature columns carried into the feature tables."""

_INT64_SAFE = 2.0**62


def build_features(txns: pd.DataFrame) -> pd.DataFrame:
    """Compute every model feature for every row of ``txns``.

    Args:
        txns: Transactions with at least ``INPUT_COLUMNS``. Rows may be in any order and
            the table may hold many cards. A row's history is every other row of the same
            card that is earlier in ``(event_ts, transaction_id)`` order, so the table must
            contain the 30 days before any row whose features matter. Categories must
            already be normalised to ``definitions.CATEGORIES``.

    Returns:
        A frame with the same index and row order as ``txns`` and one column per name in
        ``FEATURE_NAMES``. Counts and flags are ``int64``, ``category`` is categorical
        over ``CATEGORIES``, everything else is ``float64`` with NaN where the definition
        returns NaN.

    Raises:
        ValueError: if a column is missing, a transaction id repeats, a category is not
            in ``CATEGORIES``, or amounts are too large for exact integer statistics.
    """
    missing = [column for column in INPUT_COLUMNS if column not in txns.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    if txns["transaction_id"].duplicated().any():
        raise ValueError("transaction_id must be unique")
    unknown = sorted(set(txns["category"].unique()) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"categories not normalised: {unknown}")

    n = len(txns)
    card_code = pd.factorize(txns["card_id"])[0].astype(np.int64)
    keys = pd.DataFrame(
        {
            "card": card_code,
            "ts": txns["event_ts"].to_numpy(np.int64),
            "tid": txns["transaction_id"].to_numpy(),
        }
    )
    order: IntArray = keys.sort_values(["card", "ts", "tid"], kind="stable").index.to_numpy()

    def sorted_floats(column: str) -> FloatArray:
        values: FloatArray = txns[column].to_numpy(np.float64)[order]
        return values

    card = card_code[order]
    ts: IntArray = keys["ts"].to_numpy()[order]
    amount = sorted_floats("amount")
    cents: IntArray = np.rint(amount * 100.0).astype(np.int64)
    merchant_lat = sorted_floats("merchant_lat")
    merchant_long = sorted_floats("merchant_long")
    idx = np.arange(n, dtype=np.int64)

    if n and float((cents.astype(np.float64) ** 2).sum()) >= _INT64_SAFE:
        raise ValueError("amounts are too large for exact integer window statistics")

    # --- Window bounds: features for row i read rows [lo, i). ---
    if n:
        rel = ts - ts.min() + HISTORY_WINDOW_S
        key = card * (int(rel.max()) + 1) + rel
    else:
        key = idx
    cap = np.maximum(idx - HISTORY_MAX_ENTRIES, 0)

    def lower_bound(window_s: int) -> IntArray:
        by_time = np.searchsorted(key, key - window_s, side="left").astype(np.int64)
        bound: IntArray = np.maximum(by_time, cap)
        return bound

    lo_1h = lower_bound(WINDOW_1H_S)
    lo_24h = lower_bound(WINDOW_24H_S)
    lo_7d = lower_bound(WINDOW_7D_S)
    lo_30d = lower_bound(HISTORY_WINDOW_S)
    n_prior = idx - lo_30d

    # --- Exact window sums from integer prefix sums. ---
    prefix = np.concatenate(([0], np.cumsum(cents)))
    prefix_sq = np.concatenate(([0], np.cumsum(cents * cents)))
    sum_cents = prefix[idx] - prefix[lo_30d]
    sum_sq_cents = prefix_sq[idx] - prefix_sq[lo_30d]
    sum_24h_cents = prefix[idx] - prefix[lo_24h]
    if n and (
        float(sum_cents.max()) ** 2 >= _INT64_SAFE
        or float(n_prior.max()) * float(sum_sq_cents.max()) >= _INT64_SAFE
    ):
        raise ValueError("amounts are too large for exact integer window statistics")

    # --- Spending pattern: the same IEEE operations, in the same order, as the definition. ---
    this_cents = cents.astype(np.float64)
    has_one = n_prior >= 1
    has_two = n_prior >= 2
    nan = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_cents = np.where(has_one, sum_cents.astype(np.float64) / n_prior, np.nan)
        numerator = n_prior * sum_sq_cents - sum_cents * sum_cents
        denominator = (n_prior * (n_prior - 1)).astype(np.float64)
        std_cents = np.where(has_two, np.sqrt(numerator.astype(np.float64) / denominator), np.nan)
        ratio = np.where(has_one & (mean_cents > 0.0), this_cents / mean_cents, np.nan)
        zscore = np.where(
            has_two & (std_cents > 0.0), (this_cents - mean_cents) / std_cents, np.nan
        )
    max_cents = _window_max(cents, lo_30d)
    max_30d = np.where(has_one, max_cents.astype(np.float64) / 100.0, np.nan)

    # --- Previous transaction: recency, distance, implied speed. ---
    prev = np.maximum(idx - 1, 0)
    secs_since_prev = np.where(has_one, (ts - ts[prev]).astype(np.float64), np.nan)
    dist_prev = np.where(
        has_one,
        haversine_km_array(merchant_lat[prev], merchant_long[prev], merchant_lat, merchant_long)
        if n
        else nan,
        np.nan,
    )
    with np.errstate(invalid="ignore"):
        speed_prev = dist_prev / (np.maximum(secs_since_prev, 1.0) / 3600.0)

    # --- Novelty: position of the previous row with the same card and merchant/category. ---
    merchant_code = pd.factorize(txns["merchant_id"])[0].astype(np.int64)[order]
    category_code = pd.factorize(txns["category"])[0].astype(np.int64)[order]
    prev_same_merchant = _previous_same(card, merchant_code)
    prev_same_category = _previous_same(card, category_code)

    hour = (ts % DAY_S) // HOUR_S
    day_of_week = (ts // DAY_S + 3) % 7

    columns: dict[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]] = {
        "amount": amount,
        "log_amount": np.log1p(amount),
        "hour": hour,
        "day_of_week": day_of_week,
        "is_night": ((hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR)).astype(np.int64),
        "is_weekend": (day_of_week >= 5).astype(np.int64),
        "amt_mean_30d": mean_cents / 100.0,
        "amt_std_30d": std_cents / 100.0,
        "amt_max_30d": max_30d,
        "amt_ratio_30d": ratio,
        "amt_zscore_30d": zscore,
        "txn_count_1h": idx - lo_1h,
        "txn_count_24h": idx - lo_24h,
        "txn_count_7d": idx - lo_7d,
        "amt_sum_24h": sum_24h_cents.astype(np.float64) / 100.0,
        "secs_since_prev": secs_since_prev,
        "dist_home_km": haversine_km_array(
            sorted_floats("home_lat"), sorted_floats("home_long"), merchant_lat, merchant_long
        ),
        "dist_prev_km": dist_prev,
        "speed_prev_kmh": speed_prev,
        "is_new_merchant_30d": (prev_same_merchant < lo_30d).astype(np.int64),
        "is_new_category_30d": (prev_same_category < lo_30d).astype(np.int64),
        "n_merchants_24h": _count_distinct(prev_same_merchant, lo_24h),
        "n_categories_24h": _count_distinct(prev_same_category, lo_24h),
        "cardholder_age": sorted_floats("cardholder_age"),
        "log_city_pop": np.log1p(sorted_floats("city_pop")),
        "n_prior_30d": n_prior,
    }

    # Undo the sort so the output lines up with the input, row for row.
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = idx
    out = pd.DataFrame({name: values[inverse] for name, values in columns.items()})
    out["category"] = pd.Categorical(txns["category"].to_numpy(), categories=list(CATEGORIES))
    out = out[list(FEATURE_NAMES)]
    out.index = txns.index
    return out


def _window_max(values: IntArray, lo: IntArray) -> IntArray:
    """Return ``max(values[lo[i]:i])`` for every row ``i``, or 0 where the window is empty.

    Walks back one position at a time over the rows whose window still reaches that far.
    Total work is the sum of the window lengths.
    """
    n = len(values)
    out = np.zeros(n, dtype=np.int64)
    length = np.arange(n, dtype=np.int64) - lo
    active = np.flatnonzero(length >= 1)
    step = 1
    while active.size:
        out[active] = np.maximum(out[active], values[active - step])
        step += 1
        active = active[length[active] >= step]
    return out


def _previous_same(card: IntArray, code: IntArray) -> IntArray:
    """For each row, the position of the previous row with the same card and code, or -1.

    Rows must already be in ``(card, event_ts, transaction_id)`` order.
    """
    position = pd.Series(np.arange(len(card), dtype=np.int64))
    group = card * (int(code.max()) + 1 if len(code) else 1) + code
    previous: IntArray = position.groupby(group).shift(1).fillna(-1).to_numpy(np.int64)
    return previous


def _count_distinct(previous_same: IntArray, lo: IntArray) -> IntArray:
    """Count distinct codes among rows ``[lo[i], i)`` for every row ``i``.

    A row ``j`` in the window is the first of its code there exactly when the previous row
    with the same code lies before the window, so the count is the number of such rows.
    """
    n = len(lo)
    out = np.zeros(n, dtype=np.int64)
    length = np.arange(n, dtype=np.int64) - lo
    active = np.flatnonzero(length >= 1)
    step = 1
    while active.size:
        out[active] += previous_same[active - step] < lo[active]
        step += 1
        active = active[length[active] >= step]
    return out


def write_feature_tables(
    transactions_path: Path = TRANSACTIONS_PATH,
    trainval_path: Path = TRAINVAL_FEATURES_PATH,
    test_path: Path = TEST_FEATURES_PATH,
) -> dict[str, int]:
    """Build features for every ingested transaction and write one table per partition.

    Features are computed over train and test rows together, because the first 30 days of
    the test period need history from the end of the training period. Features use only
    earlier transactions and never a label, so this leaks nothing. The rows are then
    written to separate files so that the test partition can be read in one place only.

    Args:
        transactions_path: PII-free transactions written by ``fraudlens.data.ingest``.
        trainval_path: Output for the ``train`` partition (training and validation rows).
        test_path: Output for the ``test`` partition.

    Returns:
        ``{partition: row count}``.
    """
    txns = pd.read_parquet(transactions_path)
    started = time.perf_counter()
    features = build_features(txns)
    logger.info(
        "features built",
        extra={"rows": len(txns), "seconds": round(time.perf_counter() - started, 1)},
    )
    table = pd.concat([txns[list(KEY_COLUMNS)], features], axis=1)

    counts: dict[str, int] = {}
    for partition, path in (("train", trainval_path), ("test", test_path)):
        rows = table[txns["partition"] == partition]
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path, index=False)
        counts[partition] = len(rows)
        logger.info("feature table written", extra={"path": str(path), "rows": len(rows)})
    return counts


def main() -> None:
    """Entry point for ``python -m fraudlens.features.batch``."""
    configure_logging()
    write_feature_tables()


if __name__ == "__main__":  # pragma: no cover
    main()
