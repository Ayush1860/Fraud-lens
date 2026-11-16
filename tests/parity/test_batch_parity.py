"""Training-serving parity: the batch builder must equal ``compute_features`` row for row.

Integer-derived features must be exactly equal. Features that pass through ``log`` or
trigonometric functions are compared to 1e-9, because numpy and ``math`` may differ in the
last bit. That tolerance is fixed; a mismatch is a bug in the builder, not a reason to
widen it.
"""

import numpy as np
import pandas as pd
import pytest

from fraudlens.common.paths import TRAINVAL_FEATURES_PATH, TRANSACTIONS_PATH
from fraudlens.features.batch import INPUT_COLUMNS, build_features
from fraudlens.features.definitions import CATEGORIES, FEATURE_NAMES
from helpers import make_transactions, mismatches, reference_features

SAMPLE_ROWS = 5_000


def test_parity_on_5000_synthetic_transactions() -> None:
    txns = make_transactions(n_cards=25, n_txns=20_000, seed=7, span_days=150)
    sample = txns.sample(SAMPLE_ROWS, random_state=7).index

    batch = build_features(txns)
    reference = reference_features(txns, sample)

    assert len(reference) == SAMPLE_ROWS
    assert mismatches(reference, batch) == []


def test_parity_when_the_history_cap_binds() -> None:
    # 3 cards and 6,000 transactions in 20 days: every 30-day window overflows 500 entries.
    txns = make_transactions(n_cards=3, n_txns=6_000, seed=11, span_days=20)
    sample = txns.sample(800, random_state=11).index

    batch = build_features(txns)

    assert batch["n_prior_30d"].max() == 500
    assert mismatches(reference_features(txns, sample), batch) == []


def test_output_is_aligned_to_the_input_index_whatever_its_order() -> None:
    txns = make_transactions(n_cards=5, n_txns=600, seed=3)
    shuffled = txns.sample(frac=1.0, random_state=3)

    in_order = build_features(txns)
    out_of_order = build_features(shuffled)

    assert out_of_order.index.equals(shuffled.index)
    pd.testing.assert_frame_equal(in_order, out_of_order.loc[in_order.index])


def test_output_columns_and_dtypes() -> None:
    batch = build_features(make_transactions(n_cards=4, n_txns=300, seed=5))

    assert tuple(batch.columns) == FEATURE_NAMES
    assert isinstance(batch["category"].dtype, pd.CategoricalDtype)
    assert tuple(batch["category"].cat.categories) == CATEGORIES
    for name in ("hour", "txn_count_1h", "is_new_merchant_30d", "n_prior_30d"):
        assert batch[name].dtype == np.int64, name
    for name in ("amount", "amt_mean_30d", "secs_since_prev", "dist_home_km"):
        assert batch[name].dtype == np.float64, name


def test_missing_input_column_is_reported_by_name() -> None:
    txns = make_transactions(n_cards=2, n_txns=50).drop(columns=["merchant_lat"])

    with pytest.raises(ValueError, match="merchant_lat"):
        build_features(txns)


def test_duplicate_transaction_ids_are_rejected() -> None:
    txns = make_transactions(n_cards=2, n_txns=50)
    txns.loc[1, "transaction_id"] = txns.loc[0, "transaction_id"]

    with pytest.raises(ValueError, match="transaction_id"):
        build_features(txns)


def test_amounts_too_large_for_exact_integer_statistics_are_rejected() -> None:
    txns = make_transactions(n_cards=1, n_txns=50)
    txns["amount"] = 5.0e9

    with pytest.raises(ValueError, match="exact"):
        build_features(txns)


def test_empty_input_gives_an_empty_frame_with_the_right_columns() -> None:
    batch = build_features(make_transactions(n_cards=2, n_txns=50).iloc[0:0])

    assert len(batch) == 0
    assert tuple(batch.columns) == FEATURE_NAMES


@pytest.mark.data
def test_parity_on_5000_real_transactions() -> None:
    if not (TRANSACTIONS_PATH.exists() and TRAINVAL_FEATURES_PATH.exists()):
        pytest.skip("real data not built; run `make data features`")

    txns = pd.read_parquet(
        TRANSACTIONS_PATH,
        columns=list(INPUT_COLUMNS),
        filters=[("partition", "==", "train")],
    )
    txns.index = pd.Index(txns["transaction_id"].to_numpy())
    batch = pd.read_parquet(TRAINVAL_FEATURES_PATH).set_index("transaction_id")
    sample = txns.sample(SAMPLE_ROWS, random_state=20_261_116).index

    # Only the sampled cards are needed to rebuild each sampled row's history.
    cards = txns.loc[sample, "card_id"].unique()
    reference = reference_features(txns[txns["card_id"].isin(cards)], sample)

    assert len(reference) == SAMPLE_ROWS
    assert mismatches(reference, batch) == []
