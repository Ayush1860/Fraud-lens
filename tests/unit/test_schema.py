"""Raw data is rejected unless it has exactly the shape the pipeline was built against."""

import pandas as pd
import pandera.errors
import pytest

from fraudlens.data.schema import (
    SPARKOV_ROWS,
    ULB_FRAUDS,
    ULB_ROWS,
    RowCountError,
    validate_raw_sparkov,
    validate_raw_ulb,
)
from helpers import make_raw_sparkov


def test_published_row_counts_are_pinned() -> None:
    assert SPARKOV_ROWS == {"train": 1_296_675, "test": 555_719}
    assert (ULB_ROWS, ULB_FRAUDS) == (284_807, 492)


def test_a_well_formed_frame_passes() -> None:
    raw = make_raw_sparkov(200)

    assert validate_raw_sparkov(raw, expected_rows=200) is raw


def test_wrong_row_count_is_rejected() -> None:
    with pytest.raises(RowCountError, match="expected 201 rows, found 200"):
        validate_raw_sparkov(make_raw_sparkov(200), expected_rows=201)


@pytest.mark.parametrize("column", ["cc_num", "trans_num", "amt", "merch_lat", "is_fraud", "dob"])
def test_missing_column_is_rejected(column: str) -> None:
    raw = make_raw_sparkov(50).drop(columns=[column])

    with pytest.raises(pandera.errors.SchemaErrors, match=column):
        validate_raw_sparkov(raw, expected_rows=50)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("amt", 0.0),
        ("amt", -5.0),
        ("lat", 91.0),
        ("merch_long", -181.0),
        ("is_fraud", 2),
        ("category", "crypto_atm"),
        ("gender", "X"),
        ("merchant", "Shop without the prefix"),
        ("trans_date_trans_time", "01/02/2019 10:00"),
        ("dob", "March 1988"),
        ("city_pop", -1),
    ],
)
def test_out_of_range_value_is_rejected(column: str, bad_value: object) -> None:
    raw = make_raw_sparkov(50)
    raw.loc[3, column] = bad_value

    with pytest.raises(pandera.errors.SchemaErrors, match=column):
        validate_raw_sparkov(raw, expected_rows=50)


def test_duplicate_transaction_id_is_rejected() -> None:
    raw = make_raw_sparkov(50)
    raw.loc[1, "trans_num"] = raw.loc[0, "trans_num"]

    with pytest.raises(pandera.errors.SchemaErrors, match="trans_num"):
        validate_raw_sparkov(raw, expected_rows=50)


def test_null_in_a_key_column_is_rejected() -> None:
    raw = make_raw_sparkov(50)
    raw.loc[2, "trans_num"] = None

    with pytest.raises(pandera.errors.SchemaErrors, match="trans_num"):
        validate_raw_sparkov(raw, expected_rows=50)


def make_raw_ulb(n: int = 30) -> pd.DataFrame:
    columns: dict[str, object] = {"Time": [float(value) for value in range(n)]}
    for component in range(1, 29):
        columns[f"V{component}"] = 0.1 * component
    columns["Amount"] = 12.5
    columns["Class"] = [1] * 2 + [0] * (n - 2)
    return pd.DataFrame(columns)


def test_ulb_frame_passes() -> None:
    raw = make_raw_ulb()

    assert validate_raw_ulb(raw, expected_rows=30, expected_frauds=2) is raw


def test_ulb_wrong_fraud_count_is_rejected() -> None:
    with pytest.raises(RowCountError, match="frauds"):
        validate_raw_ulb(make_raw_ulb(), expected_rows=30, expected_frauds=3)


def test_ulb_missing_component_is_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaErrors, match="V14"):
        validate_raw_ulb(make_raw_ulb().drop(columns=["V14"]), expected_rows=30, expected_frauds=2)
