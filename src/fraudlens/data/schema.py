"""Schemas the raw and cleaned data must satisfy before anything is built on them.

The raw schemas are the authority on what the downloaded files contain. If Kaggle ever
serves a different file, ingestion stops here rather than producing subtly different
features.
"""

from collections.abc import Mapping

import pandas as pd
import pandera.pandas as pa

from fraudlens.features.definitions import CATEGORIES, UNKNOWN_CATEGORY

SPARKOV_ROWS: Mapping[str, int] = {"train": 1_296_675, "test": 555_719}
"""Row counts of ``fraudTrain.csv`` and ``fraudTest.csv``."""

ULB_ROWS = 284_807
ULB_FRAUDS = 492

MERCHANT_PREFIX = "fraud_"
"""Constant prefix on every Sparkov merchant name. It carries no information about the
label because every row has it; the raw schema asserts that, and ingestion strips it."""

PARTITIONS: tuple[str, ...] = ("train", "test")

_TIMESTAMP = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
_DATE = r"^\d{4}-\d{2}-\d{2}$"
_LATITUDE = pa.Check.in_range(-90.0, 90.0)
_LONGITUDE = pa.Check.in_range(-180.0, 180.0)


class RowCountError(ValueError):
    """A dataset does not have the number of rows (or frauds) it is known to have."""


RAW_SPARKOV_SCHEMA = pa.DataFrameSchema(
    {
        "trans_date_trans_time": pa.Column(str, pa.Check.str_matches(_TIMESTAMP)),
        "cc_num": pa.Column(int, pa.Check.gt(0)),
        "merchant": pa.Column(str, pa.Check.str_startswith(MERCHANT_PREFIX)),
        "category": pa.Column(str, pa.Check.isin([c for c in CATEGORIES if c != UNKNOWN_CATEGORY])),
        "amt": pa.Column(float, pa.Check.gt(0.0)),
        "first": pa.Column(str),
        "last": pa.Column(str),
        "gender": pa.Column(str, pa.Check.isin(["F", "M"])),
        "street": pa.Column(str),
        "city": pa.Column(str),
        "state": pa.Column(str),
        "zip": pa.Column(int),
        "lat": pa.Column(float, _LATITUDE),
        "long": pa.Column(float, _LONGITUDE),
        "city_pop": pa.Column(int, pa.Check.ge(0)),
        "job": pa.Column(str),
        "dob": pa.Column(str, pa.Check.str_matches(_DATE)),
        "trans_num": pa.Column(str, unique=True),
        "unix_time": pa.Column(int),
        "merch_lat": pa.Column(float, _LATITUDE),
        "merch_long": pa.Column(float, _LONGITUDE),
        "is_fraud": pa.Column(int, pa.Check.isin([0, 1])),
    },
    # The CSVs also carry an unnamed row-number column, which is ignored.
    strict=False,
)
"""Sparkov as downloaded. No column may be null."""

CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        "transaction_id": pa.Column(str, unique=True),
        "card_id": pa.Column(str, pa.Check.str_matches(r"^c_[0-9a-f]{16}$")),
        "event_ts": pa.Column(int, pa.Check.gt(0)),
        "amount": pa.Column(float, pa.Check.gt(0.0)),
        "category": pa.Column(str, pa.Check.isin(list(CATEGORIES))),
        "merchant_id": pa.Column(str),
        "merchant_lat": pa.Column(float, _LATITUDE),
        "merchant_long": pa.Column(float, _LONGITUDE),
        "home_lat": pa.Column(float, _LATITUDE),
        "home_long": pa.Column(float, _LONGITUDE),
        "city_pop": pa.Column(int, pa.Check.ge(0)),
        "cardholder_age": pa.Column(int, pa.Check.in_range(0, 120)),
        "gender": pa.Column(str, pa.Check.isin(["F", "M"])),
        "is_fraud": pa.Column(int, pa.Check.isin([0, 1])),
        "partition": pa.Column(str, pa.Check.isin(list(PARTITIONS))),
    },
    # Exactly these columns: anything extra, such as a name or a street, is an error.
    strict=True,
)
"""Transactions after PII handling. ``gender`` is kept for the fairness check only."""

RAW_ULB_SCHEMA = pa.DataFrameSchema(
    {
        "Time": pa.Column(float, pa.Check.ge(0.0)),
        **{f"V{component}": pa.Column(float) for component in range(1, 29)},
        "Amount": pa.Column(float, pa.Check.ge(0.0)),
        "Class": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,
)
"""The ULB benchmark set as downloaded: two original columns and 28 PCA components."""


def validate_raw_sparkov(raw: pd.DataFrame, *, expected_rows: int) -> pd.DataFrame:
    """Check one raw Sparkov file and return it unchanged.

    Args:
        raw: Contents of ``fraudTrain.csv`` or ``fraudTest.csv``.
        expected_rows: The row count that file is known to have (``SPARKOV_ROWS``).

    Raises:
        RowCountError: if the row count differs.
        pandera.errors.SchemaErrors: listing every column or value that breaks the schema.
    """
    _check_count("rows", expected_rows, len(raw))
    RAW_SPARKOV_SCHEMA.validate(raw, lazy=True)
    return raw


def validate_clean(clean: pd.DataFrame) -> pd.DataFrame:
    """Check a PII-free transaction table and return it unchanged.

    Raises:
        pandera.errors.SchemaErrors: if a column is missing, extra or out of range.
    """
    CLEAN_SCHEMA.validate(clean, lazy=True)
    return clean


def validate_raw_ulb(
    raw: pd.DataFrame, *, expected_rows: int = ULB_ROWS, expected_frauds: int = ULB_FRAUDS
) -> pd.DataFrame:
    """Check the raw ULB file and return it unchanged.

    Raises:
        RowCountError: if the number of rows or of frauds differs from the published one.
        pandera.errors.SchemaErrors: if a column is missing, extra or out of range.
    """
    _check_count("rows", expected_rows, len(raw))
    RAW_ULB_SCHEMA.validate(raw, lazy=True)
    _check_count("frauds", expected_frauds, int(raw["Class"].sum()))
    return raw


def _check_count(what: str, expected: int, found: int) -> None:
    if found != expected:
        raise RowCountError(f"expected {expected} {what}, found {found}")
