"""No personal field survives ingestion, and card numbers leave only as keyed tokens."""

import pandas as pd
import pandera.errors
import pytest

from fraudlens.common.hashing import card_token
from fraudlens.data.pii import CLEAN_COLUMNS, PII_COLUMNS, clean_sparkov
from fraudlens.data.schema import validate_clean
from fraudlens.features.batch import INPUT_COLUMNS
from fraudlens.features.definitions import FEATURE_NAMES
from helpers import make_raw_sparkov

KEY = b"unit-test-key"


def clean(raw: pd.DataFrame, partition: str = "train") -> pd.DataFrame:
    return clean_sparkov(raw, partition=partition, hmac_key=KEY)


def test_no_pii_column_survives_ingestion() -> None:
    out = clean(make_raw_sparkov(200))

    assert set(PII_COLUMNS) >= {"first", "last", "street", "cc_num", "dob"}
    assert set(PII_COLUMNS).isdisjoint(out.columns)
    assert tuple(out.columns) == CLEAN_COLUMNS


def test_no_pii_value_survives_ingestion() -> None:
    raw = make_raw_sparkov(200)
    raw["first"] = "Zebedee"
    raw["last"] = "Quixotesmith"
    raw["street"] = "742 Evergreen Terrace"

    text = clean(raw).to_csv()

    for value in ("Zebedee", "Quixotesmith", "Evergreen", "1988-03-09"):
        assert value not in text
    for card_number in raw["cc_num"].unique():
        assert str(card_number) not in text


def test_card_numbers_become_hmac_tokens() -> None:
    raw = make_raw_sparkov(200)

    out = clean(raw)

    assert out["card_id"].str.fullmatch(r"c_[0-9a-f]{16}").all()
    assert out["card_id"].nunique() == raw["cc_num"].nunique()
    assert out["card_id"].iloc[0] == card_token(int(raw["cc_num"].iloc[0]), KEY)


def test_merchant_prefix_is_stripped() -> None:
    out = clean(make_raw_sparkov(200))

    assert not out["merchant_id"].str.startswith("fraud_").any()
    assert out["merchant_id"].str.startswith("Shop ").all()


def test_event_time_is_the_transaction_timestamp_read_as_utc() -> None:
    raw = make_raw_sparkov(10)
    raw.loc[0, "trans_date_trans_time"] = "2020-08-14 03:12:44"

    assert clean(raw)["event_ts"].iloc[0] == 1_597_374_764


@pytest.mark.parametrize(
    ("dob", "when", "age"),
    [
        ("1988-03-09", "2019-03-08 23:59:59", 30),  # the day before the birthday
        ("1988-03-09", "2019-03-09 00:00:00", 31),  # on the birthday
        ("1988-03-09", "2020-12-31 12:00:00", 32),
        ("2000-02-29", "2019-02-28 10:00:00", 18),  # leap-day birthday, non-leap year
        ("2000-02-29", "2019-03-01 10:00:00", 19),
    ],
)
def test_age_is_completed_years_at_the_transaction(dob: str, when: str, age: int) -> None:
    raw = make_raw_sparkov(10)
    raw["dob"] = dob
    raw.loc[0, "trans_date_trans_time"] = when

    assert clean(raw)["cardholder_age"].iloc[0] == age


def test_output_feeds_the_feature_builder_and_passes_the_clean_schema() -> None:
    out = clean(make_raw_sparkov(200), partition="test")

    assert set(INPUT_COLUMNS) <= set(out.columns)
    assert (out["partition"] == "test").all()
    validate_clean(out)


def test_clean_schema_rejects_a_smuggled_pii_column() -> None:
    out = clean(make_raw_sparkov(50)).assign(first="Jane")

    with pytest.raises(pandera.errors.SchemaErrors, match="first"):
        validate_clean(out)


def test_clean_schema_rejects_a_raw_card_number_as_card_id() -> None:
    out = clean(make_raw_sparkov(50))
    out["card_id"] = "4000000000000001"

    with pytest.raises(pandera.errors.SchemaErrors, match="card_id"):
        validate_clean(out)


def test_gender_is_kept_for_the_fairness_check_but_is_not_a_feature() -> None:
    out = clean(make_raw_sparkov(50))

    assert set(out["gender"]) <= {"F", "M"}
    assert "gender" not in FEATURE_NAMES


def test_unknown_partition_is_refused() -> None:
    with pytest.raises(ValueError, match="partition"):
        clean(make_raw_sparkov(10), partition="holdout")
