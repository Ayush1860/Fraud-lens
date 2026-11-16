"""PII handling at ingestion.

The Sparkov names, streets and card numbers are synthetic, and are handled as if real:

* Names, street, city, state, zip code and job are dropped. Nothing downstream needs them.
* The card number becomes ``card_id``, a keyed HMAC-SHA256 token.
* The date of birth becomes the cardholder's age at the transaction, then is dropped.
* ``gender`` is kept, for the offline fairness check only. It is never a model feature.

Nothing in this module logs a row or a value.
"""

import pandas as pd

from fraudlens.common.hashing import card_token
from fraudlens.data.schema import MERCHANT_PREFIX, PARTITIONS
from fraudlens.features.definitions import normalise_category

PII_COLUMNS: tuple[str, ...] = (
    "first",
    "last",
    "street",
    "cc_num",
    "dob",
    "city",
    "state",
    "zip",
    "job",
)
"""Raw columns that must not exist after ingestion. The first five are direct
identifiers; the rest are quasi-identifiers that no feature uses."""

CLEAN_COLUMNS: tuple[str, ...] = (
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
    "gender",
    "is_fraud",
    "partition",
)


def clean_sparkov(raw: pd.DataFrame, *, partition: str, hmac_key: bytes) -> pd.DataFrame:
    """Turn a validated raw Sparkov frame into PII-free transactions.

    Args:
        raw: A frame that has passed ``schema.validate_raw_sparkov``.
        partition: ``"train"`` or ``"test"``: which public file the rows came from.
        hmac_key: Secret key for the card token.

    Returns:
        A new frame with exactly ``CLEAN_COLUMNS``. ``event_ts`` is the transaction
        timestamp read as UTC, in whole Unix seconds. ``cardholder_age`` is in completed
        years at that moment. ``amount`` is in dollars.

    Raises:
        ValueError: if ``partition`` is not a known partition.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}; expected one of {PARTITIONS}")

    # Sparkov's own ``unix_time`` column is offset by about seven years from this
    # timestamp, so the readable timestamp is the one source of event time.
    when = pd.to_datetime(raw["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S")
    born = pd.to_datetime(raw["dob"], format="%Y-%m-%d")
    birthday_still_to_come = (when.dt.month * 100 + when.dt.day) < (
        born.dt.month * 100 + born.dt.day
    )
    age = when.dt.year - born.dt.year - birthday_still_to_come.astype("int64")

    tokens = {number: card_token(int(number), hmac_key) for number in raw["cc_num"].unique()}

    clean = pd.DataFrame(
        {
            "transaction_id": raw["trans_num"].astype("str"),
            "card_id": raw["cc_num"].map(tokens).astype("str"),
            "event_ts": when.astype("datetime64[s]").astype("int64"),
            "amount": raw["amt"].astype("float64"),
            "category": raw["category"].map(normalise_category).astype("str"),
            "merchant_id": raw["merchant"].str.removeprefix(MERCHANT_PREFIX).astype("str"),
            "merchant_lat": raw["merch_lat"].astype("float64"),
            "merchant_long": raw["merch_long"].astype("float64"),
            "home_lat": raw["lat"].astype("float64"),
            "home_long": raw["long"].astype("float64"),
            "city_pop": raw["city_pop"].astype("int64"),
            "cardholder_age": age.astype("int64"),
            "gender": raw["gender"].astype("str"),
            "is_fraud": raw["is_fraud"].astype("int64"),
            "partition": partition,
        }
    )
    return clean[list(CLEAN_COLUMNS)].reset_index(drop=True)
