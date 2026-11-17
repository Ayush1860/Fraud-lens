"""Request validation for the scoring endpoint."""

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from fraudlens.service.schemas import ScoreRequest

VALID: dict[str, Any] = {
    "transaction_id": "2da90c7d74bd46a0caf3777415b3ebd3",
    "card_id": "c_9f2e61b07a3c4d5e",
    "event_time": "2020-08-14T03:12:44.900Z",
    "amount": 336.10,
    "category": "shopping_net",
    "merchant_id": "m_4411",
    "merchant_lat": 41.02,
    "merchant_long": -74.91,
    "home_lat": 40.34,
    "home_long": -74.12,
    "city_pop": 12335,
    "cardholder_age": 47,
}


def test_a_valid_request_becomes_a_transaction() -> None:
    txn = ScoreRequest.model_validate(VALID).to_transaction()

    assert txn.event_ts == 1_597_374_764  # fractional seconds are dropped
    assert txn.amount == 336.10
    assert txn.city_pop == 12_335.0
    assert txn.category == "shopping_net"


def test_optional_fields_become_missing_values() -> None:
    body = {k: v for k, v in VALID.items() if k not in {"home_lat", "cardholder_age"}}

    txn = ScoreRequest.model_validate(body).to_transaction()

    assert math.isnan(txn.home_lat)
    assert math.isnan(txn.cardholder_age)


def test_an_unknown_category_is_scored_as_unknown() -> None:
    txn = ScoreRequest.model_validate({**VALID, "category": "crypto_atm"}).to_transaction()

    assert txn.category == "unknown"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("card_id", "4532015112830366"),  # a raw card number is never accepted
        ("card_id", "c_9F2E61B07A3C4D5E"),
        ("amount", 0),
        ("amount", -5),
        ("amount", float("inf")),
        ("merchant_lat", 91.0),
        ("home_long", -181.0),
        ("event_time", "2020-08-14T03:12:44"),  # no time zone
        ("transaction_id", "has spaces"),
        ("cardholder_age", 200),
        ("category", ""),
    ],
)
def test_invalid_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate({**VALID, field: value})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate({**VALID, "cc_num": "4532015112830366"})


def test_event_time_may_not_be_far_in_the_future() -> None:
    soon = datetime.now(UTC) + timedelta(minutes=4)
    later = datetime.now(UTC) + timedelta(minutes=6)

    ScoreRequest.model_validate({**VALID, "event_time": soon.isoformat()})
    with pytest.raises(ValidationError, match="future"):
        ScoreRequest.model_validate({**VALID, "event_time": later.isoformat()})
