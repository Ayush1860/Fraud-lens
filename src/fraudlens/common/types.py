"""Value types shared by the feature definition, the pipeline and the service."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """What the feature store keeps about one past transaction on a card.

    Attributes:
        transaction_id: Unique id; breaks ties between transactions in the same second.
        event_ts: When the transaction happened, in whole Unix seconds (UTC).
        amount_cents: Amount in integer cents, so window sums are exact.
        category: Merchant category, already normalised to a known value.
        merchant_id: Merchant identifier.
        merchant_lat: Merchant latitude in degrees, NaN if unknown.
        merchant_long: Merchant longitude in degrees, NaN if unknown.
    """

    transaction_id: str
    event_ts: int
    amount_cents: int
    category: str
    merchant_id: str
    merchant_lat: float
    merchant_long: float


@dataclass(frozen=True, slots=True)
class Transaction:
    """One card transaction to score.

    Attributes:
        transaction_id: Unique id, also the idempotency key.
        card_id: HMAC token of the card number; never the number itself.
        event_ts: When the transaction happened, in whole Unix seconds (UTC).
        amount: Amount in dollars, positive.
        category: Merchant category, already normalised to a known value.
        merchant_id: Merchant identifier.
        merchant_lat: Merchant latitude in degrees, NaN if unknown.
        merchant_long: Merchant longitude in degrees, NaN if unknown.
        home_lat: Cardholder home latitude in degrees, NaN if unknown.
        home_long: Cardholder home longitude in degrees, NaN if unknown.
        city_pop: Population of the cardholder's city, NaN if unknown.
        cardholder_age: Cardholder age in completed years at ``event_ts``, NaN if unknown.
    """

    transaction_id: str
    card_id: str
    event_ts: int
    amount: float
    category: str
    merchant_id: str
    merchant_lat: float
    merchant_long: float
    home_lat: float
    home_long: float
    city_pop: float
    cardholder_age: float
