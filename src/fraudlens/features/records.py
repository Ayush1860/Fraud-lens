"""Convert table rows to the value types ``compute_features`` takes.

Conversion only: no feature is computed here.
"""

from collections.abc import Mapping
from typing import Any

from fraudlens.common.types import HistoryEntry, Transaction
from fraudlens.features.definitions import amount_to_cents


def to_transaction(row: Mapping[str, Any], *, amount: float | None = None) -> Transaction:
    """Build a ``Transaction`` from a row with the ingested-transaction columns.

    Args:
        row: A row of ``data/interim/transactions.parquet`` (or any mapping with the same
            keys).
        amount: Replaces the row's amount, in dollars, for what-if checks.
    """
    return Transaction(
        transaction_id=str(row["transaction_id"]),
        card_id=str(row["card_id"]),
        event_ts=int(row["event_ts"]),
        amount=float(row["amount"] if amount is None else amount),
        category=str(row["category"]),
        merchant_id=str(row["merchant_id"]),
        merchant_lat=float(row["merchant_lat"]),
        merchant_long=float(row["merchant_long"]),
        home_lat=float(row["home_lat"]),
        home_long=float(row["home_long"]),
        city_pop=float(row["city_pop"]),
        cardholder_age=float(row["cardholder_age"]),
    )


def to_history_entry(row: Mapping[str, Any]) -> HistoryEntry:
    """Build the ``HistoryEntry`` the feature store keeps for a past transaction."""
    return HistoryEntry(
        transaction_id=str(row["transaction_id"]),
        event_ts=int(row["event_ts"]),
        amount_cents=amount_to_cents(float(row["amount"])),
        category=str(row["category"]),
        merchant_id=str(row["merchant_id"]),
        merchant_lat=float(row["merchant_lat"]),
        merchant_long=float(row["merchant_long"]),
    )


def history_entry(txn: Transaction) -> HistoryEntry:
    """Build the ``HistoryEntry`` the feature store keeps once ``txn`` has been scored."""
    return HistoryEntry(
        transaction_id=txn.transaction_id,
        event_ts=txn.event_ts,
        amount_cents=amount_to_cents(txn.amount),
        category=txn.category,
        merchant_id=txn.merchant_id,
        merchant_lat=txn.merchant_lat,
        merchant_long=txn.merchant_long,
    )
