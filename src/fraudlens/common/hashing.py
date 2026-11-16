"""Keyed tokenisation of card numbers.

A card number has roughly 10^15 possible values once the issuer prefix and check digit
are fixed, few enough to enumerate. A plain hash can therefore be reversed by hashing
every candidate. An HMAC cannot, unless the attacker also holds the key.
"""

import hashlib
import hmac

TOKEN_PREFIX = "c_"  # noqa: S105 - a label, not a credential
TOKEN_HEX_CHARS = 16
"""Hex characters kept from the digest: 64 bits, so collisions are negligible."""


def card_token(card_number: int | str, key: bytes) -> str:
    """Return the ``card_id`` token for a card number.

    Args:
        card_number: The raw card number. It is used only inside this call.
        key: Secret HMAC key (``FRAUDLENS_CARD_HMAC_KEY``). The same key must be used by
            ingestion and by the replay client, or their tokens will not match.

    Returns:
        ``"c_"`` followed by the first 16 hex characters of HMAC-SHA256(key, card number).

    Raises:
        ValueError: if ``key`` is empty.
    """
    if not key:
        raise ValueError("HMAC key must not be empty")
    digest = hmac.new(key, str(card_number).encode("ascii"), hashlib.sha256).hexdigest()
    return f"{TOKEN_PREFIX}{digest[:TOKEN_HEX_CHARS]}"
