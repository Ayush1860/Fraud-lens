"""Card numbers become keyed tokens that cannot be reversed without the key."""

import hashlib
import hmac
import re

import pytest

from fraudlens.common.hashing import card_token

KEY = b"unit-test-key"
CARD = 4_111_111_111_111_111


def test_token_is_a_truncated_hmac_sha256_of_the_card_number() -> None:
    expected = hmac.new(KEY, b"4111111111111111", hashlib.sha256).hexdigest()[:16]

    assert card_token(CARD, KEY) == f"c_{expected}"


def test_token_has_a_fixed_shape_and_hides_the_card_number() -> None:
    token = card_token(CARD, KEY)

    assert re.fullmatch(r"c_[0-9a-f]{16}", token)
    assert "4111" not in token


def test_same_card_and_key_always_give_the_same_token() -> None:
    assert card_token(CARD, KEY) == card_token("4111111111111111", KEY)


def test_token_depends_on_the_key() -> None:
    assert card_token(CARD, KEY) != card_token(CARD, b"another-key")


def test_token_is_not_a_plain_hash() -> None:
    plain = hashlib.sha256(b"4111111111111111").hexdigest()[:16]

    assert card_token(CARD, KEY) != f"c_{plain}"


def test_different_cards_give_different_tokens() -> None:
    tokens = {card_token(CARD + offset, KEY) for offset in range(2_000)}

    assert len(tokens) == 2_000


def test_an_empty_key_is_refused() -> None:
    with pytest.raises(ValueError, match="key"):
        card_token(CARD, b"")
