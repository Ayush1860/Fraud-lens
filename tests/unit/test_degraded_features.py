"""Features and reason wording when the card's history cannot be read."""

import math

import pytest

from fraudlens.common.types import Transaction
from fraudlens.explain.reason_codes import load_templates, render
from fraudlens.features.definitions import (
    FEATURE_NAMES,
    HISTORY_FEATURES,
    compute_features,
    transaction_only_features,
)

TXN = Transaction(
    transaction_id="t1",
    card_id="c_0123456789abcdef",
    event_ts=1_590_000_000,
    amount=812.40,
    category="shopping_net",
    merchant_id="m_1",
    merchant_lat=40.1,
    merchant_long=-74.2,
    home_lat=40.3,
    home_long=-74.0,
    city_pop=12_000.0,
    cardholder_age=47.0,
)


def test_history_features_are_unknown_not_zero() -> None:
    degraded = transaction_only_features(TXN)
    empty_history = compute_features(TXN, ())

    assert list(degraded) == list(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        if name in HISTORY_FEATURES:
            assert math.isnan(float(degraded[name])), name
        else:
            assert degraded[name] == empty_history[name], name


def test_history_features_are_model_features() -> None:
    assert set(HISTORY_FEATURES) < set(FEATURE_NAMES)


@pytest.mark.parametrize("group", list(load_templates().groups))
def test_every_group_words_or_skips_without_history(group: str) -> None:
    text = render(group, transaction_only_features(TXN), load_templates())

    assert text is None or "nan" not in text.lower()


def test_unknown_history_is_not_worded_as_pace_or_novelty() -> None:
    features = transaction_only_features(TXN)
    templates = load_templates()

    assert render("R02_VELOCITY", features, templates) is None
    assert render("R04_NOVELTY", features, templates) is None
    assert render("R01_AMOUNT_VS_PATTERN", features, templates) == "Amount of $812.40"
