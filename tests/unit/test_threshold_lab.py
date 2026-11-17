"""Pricing threshold pairs from the exported curve, as the console's threshold lab shows them."""

import math
from pathlib import Path

import pytest

from fraudlens.training.bundle import ModelBundle
from fraudlens.training.thresholds import price_curve


@pytest.fixture(scope="module")
def bundle(toy_bundle_dir: Path) -> ModelBundle:
    return ModelBundle.load(toy_bundle_dir)


def test_every_ordered_pair_is_priced(bundle: ModelBundle) -> None:
    curve = bundle.documents["threshold_curve.json"]
    k = len(curve["points"])

    priced = price_curve(curve)

    assert len(priced) == k * (k + 1) // 2
    assert all(pair["review"] <= pair["block"] for pair in priced)
    assert all(0.0 <= pair["review_share"] <= 1.0 for pair in priced)


def test_the_released_pair_costs_what_training_recorded(bundle: ModelBundle) -> None:
    curve = bundle.documents["threshold_curve.json"]
    thresholds = [p["threshold"] for p in curve["points"]]
    review = thresholds.index(bundle.thresholds.t_review)
    block = thresholds.index(bundle.thresholds.t_block)

    priced = {(p["review"], p["block"]): p for p in price_curve(curve)}
    released = priced[review, block]

    recorded = bundle.documents["metrics.json"]["valid_late"]["expected_cost_per_1000"]
    assert math.isclose(released["cost_per_1000"], recorded, rel_tol=1e-9)
    assert released["within_capacity"]
    cheapest = min(
        (p for p in priced.values() if p["within_capacity"]), key=lambda p: p["cost_per_1000"]
    )
    assert cheapest["cost_per_1000"] == released["cost_per_1000"]
