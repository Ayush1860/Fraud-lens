"""Population stability index and the drift report."""

import math

import numpy as np
import pytest

from fraudlens.monitoring.drift import (
    CYCLICAL,
    MIN_SEGMENT_ROWS,
    drift_report,
)
from fraudlens.monitoring.psi import (
    ALARM,
    STABLE,
    psi,
    psi_between,
    shares_for_reference,
    status,
)

NUMERIC = {
    "kind": "numeric",
    "inner_edges": [1.0, 2.0],
    "proportions": [0.3, 0.3, 0.3],
    "missing": 0.1,
}
CATEGORICAL = {
    "kind": "categorical",
    "categories": ["a", "b"],
    "proportions": [0.5, 0.5],
    "missing": 0.0,
}


def test_psi_is_zero_for_identical_shares_and_symmetric_in_sign() -> None:
    shares = np.array([0.2, 0.3, 0.5])

    assert psi(shares, shares) == 0.0
    assert psi(shares, np.array([0.5, 0.3, 0.2])) == pytest.approx(
        psi(np.array([0.5, 0.3, 0.2]), shares)
    )


def test_psi_survives_empty_bins() -> None:
    value = psi(np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.5, 0.5]))

    assert math.isfinite(value)
    assert value > ALARM


def test_live_values_are_binned_like_the_reference() -> None:
    # side="right": a value equal to an edge goes to the bin above it, as in training.
    values = np.array([0.5, 1.0, 1.5, 2.0, 3.0, np.nan])

    shares = shares_for_reference(values, NUMERIC)

    np.testing.assert_allclose(shares, [1 / 6, 2 / 6, 2 / 6, 1 / 6])


def test_categorical_values_are_binned_by_category() -> None:
    shares = shares_for_reference(np.array(["a", "a", "b", "zzz"], dtype=object), CATEGORICAL)

    np.testing.assert_allclose(shares, [0.5, 0.25, 0.0])


def test_psi_between_windows_uses_the_earlier_windows_deciles() -> None:
    rng = np.random.default_rng(0)
    base = rng.lognormal(3.5, 0.6, 2_000)

    same = psi_between(base, rng.lognormal(3.5, 0.6, 2_000))
    shifted = psi_between(base, rng.lognormal(3.5, 0.6, 2_000) * 1.5)

    assert same < STABLE
    assert shifted > ALARM


def test_status_bands() -> None:
    assert status(0.05) == "stable"
    assert status(0.15) == "watch"
    assert status(0.3) == "alarm"
    assert status(None) == "unknown"


def rows(
    n: int, *, seed: int, factor: float = 1.0, category: str = "shop"
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    amounts = rng.lognormal(3.5, 0.6, n)
    categories = rng.choice(["shop", "food"], n)
    return [
        {
            "amount": float(a * (factor if c == category else 1.0)),
            "category": str(c),
            "day_of_week": 2,
        }
        for a, c in zip(amounts, categories, strict=True)
    ]


REFERENCE = {
    "features": {
        "amount": {
            "kind": "numeric",
            "inner_edges": np.quantile(
                np.random.default_rng(9).lognormal(3.5, 0.6, 50_000), np.linspace(0, 1, 11)[1:-1]
            ).tolist(),
            "proportions": [0.1] * 10,
            "missing": 0.0,
        },
        "category": {
            "kind": "categorical",
            "categories": ["food", "shop"],
            "proportions": [0.5, 0.5],
            "missing": 0.0,
        },
        "day_of_week": {
            "kind": "numeric",
            "inner_edges": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "proportions": [1 / 7] * 7,
            "missing": 0.0,
        },
    }
}
VALIDATION_MIX = {"approve": 0.99, "review": 0.005, "block": 0.005}


def test_a_clean_window_raises_no_alarm() -> None:
    report = drift_report(
        rows(4_000, seed=1),
        rows(4_000, seed=2),
        ["approve"] * 3_960 + ["review"] * 20 + ["block"] * 20,
        reference=REFERENCE,
        validation_mix=VALIDATION_MIX,
    )

    assert report["alarms"] == []
    by_feature = {f["feature"]: f for f in report["features"]}
    assert by_feature["amount"]["status"] == "stable"
    # One day of week in the window: drifted, but never an alarm, because the window is
    # shorter than the weekly cycle.
    assert by_feature["day_of_week"]["status"] == "alarm"
    assert by_feature["day_of_week"]["cyclical"] is True
    assert {s["category"] for s in report["segments"]} == {"shop", "food"}
    assert report["decision_mix"]["status"] == "stable"


def test_inflated_amounts_in_one_category_raise_a_segment_alarm() -> None:
    report = drift_report(
        rows(4_000, seed=1, factor=1.5),
        rows(4_000, seed=2),
        ["approve"] * 4_000,
        reference=REFERENCE,
        validation_mix=VALIDATION_MIX,
    )

    assert {"kind": "segment", "name": "amount in shop"} in [
        {"kind": a["kind"], "name": a["name"]} for a in report["alarms"]
    ]
    segments = {s["category"]: s for s in report["segments"]}
    assert segments["shop"]["status"] == "alarm"
    assert segments["food"]["status"] == "stable"


def test_small_segments_and_empty_windows_are_not_judged() -> None:
    few = rows(MIN_SEGMENT_ROWS, seed=1)  # about half per category: too few
    report = drift_report(few, few, [], reference=REFERENCE, validation_mix=VALIDATION_MIX)

    assert all(s["psi"] is None and s["status"] == "unknown" for s in report["segments"])
    assert report["decision_mix"]["psi"] is None
    empty = drift_report([], [], [], reference=REFERENCE, validation_mix=VALIDATION_MIX)
    assert empty["window"] == 0
    assert empty["alarms"] == []


def test_a_shifted_decision_mix_raises_an_alarm() -> None:
    report = drift_report(
        rows(1_000, seed=1),
        [],
        ["approve"] * 800 + ["review"] * 150 + ["block"] * 50,
        reference=REFERENCE,
        validation_mix=VALIDATION_MIX,
    )

    assert report["decision_mix"]["status"] == "alarm"
    assert "decision mix" in [a["name"] for a in report["alarms"]]


def test_calendar_features_are_marked_cyclical() -> None:
    assert {"day_of_week", "is_weekend"} == set(CYCLICAL)


def test_segments_are_ordered_by_psi_with_unjudged_ones_last() -> None:
    latest = rows(4_000, seed=1, factor=1.5) + [{"amount": 10.0, "category": "rare"}] * 5
    previous = rows(4_000, seed=2) + [{"amount": 10.0, "category": "rare"}] * 5

    report = drift_report(latest, previous, [], reference=REFERENCE, validation_mix=VALIDATION_MIX)

    assert [s["category"] for s in report["segments"]] == ["shop", "food", "rare"]
