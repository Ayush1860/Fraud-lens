"""Evaluation metrics, checked on arrays small enough to work out by hand."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fraudlens.training.metrics import (
    APPROVE,
    BLOCK,
    REVIEW,
    Costs,
    alert_metrics,
    average_precision,
    bootstrap_days,
    brier_score,
    daily_top_k,
    day_blocks,
    expected_cost_per_1000,
    percentile_interval,
    reliability_curve,
)

COSTS = Costs(review_cost=3.0, margin_rate=0.02, friction_cost=10.0)


def test_average_precision_of_a_perfect_ranking_is_one() -> None:
    assert average_precision(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_average_precision_by_hand() -> None:
    # Ranked by score: 1, 0, 1, 0. Precision at each hit is 1/1 and 2/3; AP is their mean.
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1])

    assert average_precision(y, score) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_of_a_model_that_flags_nothing_is_the_base_rate() -> None:
    y = np.array([0] * 99 + [1])

    assert average_precision(y, np.zeros(100)) == pytest.approx(0.01)


def test_daily_top_k_flags_a_share_of_each_day_rounded_up() -> None:
    day = np.array([0] * 10 + [1] * 4)
    score = np.concatenate([np.arange(10), np.arange(4)]).astype(float)

    flagged = daily_top_k(score, day, budget=0.2)

    # Day 0: ceil(0.2 * 10) = 2 highest scores. Day 1: ceil(0.2 * 4) = 1 highest score.
    assert flagged.tolist() == [False] * 8 + [True, True] + [False] * 3 + [True]


def test_daily_top_k_does_not_depend_on_row_order() -> None:
    rng = np.random.default_rng(0)
    day = rng.integers(0, 5, 300)
    score = rng.random(300)
    order = rng.permutation(300)

    flagged = daily_top_k(score, day, budget=0.05)

    assert (daily_top_k(score[order], day[order], budget=0.05) == flagged[order]).all()


def test_alert_metrics_by_hand() -> None:
    y = np.array([1, 1, 0, 0, 1])
    amount = np.array([100.0, 300.0, 50.0, 20.0, 600.0])
    flagged = np.array([True, False, True, False, True])

    metrics = alert_metrics(y, amount, flagged)

    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["dollar_recall"] == pytest.approx(700 / 1000)


def test_alert_metrics_with_no_fraud_or_no_alerts_are_nan_not_errors() -> None:
    nothing = alert_metrics(np.array([0, 0]), np.array([1.0, 2.0]), np.array([False, False]))

    assert np.isnan(nothing["recall"])
    assert np.isnan(nothing["precision"])
    assert np.isnan(nothing["dollar_recall"])


def test_brier_score_by_hand() -> None:
    assert brier_score(np.array([1, 0]), np.array([0.8, 0.3])) == pytest.approx(
        (0.2**2 + 0.3**2) / 2
    )


def test_reliability_curve_reports_mean_prediction_and_observed_rate_per_bin() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.05, 0.05, 0.05, 0.9, 0.9, 0.9])

    curve = reliability_curve(y, p, edges=(0.0, 0.5, 1.0))

    assert len(curve) == 2
    assert curve[0] == pytest.approx(
        {"lower": 0.0, "upper": 0.5, "count": 3, "mean_predicted": 0.05, "observed": 1 / 3}
    )
    assert curve[1] == pytest.approx(
        {"lower": 0.5, "upper": 1.0, "count": 3, "mean_predicted": 0.9, "observed": 2 / 3}
    )


def test_reliability_curve_skips_empty_bins_and_includes_probability_one() -> None:
    curve = reliability_curve(np.array([1, 0]), np.array([1.0, 0.0]), edges=(0.0, 0.5, 0.9, 1.0))

    assert [point["count"] for point in curve] == [1, 1]


def test_expected_cost_charges_each_outcome_what_it_costs() -> None:
    y = np.array([1, 0, 1, 0, 1, 0])
    amount = np.array([200.0, 50.0, 300.0, 80.0, 400.0, 1000.0])
    decision = np.array([APPROVE, APPROVE, REVIEW, REVIEW, BLOCK, BLOCK])

    # Missed fraud 200, two reviews at 3 each, one wrongly blocked 1000: 0.02 * 1000 + 10.
    total = 200.0 + 2 * 3.0 + (0.02 * 1000.0 + 10.0)
    assert expected_cost_per_1000(y, amount, decision, COSTS) == pytest.approx(total / 6 * 1000)


def test_approving_everything_costs_the_fraud_amount() -> None:
    y = np.array([1, 0, 0, 1])
    amount = np.array([10.0, 20.0, 30.0, 40.0])

    assert expected_cost_per_1000(y, amount, np.full(4, APPROVE), COSTS) == pytest.approx(
        50.0 / 4 * 1000
    )


def test_day_blocks_group_row_positions_by_day() -> None:
    blocks = day_blocks(np.array([7, 3, 7, 3, 9]))

    assert [block.tolist() for block in blocks] == [[1, 3], [0, 2], [4]]


def test_bootstrap_resamples_whole_days() -> None:
    day = np.repeat(np.arange(6), 4)
    blocks = day_blocks(day)

    for rows in bootstrap_days(blocks, n_boot=20, seed=0):
        counts = np.bincount(day[rows], minlength=6)
        assert len(rows) == len(day)
        assert set(counts) <= {0, 4, 8, 12, 16, 20, 24}  # every day appears whole or not at all


def test_bootstrap_is_reproducible_for_a_seed() -> None:
    blocks = day_blocks(np.repeat(np.arange(5), 3))

    first = [rows.tolist() for rows in bootstrap_days(blocks, n_boot=5, seed=42)]
    second = [rows.tolist() for rows in bootstrap_days(blocks, n_boot=5, seed=42)]

    assert first == second
    assert first != [rows.tolist() for rows in bootstrap_days(blocks, n_boot=5, seed=43)]


def test_percentile_interval_by_hand() -> None:
    low, high = percentile_interval(np.arange(1, 101, dtype=float), level=0.90)

    assert low == pytest.approx(5.95)
    assert high == pytest.approx(95.05)


def test_percentile_interval_ignores_nan_replicates() -> None:
    low, high = percentile_interval(np.array([1.0, 2.0, 3.0, np.nan]), level=0.5)

    assert (low, high) == (pytest.approx(1.5), pytest.approx(2.5))


@given(st.lists(st.tuples(st.booleans(), st.floats(0, 1)), min_size=2, max_size=60))
def test_brier_score_is_between_zero_and_one(rows: list[tuple[bool, float]]) -> None:
    y = np.array([int(label) for label, _ in rows])
    p = np.array([probability for _, probability in rows])

    assert 0.0 <= brier_score(y, p) <= 1.0
