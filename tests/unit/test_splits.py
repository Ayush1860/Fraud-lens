"""Splits are by time only, with a gap, and no fold ever trains on its own future."""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fraudlens.data.splits import (
    GAP_S,
    TimeOrderError,
    assert_time_ordered,
    assign_roles,
    prequential_folds,
    split_bounds,
)

DAY = 86_400
START = 1_546_300_800  # 2019-01-01T00:00:00Z
VALID = ["valid_early", "valid_late"]


def hourly(days: int) -> npt.NDArray[np.int64]:
    return START + np.arange(days * 24, dtype=np.int64) * 3_600


def test_roles_cover_every_row_in_time_order() -> None:
    ts = hourly(400)

    roles = assign_roles(ts, split_bounds(ts))

    assert set(roles) == {"train", "gap", "valid_early", "valid_late"}
    order = {"train": 0, "gap": 1, "valid_early": 2, "valid_late": 3}
    ranks = np.array([order[role] for role in roles])
    assert (np.diff(ranks) >= 0).all()


def test_validation_is_the_last_quarter_and_training_stops_a_week_before() -> None:
    ts = hourly(400)

    bounds = split_bounds(ts)
    roles = assign_roles(ts, bounds)

    assert bounds.valid_start_ts == START + 300 * DAY
    assert bounds.train_end_ts == bounds.valid_start_ts - 7 * DAY
    assert GAP_S == 7 * DAY
    assert ts[roles == "train"].max() < bounds.train_end_ts
    assert ts[roles == "valid_early"].min() == bounds.valid_start_ts
    assert (roles == "gap").sum() == 7 * 24


def test_validation_is_split_sixty_forty_by_time() -> None:
    ts = hourly(400)

    roles = assign_roles(ts, split_bounds(ts))

    assert (roles == "valid_early").sum() == 60 * 24
    assert (roles == "valid_late").sum() == 40 * 24
    assert ts[roles == "valid_early"].max() < ts[roles == "valid_late"].min()


def test_latest_training_row_is_earlier_than_the_first_validation_row() -> None:
    ts = hourly(400)
    roles = assign_roles(ts, split_bounds(ts))
    train, valid = ts[roles == "train"], ts[np.isin(roles, VALID)]

    assert_time_ordered(train, valid)
    assert train.max() + GAP_S <= valid.min()


def test_split_does_not_depend_on_row_order() -> None:
    ts = hourly(400)
    shuffled = np.random.default_rng(0).permutation(ts)

    assert split_bounds(shuffled) == split_bounds(ts)


def test_rows_sharing_the_boundary_second_all_go_to_validation() -> None:
    ts = np.sort(np.concatenate([hourly(400), np.full(50, START + 300 * DAY)]))

    bounds = split_bounds(ts)
    roles = assign_roles(ts, bounds)

    assert set(roles[ts == bounds.valid_start_ts]) == {"valid_early"}


def test_assert_time_ordered_rejects_overlap_and_empty_sides() -> None:
    with pytest.raises(TimeOrderError):
        assert_time_ordered(np.array([1, 2, 10]), np.array([10, 11]))
    with pytest.raises(TimeOrderError):
        assert_time_ordered(np.array([], dtype=np.int64), np.array([10, 11]))


def test_prequential_folds_shift_one_block_back_each_time() -> None:
    ts = hourly(400)
    bounds = split_bounds(ts)

    folds = prequential_folds(
        ts, valid_start_ts=bounds.valid_start_ts, valid_end_ts=bounds.calib_start_ts
    )

    assert len(folds) == 3
    block = bounds.calib_start_ts - bounds.valid_start_ts
    for number, fold in enumerate(folds):
        assert fold.valid_start_ts == bounds.valid_start_ts - number * block
        assert fold.valid_end_ts == bounds.calib_start_ts - number * block
        assert fold.train_end_ts == fold.valid_start_ts - GAP_S
        assert_time_ordered(ts[fold.train], ts[fold.valid])
        assert not (fold.train & fold.valid).any()
        assert ts[fold.valid].min() >= fold.valid_start_ts
        assert ts[fold.valid].max() < fold.valid_end_ts
    # The most recent fold is exactly the train / early-validation split.
    roles = assign_roles(ts, bounds)
    assert (folds[0].valid == (roles == "valid_early")).all()
    assert (folds[0].train == (roles == "train")).all()


def test_no_fold_reads_the_late_validation_slice() -> None:
    ts = hourly(400)
    bounds = split_bounds(ts)
    late = assign_roles(ts, bounds) == "valid_late"

    for fold in prequential_folds(
        ts, valid_start_ts=bounds.valid_start_ts, valid_end_ts=bounds.calib_start_ts
    ):
        assert not (fold.train & late).any()
        assert not (fold.valid & late).any()


def test_too_many_folds_for_the_history_is_an_error() -> None:
    ts = hourly(400)
    bounds = split_bounds(ts)

    with pytest.raises(ValueError, match="fold"):
        prequential_folds(
            ts,
            valid_start_ts=bounds.valid_start_ts,
            valid_end_ts=bounds.calib_start_ts,
            n_folds=6,
        )


def test_empty_input_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        split_bounds(np.array([], dtype=np.int64))


@given(st.lists(st.integers(START, START + 700 * DAY), min_size=200, max_size=400, unique=True))
def test_training_always_ends_a_gap_before_validation(times: list[int]) -> None:
    ts = np.array(times, dtype=np.int64)
    roles = assign_roles(ts, split_bounds(ts))

    train, valid = ts[roles == "train"], ts[np.isin(roles, VALID)]
    assert len(valid) >= len(ts) // 4
    if len(train):
        assert train.max() + GAP_S <= valid.min()
