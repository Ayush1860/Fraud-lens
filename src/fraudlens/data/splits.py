"""Time-ordered splits. Nothing here shuffles, and nothing here reads a label.

The public training partition is cut by time into four consecutive pieces:

``train`` | ``gap`` (7 days, unused) | ``valid_early`` | ``valid_late``

* ``train`` fits models.
* ``gap`` is discarded. Fraud labels arrive late in practice, so a model deployed at time
  ``t`` cannot have trained on the days just before ``t``.
* ``valid_early`` drives tuning and early stopping.
* ``valid_late`` fits the calibrator and chooses thresholds, and is never used for tuning.

Every boundary is a timestamp, not a row number, so all rows in the same second land on
the same side.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

DAY_S = 86_400
GAP_S = 7 * DAY_S
"""Delay between the end of training and the start of validation, in seconds."""

TRAIN_FRACTION = 0.75
EARLY_VALID_FRACTION = 0.60


class TimeOrderError(AssertionError):
    """Training data is not strictly earlier than the data it is evaluated on."""


@dataclass(frozen=True, slots=True)
class SplitBounds:
    """Boundaries of the split, in Unix seconds (UTC).

    Attributes:
        train_end_ts: Training rows have ``event_ts < train_end_ts``.
        valid_start_ts: Validation rows have ``event_ts >= valid_start_ts``.
        calib_start_ts: Late-validation rows have ``event_ts >= calib_start_ts``.
    """

    train_end_ts: int
    valid_start_ts: int
    calib_start_ts: int


@dataclass(frozen=True, slots=True, eq=False)
class Fold:
    """One prequential fold: boolean row masks plus the boundaries that produced them.

    Training rows have ``event_ts < train_end_ts``; validation rows have
    ``valid_start_ts <= event_ts < valid_end_ts``.
    """

    train: BoolArray
    valid: BoolArray
    train_end_ts: int
    valid_start_ts: int
    valid_end_ts: int


def split_bounds(
    event_ts: IntArray,
    *,
    train_fraction: float = TRAIN_FRACTION,
    early_valid_fraction: float = EARLY_VALID_FRACTION,
    gap_s: int = GAP_S,
) -> SplitBounds:
    """Find the split boundaries for a set of event times.

    Validation starts at the timestamp of the row ``train_fraction`` of the way through
    the rows in time order, so it holds the last quarter of the rows by default. Training
    ends ``gap_s`` seconds before that. Validation is cut again at the timestamp
    ``early_valid_fraction`` of the way through its own rows.

    Args:
        event_ts: Event times in Unix seconds, in any order.
        train_fraction: Share of rows, by time, before validation starts.
        early_valid_fraction: Share of validation rows, by time, in the early slice.
        gap_s: Delay between training and validation, in seconds.

    Raises:
        ValueError: if ``event_ts`` is empty.
    """
    if len(event_ts) == 0:
        raise ValueError("cannot split an empty set of transactions")
    ordered = np.sort(event_ts)
    valid_start = int(ordered[int(len(ordered) * train_fraction)])
    valid = ordered[np.searchsorted(ordered, valid_start, side="left") :]
    calib_start = int(valid[int(len(valid) * early_valid_fraction)])
    return SplitBounds(
        train_end_ts=valid_start - gap_s,
        valid_start_ts=valid_start,
        calib_start_ts=calib_start,
    )


def assign_roles(event_ts: IntArray, bounds: SplitBounds) -> npt.NDArray[np.str_]:
    """Label each row ``train``, ``gap``, ``valid_early`` or ``valid_late`` by its time."""
    roles: npt.NDArray[np.str_] = np.select(
        [
            event_ts < bounds.train_end_ts,
            event_ts < bounds.valid_start_ts,
            event_ts < bounds.calib_start_ts,
        ],
        ["train", "gap", "valid_early"],
        default="valid_late",
    )
    return roles


def assert_time_ordered(train_ts: IntArray, eval_ts: IntArray) -> None:
    """Fail unless every training time is strictly earlier than every evaluation time.

    Raises:
        TimeOrderError: if either side is empty or the two overlap in time.
    """
    if len(train_ts) == 0 or len(eval_ts) == 0:
        raise TimeOrderError("training and evaluation sets must both be non-empty")
    latest_train, first_eval = int(train_ts.max()), int(eval_ts.min())
    if latest_train >= first_eval:
        raise TimeOrderError(
            f"latest training timestamp {latest_train} is not earlier than "
            f"first evaluation timestamp {first_eval}"
        )


def prequential_folds(
    event_ts: IntArray,
    *,
    valid_start_ts: int,
    valid_end_ts: int,
    n_folds: int = 3,
    gap_s: int = GAP_S,
) -> list[Fold]:
    """Build prequential folds, each one validation block further back in time.

    Fold 0 validates on ``[valid_start_ts, valid_end_ts)``. Each later fold moves that
    window back by its own length. Every fold trains on all rows more than ``gap_s``
    seconds before its validation window, so later folds train on less history.

    Args:
        event_ts: Event times in Unix seconds, one per row, in any order.
        valid_start_ts: Start of the most recent validation block (inclusive).
        valid_end_ts: End of the most recent validation block (exclusive).
        n_folds: Number of folds.
        gap_s: Delay between each fold's training and validation, in seconds.

    Returns:
        Folds from most recent to oldest, with masks aligned to ``event_ts``.

    Raises:
        ValueError: if a fold would have no training rows or no validation rows.
    """
    block = valid_end_ts - valid_start_ts
    folds: list[Fold] = []
    for number in range(n_folds):
        start = valid_start_ts - number * block
        end = valid_end_ts - number * block
        train = event_ts < start - gap_s
        valid = (event_ts >= start) & (event_ts < end)
        if not train.any() or not valid.any():
            raise ValueError(
                f"fold {number} is empty: not enough history for {n_folds} folds "
                f"of {block // DAY_S} days"
            )
        folds.append(
            Fold(
                train=train,
                valid=valid,
                train_end_ts=start - gap_s,
                valid_start_ts=start,
                valid_end_ts=end,
            )
        )
    return folds
