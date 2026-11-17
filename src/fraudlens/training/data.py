"""Model-ready arrays, and the time split of the training partition.

A :class:`Frame` is dataset-agnostic: the Sparkov study and the ULB benchmark both build
one, so the experiment harness never needs to know which dataset it is running on.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraudlens.common.paths import TRAINVAL_FEATURES_PATH
from fraudlens.data.splits import SplitBounds, assign_roles, split_bounds
from fraudlens.features.definitions import DAY_S, FEATURE_NAMES
from fraudlens.features.encoding import FEATURE_TYPES, encode_frame

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """Features, labels and the context metrics need, row-aligned.

    Attributes:
        X: Encoded features, one column per name in ``feature_names``.
        y: Labels, 1 for fraud.
        amount: Transaction amounts in dollars.
        event_ts: Event times in Unix seconds.
        block: Bootstrap and alert-budget block of each row (the day, for Sparkov).
        ids: Transaction ids.
        feature_names: Column names of ``X``.
        feature_types: XGBoost type per column: ``c`` categorical, ``q`` numeric.
    """

    X: FloatArray
    y: IntArray
    amount: FloatArray
    event_ts: IntArray
    block: IntArray
    ids: StrArray
    feature_names: tuple[str, ...]
    feature_types: tuple[str, ...]

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.y)

    def subset(self, rows: npt.NDArray[np.bool_] | IntArray) -> "Frame":
        """Return the rows selected by a boolean mask or an array of positions."""
        return replace(
            self,
            X=self.X[rows],
            y=self.y[rows],
            amount=self.amount[rows],
            event_ts=self.event_ts[rows],
            block=self.block[rows],
            ids=self.ids[rows],
        )

    @property
    def categorical_indices(self) -> tuple[int, ...]:
        """Positions of the categorical columns."""
        return tuple(i for i, kind in enumerate(self.feature_types) if kind == "c")


def sparkov_frame(table: pd.DataFrame) -> Frame:
    """Build a frame from a Sparkov feature table (``features.batch`` output).

    Blocks are UTC days, so bootstrap intervals and alert budgets work day by day.
    """
    event_ts = table["event_ts"].to_numpy(np.int64)
    return Frame(
        X=encode_frame(table),
        y=table["is_fraud"].to_numpy(np.int64),
        amount=table["amount"].to_numpy(np.float64),
        event_ts=event_ts,
        block=event_ts // DAY_S,
        ids=table["transaction_id"].to_numpy(np.str_),
        feature_names=FEATURE_NAMES,
        feature_types=FEATURE_TYPES,
    )


def load_trainval(path: Path = TRAINVAL_FEATURES_PATH) -> tuple[Frame, SplitBounds, StrArray]:
    """Load the public training partition and split it by time.

    Returns:
        The frame, the split boundaries, and each row's role: ``train``, ``gap``,
        ``valid_early`` or ``valid_late``.
    """
    frame = sparkov_frame(pd.read_parquet(path))
    bounds = split_bounds(frame.event_ts)
    return frame, bounds, assign_roles(frame.event_ts, bounds)
