"""The ULB benchmark: the same arms on the public European card dataset.

ULB covers two days, so the 7-day delay gap of the Sparkov split cannot apply: the first
75% of rows by time train, the rest evaluate. Blocks are hours, which serve as both the
bootstrap unit and the alert-budget period. ``Time`` orders rows and is not a feature.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudlens.common.paths import RAW_DIR
from fraudlens.data import schema
from fraudlens.data.schema import validate_raw_ulb
from fraudlens.training.arms import Arm
from fraudlens.training.config import TrainingConfig
from fraudlens.training.data import Frame
from fraudlens.training.experiments import run_study

ULB_FEATURES: tuple[str, ...] = (*(f"V{i}" for i in range(1, 29)), "Amount")
TRAIN_FRACTION = 0.75
HOUR_S = 3_600


def ulb_frame(raw: pd.DataFrame) -> Frame:
    """Build a frame from the raw ULB table, rows in time order."""
    ordered = raw.sort_values("Time", kind="stable").reset_index(drop=True)
    seconds = ordered["Time"].to_numpy(np.float64).astype(np.int64)
    return Frame(
        X=ordered[list(ULB_FEATURES)].to_numpy(np.float64),
        y=ordered["Class"].to_numpy(np.int64),
        amount=ordered["Amount"].to_numpy(np.float64),
        event_ts=seconds,
        block=seconds // HOUR_S,
        ids=np.array([f"ulb_{i}" for i in ordered.index], dtype=np.str_),
        feature_names=ULB_FEATURES,
        feature_types=tuple("q" for _ in ULB_FEATURES),
    )


def time_split(frame: Frame, fraction: float = TRAIN_FRACTION) -> tuple[Frame, Frame]:
    """Split by time; rows sharing the boundary second all go to evaluation."""
    boundary = int(np.sort(frame.event_ts)[int(len(frame) * fraction)])
    early = frame.event_ts < boundary
    return frame.subset(early), frame.subset(~early)


def run_ulb(
    arms: list[Arm],
    config: TrainingConfig,
    *,
    device: str,
    path: Path = RAW_DIR / "ulb" / "creditcard.csv",
    expected_rows: int | None = None,
    expected_frauds: int | None = None,
) -> dict[str, Any]:
    """Validate the ULB file, split it by time and run the study on it.

    The row and fraud counts default to the published ones (``schema.ULB_ROWS`` and
    ``schema.ULB_FRAUDS``), read when called.
    """
    raw = pd.read_csv(path)
    validate_raw_ulb(
        raw,
        expected_rows=schema.ULB_ROWS if expected_rows is None else expected_rows,
        expected_frauds=schema.ULB_FRAUDS if expected_frauds is None else expected_frauds,
    )
    train, valid = time_split(ulb_frame(raw))
    return run_study(
        arms,
        train,
        valid,
        config,
        device=device,
        dataset="ulb",
        evaluated_on="last 25% of rows by time",
    )
