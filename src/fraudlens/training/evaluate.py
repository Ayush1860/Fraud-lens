"""The final, one-time evaluation on the public test partition.

:func:`final_test` is the only function in the project that reads the test features. It
refuses to run twice for the same bundle, so the test set cannot quietly turn into a
second validation set. Its results go into the bundle as ``test_metrics.json``.

Ranking metrics (average precision, alert budgets) use the uncalibrated score: isotonic
calibration maps ranges of scores to one value, which would create ties in the ranking.
Probability and decision metrics (Brier score, cost, fairness) use the calibrated score.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraudlens.common.paths import TEST_FEATURES_PATH, TRANSACTIONS_PATH
from fraudlens.training.bundle import ModelBundle, write_json, write_manifest
from fraudlens.training.data import sparkov_frame
from fraudlens.training.experiments import summarise
from fraudlens.training.metrics import (
    APPROVE,
    BLOCK,
    REVIEW,
    Costs,
    alert_metrics,
    brier_score,
    daily_top_k,
    expected_cost_per_1000,
    reliability_curve,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

TEST_METRICS = "test_metrics.json"
AGE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("under 25", 0, 25),
    ("25-39", 25, 40),
    ("40-54", 40, 55),
    ("55-69", 55, 70),
    ("70 and over", 70, 200),
)


class AlreadyEvaluatedError(RuntimeError):
    """The bundle has already been scored on the test set."""


def final_test(
    bundle_dir: Path,
    *,
    test_path: Path = TEST_FEATURES_PATH,
    transactions_path: Path = TRANSACTIONS_PATH,
    budgets: tuple[float, ...] = (0.005, 0.01),
    n_boot: int = 1_000,
) -> dict[str, Any]:
    """Score the test partition once with a verified bundle and record the results.

    Args:
        bundle_dir: The release to evaluate.
        test_path: Test-partition features (``features.batch`` output).
        transactions_path: Ingested transactions, read only for the test rows' gender,
            which the fairness check needs and the model never sees.
        budgets: Alert budgets as shares of daily volume.
        n_boot: Day-block bootstrap resamples for the 95% intervals.

    Returns:
        The metrics also written to ``test_metrics.json`` in the bundle.

    Raises:
        AlreadyEvaluatedError: if the bundle already has test metrics.
    """
    if (bundle_dir / TEST_METRICS).exists():
        raise AlreadyEvaluatedError(
            f"{bundle_dir.name} was already scored on the test set; the test set is used once"
        )
    bundle = ModelBundle.load(bundle_dir)

    table = pd.read_parquet(test_path)
    gender = pd.read_parquet(
        transactions_path,
        columns=["transaction_id", "gender"],
        filters=[("partition", "==", "test")],
    )
    table = table.merge(gender, on="transaction_id", how="left", validate="one_to_one")
    frame = sparkov_frame(table)

    raw = bundle.raw_probability(frame.X)
    score = bundle.calibrator(raw)
    decision = bundle.decide(score)
    cost_doc = bundle.documents["costs.json"]
    costs = Costs(
        review_cost=float(cost_doc["review_cost"]),
        margin_rate=float(cost_doc["margin_rate"]),
        friction_cost=float(cost_doc["friction_cost"]),
    )

    by_value = {}
    for budget in budgets:
        label = f"{budget * 100:g}pct".replace(".", "_")
        by_value[label] = {
            "rank_by_score": alert_metrics(
                frame.y, frame.amount, daily_top_k(score, frame.block, budget)
            ),
            "rank_by_score_times_amount": alert_metrics(
                frame.y, frame.amount, daily_top_k(score * frame.amount, frame.block, budget)
            ),
        }

    results: dict[str, Any] = {
        "bundle": bundle.version,
        "rows": len(frame),
        "frauds": int(frame.y.sum()),
        "days": len(np.unique(frame.block)),
        "ranking": summarise([raw], frame, budgets, n_boot=n_boot),
        "brier_raw": brier_score(frame.y, raw),
        "brier_calibrated": brier_score(frame.y, score),
        "reliability_raw": reliability_curve(frame.y, raw),
        "reliability_calibrated": reliability_curve(frame.y, score),
        "thresholds": {
            "t_review": bundle.thresholds.t_review,
            "t_block": bundle.thresholds.t_block,
        },
        "expected_cost_per_1000": expected_cost_per_1000(frame.y, frame.amount, decision, costs),
        "cost_of_approving_everything_per_1000": expected_cost_per_1000(
            frame.y, frame.amount, np.full(len(frame), APPROVE), costs
        ),
        "decisions": _decision_table(frame.y, frame.amount, decision),
        "queue_ordering": by_value,
        "fairness": _fairness(
            table["gender"].to_numpy(np.str_),
            table["cardholder_age"].to_numpy(np.float64),
            frame.y,
            decision,
        ),
    }
    write_json(bundle_dir / TEST_METRICS, results)
    write_manifest(bundle_dir)
    logger.info("test set scored", extra={"bundle": bundle.version})
    return results


def _decision_table(
    y: npt.NDArray[np.int64], amount: FloatArray, decision: npt.NDArray[np.int64]
) -> dict[str, dict[str, float]]:
    table = {}
    for name, code in (("approve", APPROVE), ("review", REVIEW), ("block", BLOCK)):
        chosen = decision == code
        table[name] = {
            "rows": int(chosen.sum()),
            "share": float(chosen.mean()),
            "frauds": int((chosen & (y == 1)).sum()),
            "fraud_amount": float(amount[chosen & (y == 1)].sum()),
            "genuine": int((chosen & (y == 0)).sum()),
        }
    return table


def _fairness(
    gender: npt.NDArray[np.str_],
    age: FloatArray,
    y: npt.NDArray[np.int64],
    decision: npt.NDArray[np.int64],
) -> dict[str, list[dict[str, Any]]]:
    """False-positive rate of any flag (review or block) among genuine rows, by group."""
    flagged = decision != APPROVE
    genuine = y == 0

    def row(name: str, members: npt.NDArray[np.bool_]) -> dict[str, Any]:
        group = members & genuine
        return {
            "group": name,
            "genuine_rows": int(group.sum()),
            "false_positive_rate": float(flagged[group].mean()) if group.any() else None,
            "fraud_rows": int((members & ~genuine).sum()),
            "recall": float(flagged[members & ~genuine].mean())
            if (members & ~genuine).any()
            else None,
        }

    return {
        "gender": [row(g, gender == g) for g in sorted(set(gender.tolist()))],
        "age_band": [row(name, (age >= low) & (age < high)) for name, low, high in AGE_BANDS],
    }
