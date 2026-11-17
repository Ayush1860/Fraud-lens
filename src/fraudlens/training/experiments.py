"""Run the imbalance arms over several seeds and summarise them with 95% intervals.

Each interval comes from a block bootstrap: whole days (or, for ULB, hours) are drawn
with replacement, every seed's predictions are scored on the resample, and the seeds are
averaged. The interval therefore covers both sampling noise in the evaluation days and
the average over seeds. ``seed_std`` reports the spread between seeds on its own.

Alerts are chosen within each block (a daily budget), so a resampled day keeps exactly
the alerts it had. That lets the alert metrics and the Brier score be bootstrapped from
per-block totals; only average precision needs a full recomputation per resample.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import precision_recall_curve, roc_auc_score

from fraudlens.training.arms import Arm, fit_arm
from fraudlens.training.config import TrainingConfig
from fraudlens.training.data import Frame
from fraudlens.training.metrics import (
    alert_metrics,
    average_precision,
    bootstrap_days,
    brier_score,
    daily_top_k,
    day_blocks,
    percentile_interval,
    reliability_curve,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

PR_CURVE_POINTS = 101


def point_metrics(
    probability: FloatArray, frame: Frame, budgets: Sequence[float]
) -> dict[str, float]:
    """Return every metric of one set of predictions on ``frame``.

    Ranking metrics use only the order of ``probability``; ``brier`` uses its values.
    """
    metrics = {
        "average_precision": average_precision(frame.y, probability),
        "roc_auc": float(roc_auc_score(frame.y, probability)),
        "brier": brier_score(frame.y, probability),
    }
    for budget in budgets:
        flagged = daily_top_k(probability, frame.block, budget)
        for name, value in alert_metrics(frame.y, frame.amount, flagged).items():
            metrics[f"{name}_at_{_budget_label(budget)}"] = value
    return metrics


def summarise(
    predictions: Sequence[FloatArray],
    frame: Frame,
    budgets: Sequence[float],
    *,
    n_boot: int,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Summarise one arm's predictions, one array per seed, on the same ``frame``.

    Returns:
        ``{metric: {"mean": ..., "seed_std": ..., "ci95": [low, high]}}``. ``mean`` is the
        average over seeds of the metric on the full evaluation rows. ``roc_auc`` is
        reported as a point value only.
    """
    per_seed = [point_metrics(p, frame, budgets) for p in predictions]
    summary: dict[str, dict[str, Any]] = {
        name: {
            "mean": float(np.mean([m[name] for m in per_seed])),
            "seed_std": float(np.std([m[name] for m in per_seed])),
            "ci95": None,
        }
        for name in per_seed[0]
    }

    block_index = np.unique(frame.block, return_inverse=True)[1].astype(np.int64)
    n_blocks = int(block_index.max()) + 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, (n_boot, n_blocks))
    counts = np.stack([np.bincount(row, minlength=n_blocks) for row in draws]).astype(np.float64)

    def per_block(values: FloatArray) -> FloatArray:
        totals: FloatArray = np.bincount(block_index, weights=values, minlength=n_blocks)
        return totals

    n_rows = per_block(np.ones(len(frame)))
    fraud = (frame.y == 1).astype(np.float64)
    fraud_rows = per_block(fraud)
    fraud_value = per_block(fraud * frame.amount)

    replicates: dict[str, list[FloatArray]] = {name: [] for name in summary}
    for probability in predictions:
        squared = per_block((probability - frame.y) ** 2)
        replicates["brier"].append(_ratio(counts @ squared, counts @ n_rows))
        for budget in budgets:
            label = _budget_label(budget)
            flagged = daily_top_k(probability, frame.block, budget).astype(np.float64)
            caught = per_block(flagged * fraud)
            replicates[f"recall_at_{label}"].append(_ratio(counts @ caught, counts @ fraud_rows))
            replicates[f"precision_at_{label}"].append(
                _ratio(counts @ caught, counts @ per_block(flagged))
            )
            replicates[f"dollar_recall_at_{label}"].append(
                _ratio(counts @ per_block(flagged * fraud * frame.amount), counts @ fraud_value)
            )

    # Average precision needs the rows themselves. The block draws are regenerated
    # from the same seed, so they line up with the ones above.
    blocks = day_blocks(block_index)
    ap = np.array(
        [
            [average_precision(frame.y[rows], p[rows]) for p in predictions]
            for rows in bootstrap_days(blocks, n_boot, seed)
        ]
    )
    replicates["average_precision"] = [ap[:, column] for column in range(len(predictions))]

    for name, values in replicates.items():
        if values:
            low, high = percentile_interval(np.mean(np.stack(values), axis=0))
            summary[name]["ci95"] = [low, high]
    return summary


def pr_curve(y: IntArray, probability: FloatArray) -> dict[str, list[float]]:
    """Return interpolated precision at 101 evenly spaced recall levels, for plotting."""
    precision, recall, _ = precision_recall_curve(y, probability)
    levels = np.linspace(0.0, 1.0, PR_CURVE_POINTS)
    best = [float(precision[recall >= level].max()) for level in levels]
    return {"recall": [float(level) for level in levels], "precision": best}


def run_study(
    arms: Sequence[Arm],
    train: Frame,
    valid: Frame,
    config: TrainingConfig,
    *,
    device: str,
    dataset: str,
    evaluated_on: str,
    params: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] = logger.info,
) -> dict[str, Any]:
    """Fit every arm with every seed on ``train`` and evaluate it on ``valid``.

    ``valid`` also drives early stopping, identically for every arm, so the comparison
    between arms is fair even though the absolute numbers are slightly optimistic.

    Returns:
        A JSON-ready dict: settings, then one entry per arm with its summary, the
        per-seed details, a precision-recall curve and a reliability curve (seed 0).
    """
    base = dict(config.base_params if params is None else params)
    results: list[dict[str, Any]] = []
    for arm in arms:
        predictions: list[FloatArray] = []
        seeds: list[dict[str, Any]] = []
        for seed in config.seeds:
            fitted = fit_arm(
                arm,
                train,
                valid,
                params=base,
                seed=seed,
                device=device,
                max_rounds=config.max_rounds,
                early_stopping_rounds=config.early_stopping_rounds,
                smote_k=config.smote_k_neighbors,
            )
            probability = fitted.predict_proba(valid.X)
            predictions.append(probability)
            seeds.append(
                {
                    "seed": seed,
                    "best_iteration": fitted.best_iteration,
                    "fit_seconds": round(fitted.fit_seconds, 2),
                    **point_metrics(probability, valid, config.alert_budgets),
                }
            )
            progress(
                f"{arm.name} seed {seed}: AP {seeds[-1]['average_precision']:.4f} "
                f"in {fitted.fit_seconds:.0f}s"
            )
        results.append(
            {
                "arm": arm.name,
                "description": arm.description,
                "summary": summarise(
                    predictions, valid, config.alert_budgets, n_boot=config.n_boot
                ),
                "per_seed": seeds,
                "pr_curve": pr_curve(valid.y, predictions[0]),
                "reliability": reliability_curve(valid.y, predictions[0]),
            }
        )
    return {
        "dataset": dataset,
        "evaluated_on": evaluated_on,
        "train_rows": len(train),
        "train_frauds": int(train.y.sum()),
        "eval_rows": len(valid),
        "eval_frauds": int(valid.y.sum()),
        "seeds": list(config.seeds),
        "n_boot": config.n_boot,
        "alert_budgets": list(config.alert_budgets),
        "params": base,
        "arms": results,
    }


def best_arm(study: Mapping[str, Any], candidates: Sequence[str]) -> str:
    """Return the candidate arm with the highest mean average precision in ``study``."""
    scores = {
        entry["arm"]: entry["summary"]["average_precision"]["mean"]
        for entry in study["arms"]
        if entry["arm"] in candidates
    }
    if not scores:
        raise ValueError(f"none of {list(candidates)} is in the study")
    return str(max(scores, key=lambda name: scores[name]))


def _budget_label(budget: float) -> str:
    return f"{budget * 100:g}pct".replace(".", "_")


def _ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio: FloatArray = np.where(denominator > 0, numerator / denominator, np.nan)
    return ratio
