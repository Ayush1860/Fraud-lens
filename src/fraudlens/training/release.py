"""Fit the final model, calibrate it, choose thresholds and write the release bundle.

Roles of the training partition, in time order:

* ``train`` fits the booster;
* ``valid_early`` drives early stopping;
* ``valid_late`` fits the calibrator and chooses the thresholds.

No row is used for two of these jobs, and the public test partition is not read here.
"""

import json
import logging
import subprocess
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from fraudlens.common.paths import RAW_DIR
from fraudlens.training.arms import Arm, fit_arm
from fraudlens.training.bundle import (
    ModelBundle,
    booster_margin,
    logistic,
    next_version,
    write_bundle,
    write_json,
    write_manifest,
)
from fraudlens.training.calibrate import fit_isotonic
from fraudlens.training.config import CostConfig, TrainingConfig
from fraudlens.training.data import Frame, StrArray
from fraudlens.training.experiments import point_metrics
from fraudlens.training.metrics import (
    APPROVE,
    BLOCK,
    REVIEW,
    brier_score,
    expected_cost_per_1000,
    reliability_curve,
)
from fraudlens.training.thresholds import choose_thresholds, decide, threshold_curve

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

REFERENCE_BINS = 10
REPRODUCTION_ROWS = 1_000


def reference_histograms(
    train: Frame, score: FloatArray, categories: tuple[str, ...]
) -> dict[str, Any]:
    """Distributions the drift monitor compares live traffic against.

    Numeric features get decile edges from the training rows and the share of rows in
    each bin, plus the share that is missing. The categorical feature gets the share of
    each category. The score gets decile edges and shares from ``score``, the calibrated
    scores on the late validation slice (out-of-sample, unlike training scores).
    """
    features: dict[str, Any] = {}
    for column, name in enumerate(train.feature_names):
        values = train.X[:, column]
        missing = np.isnan(values)
        if train.feature_types[column] == "c":
            codes = values[~missing].astype(np.int64)
            counts = np.bincount(codes, minlength=len(categories))
            features[name] = {
                "kind": "categorical",
                "categories": list(categories),
                "proportions": (counts / max(1, len(values))).tolist(),
                "missing": float(missing.mean()),
            }
        else:
            features[name] = {"kind": "numeric", **_histogram(values[~missing], len(values))}
            features[name]["missing"] = float(missing.mean())
    return {"features": features, "score": _histogram(score, len(score))}


def _histogram(values: FloatArray, total: int) -> dict[str, list[float]]:
    """Quantile edges (outer edges open) and the share of ``total`` rows in each bin."""
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, REFERENCE_BINS + 1)[1:-1])
    inner = np.unique(quantiles)
    counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
    return {"inner_edges": inner.tolist(), "proportions": (counts / max(1, total)).tolist()}


def git_commit() -> str:
    """Return the current git commit, or ``unknown`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def build_release(
    frame: Frame,
    roles: StrArray,
    *,
    arm: Arm,
    params: Mapping[str, Any],
    config: TrainingConfig,
    costs: CostConfig,
    categories: tuple[str, ...],
    device: str,
    models_dir: Path,
    raw_dir: Path = RAW_DIR,
    today: date | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Fit, calibrate, choose thresholds and write a verified bundle.

    Args:
        frame: The whole training partition.
        roles: Each row's role from the time split.
        arm: The arm to fit; must be an XGBoost arm.
        params: XGBoost parameters, normally the tuned ones.
        config: Training configuration.
        costs: Business costs and review capacity.
        categories: Category order of the categorical feature's codes.
        device: ``cpu`` or ``cuda`` for fitting. Scoring for the bundle is on CPU.
        models_dir: Where ``<version>/`` is created.
        raw_dir: Directory holding ``SHA256SUMS.json`` of the raw data.
        today: Release date for the version; defaults to today (UTC).
        provenance: Extra facts recorded in ``training.json`` (for example the run host).

    Returns:
        The bundle directory.
    """
    if arm.learner != "xgboost":
        raise ValueError(f"only XGBoost arms can be released, not {arm.name}")
    train = frame.subset(roles == "train")
    early = frame.subset(roles == "valid_early")
    late = frame.subset(roles == "valid_late")

    fitted = fit_arm(
        arm,
        train,
        early,
        params=params,
        seed=config.seeds[0],
        device=device,
        max_rounds=config.max_rounds,
        early_stopping_rounds=config.early_stopping_rounds,
        smote_k=config.smote_k_neighbors,
    )
    if fitted.booster is None:
        raise RuntimeError(f"arm {arm.name} produced no booster")
    booster = fitted.booster
    booster.set_param({"device": "cpu"})

    raw_early = logistic(booster_margin(booster, early.X))
    raw_late = logistic(booster_margin(booster, late.X))
    calibrator = fit_isotonic(raw_late, late.y)
    score_late = calibrator(raw_late)
    thresholds, cost = choose_thresholds(score_late, late.y, late.amount, costs)
    decision = decide(score_late, thresholds)

    metrics = {
        "valid_early": {
            "rows": len(early),
            "frauds": int(early.y.sum()),
            **point_metrics(raw_early, early, config.alert_budgets),
        },
        "valid_late": {
            "rows": len(late),
            "frauds": int(late.y.sum()),
            "ranking_on_raw_score": point_metrics(raw_late, late, config.alert_budgets),
            "brier_raw": brier_score(late.y, raw_late),
            "brier_calibrated": brier_score(late.y, score_late),
            "reliability_raw": reliability_curve(late.y, raw_late),
            "reliability_calibrated": reliability_curve(late.y, score_late),
            "expected_cost_per_1000": cost,
            "expected_cost_check": expected_cost_per_1000(
                late.y, late.amount, decision, costs.costs
            ),
            "decision_share": {
                "approve": float((decision == APPROVE).mean()),
                "review": float((decision == REVIEW).mean()),
                "block": float((decision == BLOCK).mean()),
            },
        },
        "note": (
            "Calibrator and thresholds are fitted on valid_late, so its calibrated "
            "metrics are in-sample. Out-of-sample numbers come from the final test."
        ),
    }
    raw_hashes_path = raw_dir / "SHA256SUMS.json"
    training = {
        "arm": arm.name,
        "arm_description": arm.description,
        "params": dict(params),
        "best_iteration": fitted.best_iteration,
        "seed": config.seeds[0],
        "fit_device": device,
        "fit_seconds": round(fitted.fit_seconds, 1),
        "rows": {role: int((roles == role).sum()) for role in np.unique(roles)},
        "frauds": {role: int(frame.y[roles == role].sum()) for role in np.unique(roles)},
        "raw_data_sha256": (
            json.loads(raw_hashes_path.read_text(encoding="utf-8"))
            if raw_hashes_path.exists()
            else None
        ),
        "git_commit": git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **(provenance or {}),
    }

    version = next_version(models_dir, today or datetime.now(UTC).date())
    directory = write_bundle(
        models_dir / version,
        booster=booster,
        calibrator=calibrator,
        thresholds=thresholds,
        files={
            "features.json": {
                "names": list(frame.feature_names),
                "types": list(frame.feature_types),
                "categories": list(categories),
                "encoding": "categorical features hold their index in `categories`",
            },
            "costs.json": {
                "review_cost": costs.costs.review_cost,
                "margin_rate": costs.costs.margin_rate,
                "friction_cost": costs.costs.friction_cost,
                "review_capacity_share": costs.review_capacity_share,
            },
            "threshold_curve.json": threshold_curve(
                score_late, late.y, late.amount, late.block, costs
            ),
            "reference.json": reference_histograms(train, score_late, categories),
            "metrics.json": metrics,
            "training.json": training,
        },
    )

    # Record what the saved bundle, reloaded on CPU, scores for a sample of rows, so any
    # later environment can prove it reproduces the release exactly.
    bundle = ModelBundle.load(directory)
    rng = np.random.default_rng(config.seeds[0])
    rows = np.sort(rng.choice(len(late), min(REPRODUCTION_ROWS, len(late)), replace=False))
    sample = late.subset(rows)
    write_json(
        directory / "reproduction.json",
        {
            "source": "valid_late",
            "transaction_ids": sample.ids.tolist(),
            "margin": bundle.margin(sample.X).tolist(),
            "score": bundle.score(sample.X).tolist(),
        },
    )
    write_manifest(directory)
    ModelBundle.load(directory)
    logger.info("bundle written", extra={"path": str(directory)})
    return directory
