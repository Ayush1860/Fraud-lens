"""The seven arms of the imbalance study, and how each one is fitted.

Every arm sees the same features and the same split. They differ only in how the rare
class is handled. Any resampling happens to the training rows only, after the split, so
nothing about a validation fraud can reach training.
"""

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xgboost as xgb
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudlens.training.config import TrainingConfig
from fraudlens.training.data import Frame

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_SMOTE_MAJORITY_SAMPLE = 1_000


@dataclass(frozen=True, slots=True)
class Arm:
    """One way of handling class imbalance.

    Attributes:
        name: Stable id, ``A0`` to ``A6``.
        description: What the arm does, for reports.
        learner: ``logreg`` or ``xgboost``.
        weighting: Positive-class weight: none, negatives/positives, or its square root.
        max_delta_step: XGBoost ``max_delta_step``, or ``None`` to leave it unset.
        undersample_ratio: Genuine rows kept per fraud, or ``None`` for no undersampling.
        smote_ratio: Fraud-to-genuine ratio SMOTE fills up to, or ``None`` for no SMOTE.
    """

    name: str
    description: str
    learner: Literal["logreg", "xgboost"]
    weighting: Literal["none", "full", "sqrt"] = "none"
    max_delta_step: float | None = None
    undersample_ratio: float | None = None
    smote_ratio: float | None = None


def make_arms(config: TrainingConfig) -> dict[str, Arm]:
    """Return arms ``A0`` to ``A6``, with resampling ratios taken from ``config``."""
    arms = [
        Arm("A0", "Logistic regression, balanced class weights", "logreg", weighting="full"),
        Arm("A1", "XGBoost, no weighting", "xgboost"),
        Arm("A2", "XGBoost, scale_pos_weight = negatives / positives", "xgboost", "full"),
        Arm("A3", "XGBoost, scale_pos_weight = sqrt(negatives / positives)", "xgboost", "sqrt"),
        Arm("A4", "XGBoost, no weighting, max_delta_step = 1", "xgboost", max_delta_step=1.0),
        Arm(
            "A5",
            f"XGBoost, undersample negatives to 1:{config.undersample_negatives_per_positive:g}, "
            "then prior correction",
            "xgboost",
            undersample_ratio=config.undersample_negatives_per_positive,
        ),
        Arm(
            "A6",
            f"XGBoost, SMOTE inside the training fold up to {config.smote_minority_ratio:g} "
            "frauds per genuine row",
            "xgboost",
            smote_ratio=config.smote_minority_ratio,
        ),
    ]
    return {arm.name: arm for arm in arms}


def prior_correction(probability: FloatArray, beta: float) -> FloatArray:
    """Undo the probability inflation caused by undersampling negatives.

    ``p = beta * p_s / (beta * p_s - p_s + 1)`` (Dal Pozzolo et al., 2015), where ``p_s``
    is the output of a model trained on undersampled data and ``beta`` is the share of
    negatives that were kept.
    """
    corrected: FloatArray = beta * probability / (beta * probability - probability + 1.0)
    return corrected


@dataclass(frozen=True, slots=True, eq=False)
class Resampled:
    """Training rows after an arm's resampling.

    Attributes:
        X: Features.
        y: Labels.
        beta: Share of negatives kept by undersampling, or ``None`` if none were dropped.
    """

    X: FloatArray
    y: IntArray
    beta: float | None


def resample_training(arm: Arm, train: Frame, *, seed: int, smote_k: int = 5) -> Resampled:
    """Apply the arm's resampling to the training rows. Validation rows never come here.

    * Undersampling keeps every fraud and ``undersample_ratio`` genuine rows per fraud,
      chosen at random.
    * SMOTE adds synthetic frauds, interpolated between real training frauds, until there
      are ``smote_ratio`` frauds per genuine row. Rows with a missing value are not used
      as SMOTE seeds, because interpolation is undefined for them; they stay in training.
      The categorical column is handled by SMOTE-NC, which never invents a category.
    """
    positives = np.flatnonzero(train.y == 1)
    negatives = np.flatnonzero(train.y == 0)
    if arm.undersample_ratio is not None:
        rng = np.random.default_rng(seed)
        n_keep = min(len(negatives), math.ceil(arm.undersample_ratio * len(positives)))
        kept = np.sort(np.concatenate([positives, rng.choice(negatives, n_keep, replace=False)]))
        return Resampled(train.X[kept], train.y[kept], beta=n_keep / len(negatives))
    if arm.smote_ratio is not None:
        X_new = _smote_rows(train, arm.smote_ratio, seed=seed, k=smote_k)
        return Resampled(
            np.vstack([train.X, X_new]),
            np.concatenate([train.y, np.ones(len(X_new), dtype=np.int64)]),
            beta=None,
        )
    return Resampled(train.X, train.y, beta=None)


def _smote_rows(train: Frame, ratio: float, *, seed: int, k: int) -> FloatArray:
    """Return only the synthetic fraud rows SMOTE generates for ``train``."""
    complete = ~np.isnan(train.X).any(axis=1)
    seeds = np.flatnonzero(complete & (train.y == 1))
    n_new = math.ceil(ratio * int((train.y == 0).sum())) - int((train.y == 1).sum())
    if n_new <= 0 or len(seeds) < 2:
        return np.empty((0, train.X.shape[1]), dtype=np.float64)

    # SMOTE only draws neighbours among frauds; a small genuine sample satisfies its
    # two-class input check without changing anything it generates.
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(complete & (train.y == 0))
    majority = rng.choice(pool, min(len(pool), _SMOTE_MAJORITY_SAMPLE), replace=False)
    X_in = np.vstack([train.X[seeds], train.X[majority]])
    y_in = np.concatenate([np.ones(len(seeds), np.int64), np.zeros(len(majority), np.int64)])
    options: dict[str, Any] = {
        "sampling_strategy": {1: len(seeds) + n_new},
        "k_neighbors": min(k, len(seeds) - 1),
        "random_state": seed,
    }
    sampler = (
        SMOTENC(categorical_features=list(train.categorical_indices), **options)
        if train.categorical_indices
        else SMOTE(**options)
    )
    X_out, _ = sampler.fit_resample(X_in, y_in)
    synthetic: FloatArray = np.asarray(X_out[len(X_in) :], dtype=np.float64)
    return synthetic


def xgb_params(
    arm: Arm,
    base: Mapping[str, Any],
    *,
    n_pos: int,
    n_neg: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Return the XGBoost parameters for one fit of ``arm``."""
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_cat_to_onehot": 1,
        "seed": seed,
        "device": device,
        "verbosity": 1,
        **base,
    }
    if arm.weighting == "full":
        params["scale_pos_weight"] = n_neg / n_pos
    elif arm.weighting == "sqrt":
        params["scale_pos_weight"] = math.sqrt(n_neg / n_pos)
    if arm.max_delta_step is not None:
        params["max_delta_step"] = arm.max_delta_step
    return params


@dataclass(eq=False)
class FittedArm:
    """A fitted arm that can score new rows.

    Attributes:
        arm: The arm.
        booster: The XGBoost model, cut to its best iteration, or ``None`` for ``logreg``.
        pipeline: The logistic-regression pipeline, or ``None`` for XGBoost arms.
        best_iteration: Best boosting round found by early stopping (XGBoost only).
        beta: Share of negatives kept by undersampling, used for prior correction.
        fit_seconds: Wall-clock fitting time, including resampling.
    """

    arm: Arm
    booster: xgb.Booster | None
    pipeline: Pipeline | None
    best_iteration: int | None
    beta: float | None
    fit_seconds: float

    def raw_probability(self, X: FloatArray) -> FloatArray:
        """Return the model's own fraud probability, before any prior correction."""
        if self.booster is not None:
            return booster_probability(self.booster, X)
        if self.pipeline is None:
            raise RuntimeError(f"arm {self.arm.name} has neither a booster nor a pipeline")
        probability: FloatArray = self.pipeline.predict_proba(X)[:, 1].astype(np.float64)
        return probability

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return fraud probability, prior-corrected if the arm undersampled."""
        probability = self.raw_probability(X)
        return probability if self.beta is None else prior_correction(probability, self.beta)


def booster_probability(booster: xgb.Booster, X: FloatArray) -> FloatArray:
    """Score rows with a booster through ``inplace_predict``, which skips building a DMatrix."""
    probability: FloatArray = np.asarray(booster.inplace_predict(X), dtype=np.float64)
    return probability


def fit_arm(
    arm: Arm,
    train: Frame,
    valid: Frame,
    *,
    params: Mapping[str, Any],
    seed: int,
    device: str,
    max_rounds: int,
    early_stopping_rounds: int,
    smote_k: int = 5,
) -> FittedArm:
    """Fit one arm on ``train``, early-stopping on ``valid`` for XGBoost arms.

    Args:
        arm: The arm to fit.
        train: Training rows, before the arm's resampling.
        valid: Rows that drive early stopping. They are never resampled.
        params: Base XGBoost parameters (ignored by ``logreg``).
        seed: Seed for resampling and for the learner.
        device: ``cpu`` or ``cuda``.
        max_rounds: Most boosting rounds.
        early_stopping_rounds: Rounds without improvement in validation ``aucpr``.
        smote_k: Neighbours for SMOTE.
    """
    started = time.perf_counter()
    data = resample_training(arm, train, seed=seed, smote_k=smote_k)
    if arm.learner == "logreg":
        pipeline = _logreg_pipeline(train.categorical_indices, train.X.shape[1], seed)
        pipeline.fit(data.X, data.y)
        return FittedArm(arm, None, pipeline, None, data.beta, time.perf_counter() - started)

    full = xgb_params(
        arm,
        params,
        n_pos=int((data.y == 1).sum()),
        n_neg=int((data.y == 0).sum()),
        seed=seed,
        device=device,
    )
    names, types = list(train.feature_names), list(train.feature_types)
    dtrain = xgb.QuantileDMatrix(
        data.X, label=data.y, feature_names=names, feature_types=types, enable_categorical=True
    )
    dvalid = xgb.QuantileDMatrix(
        valid.X,
        label=valid.y,
        ref=dtrain,
        feature_names=names,
        feature_types=types,
        enable_categorical=True,
    )
    booster = xgb.train(
        full,
        dtrain,
        num_boost_round=max_rounds,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    best = int(booster.best_iteration)
    return FittedArm(arm, booster[: best + 1], None, best, data.beta, time.perf_counter() - started)


def _logreg_pipeline(categorical: tuple[int, ...], n_features: int, seed: int) -> Pipeline:
    """Impute, standardise and one-hot encode, then a class-balanced logistic regression."""
    numeric = [i for i in range(n_features) if i not in categorical]
    transformers: list[tuple[str, Any, list[int]]] = [
        (
            "numeric",
            make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
            numeric,
        )
    ]
    if categorical:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=-1.0),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                list(categorical),
            )
        )
    return make_pipeline(
        ColumnTransformer(transformers),
        LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=seed),
    )
