"""Typed access to ``configs/training.yaml`` and ``configs/costs.yaml``."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from fraudlens.training.metrics import Costs

TRAINING_CONFIG_PATH = Path("configs/training.yaml")
COSTS_CONFIG_PATH = Path("configs/costs.yaml")


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """Optuna settings. ``space`` maps a parameter to ``{type, low, high, log}``."""

    n_trials: int
    n_folds: int
    seed: int
    space: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Settings shared by the imbalance study, tuning and the final fit.

    Attributes:
        seeds: Seeds each experiment arm is run with.
        max_rounds: Most boosting rounds any XGBoost fit may use.
        early_stopping_rounds: Rounds without validation improvement before stopping.
        n_boot: Day-block bootstrap resamples behind each 95% interval.
        alert_budgets: Shares of daily volume an analyst team could review.
        undersample_negatives_per_positive: Genuine rows kept per fraud in arm A5.
        smote_minority_ratio: Fraud-to-genuine ratio SMOTE fills up to in arm A6.
        smote_k_neighbors: Neighbours SMOTE interpolates between.
        base_params: XGBoost parameters every experiment arm uses.
        tuning: Optuna settings.
    """

    seeds: tuple[int, ...]
    max_rounds: int
    early_stopping_rounds: int
    n_boot: int
    alert_budgets: tuple[float, ...]
    undersample_negatives_per_positive: float
    smote_minority_ratio: float
    smote_k_neighbors: int
    base_params: dict[str, float | int]
    tuning: TuningConfig

    def with_overrides(self, **changes: Any) -> "TrainingConfig":
        """Return a copy with some fields replaced, for quick runs and tests."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Business costs plus the analysts' daily review capacity (a share of volume)."""

    costs: Costs
    review_capacity_share: float


def load_training_config(path: Path = TRAINING_CONFIG_PATH) -> TrainingConfig:
    """Read the training configuration from YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tuning = raw["tuning"]
    return TrainingConfig(
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        max_rounds=int(raw["max_rounds"]),
        early_stopping_rounds=int(raw["early_stopping_rounds"]),
        n_boot=int(raw["n_boot"]),
        alert_budgets=tuple(float(budget) for budget in raw["alert_budgets"]),
        undersample_negatives_per_positive=float(raw["undersample_negatives_per_positive"]),
        smote_minority_ratio=float(raw["smote_minority_ratio"]),
        smote_k_neighbors=int(raw["smote_k_neighbors"]),
        base_params=dict(raw["base_params"]),
        tuning=TuningConfig(
            n_trials=int(tuning["n_trials"]),
            n_folds=int(tuning["n_folds"]),
            seed=int(tuning["seed"]),
            space=dict(tuning["space"]),
        ),
    )


def load_cost_config(path: Path = COSTS_CONFIG_PATH) -> CostConfig:
    """Read the business costs from YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CostConfig(
        costs=Costs(
            review_cost=float(raw["review_cost"]),
            margin_rate=float(raw["margin_rate"]),
            friction_cost=float(raw["friction_cost"]),
        ),
        review_capacity_share=float(raw["review_capacity_share"]),
    )
