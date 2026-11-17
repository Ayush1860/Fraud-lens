"""Optuna search over the prequential folds for the chosen arm.

Each trial fits the arm once per fold and scores average precision on that fold's
validation block; the objective is the mean over folds. The late validation slice is
never part of any fold, because it is reserved for calibration and thresholds.

With a ``storage`` URL (for example a SQLite file on Google Drive) an interrupted search
resumes where it stopped: finished trials are kept and only the remainder run.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

import optuna

from fraudlens.data.splits import SplitBounds, prequential_folds
from fraudlens.training.arms import Arm, fit_arm
from fraudlens.training.config import TrainingConfig
from fraudlens.training.data import Frame
from fraudlens.training.metrics import average_precision

logger = logging.getLogger(__name__)


def suggest(trial: optuna.Trial, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Draw one parameter set from ``space`` (``{name: {type, low, high, log}}``)."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        log = bool(spec.get("log", False))
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=log)
        else:
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=log
            )
    return params


def tune(
    arm: Arm,
    frame: Frame,
    bounds: SplitBounds,
    config: TrainingConfig,
    *,
    device: str,
    storage: str | None = None,
    study_name: str = "fraudlens",
    progress: Callable[[str], None] = logger.info,
) -> dict[str, Any]:
    """Search hyperparameters for ``arm`` and return a JSON-ready record of the study.

    Args:
        arm: The arm to tune; its resampling is applied inside every fold's training rows.
        frame: The whole training partition; folds are cut from it by time.
        bounds: The split; folds validate before ``bounds.calib_start_ts``.
        config: Training configuration, including the search space and trial count.
        device: ``cpu`` or ``cuda``.
        storage: Optional Optuna storage URL, which makes the search resumable.
        study_name: Name of the study inside ``storage``.
        progress: Receives one line per finished trial.
    """
    folds = prequential_folds(
        frame.event_ts,
        valid_start_ts=bounds.valid_start_ts,
        valid_end_ts=bounds.calib_start_ts,
        n_folds=config.tuning.n_folds,
    )
    fold_frames = [(frame.subset(fold.train), frame.subset(fold.valid)) for fold in folds]

    def objective(trial: optuna.Trial) -> float:
        params = {**config.base_params, **suggest(trial, config.tuning.space)}
        scores = []
        for number, (train, valid) in enumerate(fold_frames):
            fitted = fit_arm(
                arm,
                train,
                valid,
                params=params,
                seed=config.tuning.seed,
                device=device,
                max_rounds=config.max_rounds,
                early_stopping_rounds=config.early_stopping_rounds,
                smote_k=config.smote_k_neighbors,
            )
            score = average_precision(valid.y, fitted.predict_proba(valid.X))
            trial.set_user_attr(f"fold{number}_ap", score)
            trial.set_user_attr(f"fold{number}_best_iteration", fitted.best_iteration)
            scores.append(score)
        mean = sum(scores) / len(scores)
        progress(f"trial {trial.number}: mean AP {mean:.4f} {trial.params}")
        return mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.tuning.seed),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    finished = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if finished < config.tuning.n_trials:
        study.optimize(objective, n_trials=config.tuning.n_trials - finished)

    return {
        "arm": arm.name,
        "objective": "mean average precision over prequential folds",
        "folds": [
            {
                "train_end_ts": fold.train_end_ts,
                "valid_start_ts": fold.valid_start_ts,
                "valid_end_ts": fold.valid_end_ts,
                "train_rows": int(fold.train.sum()),
                "valid_rows": int(fold.valid.sum()),
            }
            for fold in folds
        ],
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": {**config.base_params, **study.best_params},
        "trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
                **t.user_attrs,
            }
            for t in study.trials
        ],
    }
