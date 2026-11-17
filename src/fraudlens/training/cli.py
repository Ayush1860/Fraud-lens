"""Command line for the training pipeline: ``python -m fraudlens.training <step>``.

Steps, in order:

* ``experiments``: the seven-arm imbalance study -> ``experiments.json``
* ``tune``: Optuna for the best XGBoost arm -> ``tuning.json``
* ``release``: final fit, calibration, thresholds -> ``models/<version>/``
* ``evaluate``: the one-time test-set evaluation, then the report
* ``ulb``: the ULB benchmark -> ``ulb.json``
* ``report``: rebuild ``report.md`` from the artefacts
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fraudlens.common.logging import configure_logging
from fraudlens.common.paths import (
    RAW_DIR,
    TEST_FEATURES_PATH,
    TRAINVAL_FEATURES_PATH,
    TRANSACTIONS_PATH,
)
from fraudlens.features.definitions import CATEGORIES
from fraudlens.training.arms import make_arms
from fraudlens.training.bundle import write_json
from fraudlens.training.config import (
    COSTS_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
    TrainingConfig,
    load_cost_config,
    load_training_config,
)
from fraudlens.training.data import load_trainval
from fraudlens.training.evaluate import final_test
from fraudlens.training.experiments import best_arm, run_study
from fraudlens.training.release import build_release
from fraudlens.training.report import write_report
from fraudlens.training.tune import tune
from fraudlens.training.ulb import run_ulb

logger = logging.getLogger("fraudlens.training")

RESULTS_DIR = Path("docs/results")
MODELS_DIR = Path("models")
XGBOOST_ARMS = ("A1", "A2", "A3", "A4", "A5", "A6")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fraudlens.training")
    parser.add_argument(
        "step", choices=["experiments", "tune", "release", "evaluate", "ulb", "report"]
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--config", type=Path, default=TRAINING_CONFIG_PATH)
    parser.add_argument("--costs", type=Path, default=COSTS_CONFIG_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--trainval", type=Path, default=TRAINVAL_FEATURES_PATH)
    parser.add_argument("--test", type=Path, default=TEST_FEATURES_PATH)
    parser.add_argument("--transactions", type=Path, default=TRANSACTIONS_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--arm", help="override the arm chosen from experiments.json")
    parser.add_argument("--bundle", type=Path, help="bundle to evaluate (default: newest)")
    parser.add_argument("--trials", type=int, help="override tuning.n_trials")
    parser.add_argument("--seeds", type=int, help="use only the first N seeds")
    parser.add_argument("--n-boot", type=int, help="override n_boot")
    parser.add_argument("--max-rounds", type=int, help="override max_rounds")
    parser.add_argument("--storage", help="Optuna storage URL, e.g. sqlite:///tuning.db")
    return parser


def _config(args: argparse.Namespace) -> TrainingConfig:
    config = load_training_config(args.config)
    if args.seeds:
        config = config.with_overrides(seeds=config.seeds[: args.seeds])
    if args.n_boot:
        config = config.with_overrides(n_boot=args.n_boot)
    if args.max_rounds:
        config = config.with_overrides(max_rounds=args.max_rounds)
    if args.trials:
        tuning = config.tuning
        config = config.with_overrides(
            tuning=type(tuning)(args.trials, tuning.n_folds, tuning.seed, tuning.space)
        )
    return config


def _read(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} not found; run the earlier step first")
    return json.loads(path.read_text(encoding="utf-8"))


def _chosen_arm(args: argparse.Namespace) -> str:
    if args.arm:
        return str(args.arm)
    return best_arm(_read(args.results_dir / "experiments.json"), XGBOOST_ARMS)


def _newest_bundle(models_dir: Path) -> Path:
    bundles = sorted(
        (p for p in models_dir.iterdir() if (p / "manifest.json").exists()),
        key=lambda p: (p.name.split(".")[0], int(p.name.split(".")[1])),
    )
    if not bundles:
        raise SystemExit(f"no bundle in {models_dir}; run the release step first")
    return bundles[-1]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one training step. Returns the process exit code."""
    args = _parser().parse_args(argv)
    configure_logging()
    config = _config(args)
    arms = make_arms(config)
    results = args.results_dir
    results.mkdir(parents=True, exist_ok=True)

    if args.step == "experiments":
        frame, _, roles = load_trainval(args.trainval)
        study = run_study(
            list(arms.values()),
            frame.subset(roles == "train"),
            frame.subset(roles == "valid_early"),
            config,
            device=args.device,
            dataset="sparkov",
            evaluated_on="valid_early",
        )
        write_json(results / "experiments.json", study)

    elif args.step == "tune":
        frame, bounds, _ = load_trainval(args.trainval)
        record = tune(
            arms[_chosen_arm(args)],
            frame,
            bounds,
            config,
            device=args.device,
            storage=args.storage,
        )
        write_json(results / "tuning.json", record)

    elif args.step == "release":
        frame, _, roles = load_trainval(args.trainval)
        name = _chosen_arm(args)
        tuning_path = results / "tuning.json"
        tuned = (
            json.loads(tuning_path.read_text(encoding="utf-8")) if tuning_path.exists() else None
        )
        if tuned is not None and tuned["arm"] != name:
            raise SystemExit(f"tuning.json is for arm {tuned['arm']}, not {name}")
        params = tuned["best_params"] if tuned else dict(config.base_params)
        if tuned is None:
            logger.warning("no tuning.json; releasing with the base parameters")
        directory = build_release(
            frame,
            roles,
            arm=arms[name],
            params=params,
            config=config,
            costs=load_cost_config(args.costs),
            categories=CATEGORIES,
            device=args.device,
            models_dir=args.models_dir,
            raw_dir=args.raw_dir,
            provenance={"tuned": tuned is not None},
        )
        print(directory)

    elif args.step == "evaluate":
        bundle = args.bundle or _newest_bundle(args.models_dir)
        final_test(
            bundle,
            test_path=args.test,
            transactions_path=args.transactions,
            budgets=config.alert_budgets,
        )
        write_report(results, bundle)

    elif args.step == "ulb":
        write_json(
            results / "ulb.json",
            run_ulb(
                [arms[name] for name in XGBOOST_ARMS],
                config,
                device=args.device,
                path=args.raw_dir / "ulb" / "creditcard.csv",
            ),
        )

    else:
        bundle = args.bundle or (
            _newest_bundle(args.models_dir) if args.models_dir.exists() else None
        )
        print(write_report(results, bundle))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
