"""A small released bundle trained on synthetic transactions, for service tests and CI.

Run as a script to write one into a models directory, which is how CI gives the Docker
Compose service a model to load:

    uv run python tests/toy_bundle.py models
    uv run python tests/toy_bundle.py models --released-size   # for load tests
"""

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb

if __package__ in (None, ""):  # pragma: no cover - script use from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fraudlens.common.hashing import card_token
from fraudlens.data.splits import assign_roles, split_bounds
from fraudlens.features.batch import KEY_COLUMNS, build_features
from fraudlens.features.definitions import CATEGORIES
from fraudlens.training.arms import make_arms
from fraudlens.training.bundle import BOOSTER, booster_margin, logistic, write_json, write_manifest
from fraudlens.training.calibrate import fit_isotonic
from fraudlens.training.config import CostConfig
from fraudlens.training.data import Frame, sparkov_frame
from fraudlens.training.metrics import Costs
from fraudlens.training.release import build_release
from fraudlens.training.thresholds import choose_thresholds
from helpers import make_transactions
from training_fixtures import TINY_CONFIG

TOY_HMAC_KEY = b"toy-bundle-hmac-key"


def toy_transactions(n_txns: int = 12_000, *, seed: int = 5, span_days: int = 240) -> pd.DataFrame:
    """Synthetic transactions with real-looking card tokens and a learnable fraud label.

    Card ids are HMAC tokens of the synthetic card ids, so they pass the service's
    ``card_id`` validation. Fraud is likelier for large night-time purchases.
    """
    txns = make_transactions(n_cards=60, n_txns=n_txns, seed=seed, span_days=span_days)
    txns["card_id"] = [card_token(card, TOY_HMAC_KEY) for card in txns["card_id"]]
    features = build_features(txns)
    rng = np.random.default_rng(seed)
    logit = -6.5 + 0.6 * np.log1p(txns["amount"].to_numpy()) + 1.5 * features["is_night"]
    txns["is_fraud"] = (rng.random(len(txns)) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    return txns.sort_values(["event_ts", "transaction_id"]).reset_index(drop=True)


RELEASED_SIZE = {"trees": 398, "max_depth": 6}
"""Shape of release 2026-11-17.1, for load tests: timing depends on trees and depth."""


def build_toy_bundle(
    models_dir: Path, *, seed: int = 5, released_size: bool = False, n_txns: int = 12_000
) -> Path:
    """Train and release a bundle on synthetic data into ``models_dir``; return its directory.

    Args:
        models_dir: Where the release directory is created.
        seed: Seed for the synthetic transactions.
        released_size: Train exactly as many trees, as deep, as the released model, so
            that model and explanation timings in a load test are realistic.
        n_txns: Synthetic transactions to train on.
    """
    txns = toy_transactions(n_txns, seed=seed)
    table = pd.concat([txns[list(KEY_COLUMNS)], build_features(txns)], axis=1)
    frame = sparkov_frame(table)
    roles = assign_roles(frame.event_ts, split_bounds(frame.event_ts))
    costs = CostConfig(Costs(4.0, 0.02, 15.0), 0.05)
    bundle = build_release(
        frame,
        roles,
        arm=make_arms(TINY_CONFIG)["A1"],
        params=TINY_CONFIG.base_params,
        config=TINY_CONFIG,
        costs=costs,
        categories=CATEGORIES,
        device="cpu",
        models_dir=models_dir,
        raw_dir=models_dir / "raw-not-used",
    )
    if released_size:
        _refit_at_released_size(bundle, frame, roles, costs)
    return bundle


def _refit_at_released_size(
    bundle: Path, frame: Frame, roles: npt.NDArray[np.str_], costs: CostConfig
) -> None:
    """Replace the booster with one of the released model's size, then recalibrate.

    A release keeps only the trees up to its best validation round, which on synthetic
    data comes early. For timing, the tree count and depth matter, so this trains exactly
    ``RELEASED_SIZE`` trees and refits the calibrator and thresholds with the same library
    functions a release uses, keeping the flag rate cost-optimal.
    """
    train, late = frame.subset(roles == "train"), frame.subset(roles == "valid_late")
    matrix = xgb.DMatrix(
        train.X,
        label=train.y,
        feature_names=list(train.feature_names),
        feature_types=list(train.feature_types),
        enable_categorical=True,
    )
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": RELEASED_SIZE["max_depth"],
            "learning_rate": 0.05,
            "min_child_weight": 0,
            "max_cat_to_onehot": 1,
            "seed": 0,
        },
        matrix,
        num_boost_round=RELEASED_SIZE["trees"],
    )
    raw_late = logistic(booster_margin(booster, late.X))
    calibrator = fit_isotonic(raw_late, late.y)
    thresholds, _ = choose_thresholds(calibrator(raw_late), late.y, late.amount, costs)
    booster.save_model(bundle / BOOSTER)
    write_json(bundle / "calibrator.json", calibrator.to_json())
    write_json(
        bundle / "thresholds.json",
        {"t_review": thresholds.t_review, "t_block": thresholds.t_block},
    )
    write_manifest(bundle)


if __name__ == "__main__":  # pragma: no cover
    released = "--released-size" in sys.argv
    target = next((a for a in sys.argv[1:] if not a.startswith("--")), "models")
    print(
        build_toy_bundle(
            Path(target), released_size=released, n_txns=60_000 if released else 12_000
        )
    )
