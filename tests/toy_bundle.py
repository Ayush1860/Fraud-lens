"""A small released bundle trained on synthetic transactions, for service tests and CI.

Run as a script to write one into a models directory, which is how CI gives the Docker
Compose service a model to load:

    uv run python tests/toy_bundle.py models
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # pragma: no cover - script use from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fraudlens.common.hashing import card_token
from fraudlens.data.splits import assign_roles, split_bounds
from fraudlens.features.batch import KEY_COLUMNS, build_features
from fraudlens.features.definitions import CATEGORIES
from fraudlens.training.arms import make_arms
from fraudlens.training.config import CostConfig
from fraudlens.training.data import sparkov_frame
from fraudlens.training.metrics import Costs
from fraudlens.training.release import build_release
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


def build_toy_bundle(models_dir: Path, *, seed: int = 5) -> Path:
    """Train and release a tiny bundle into ``models_dir``; return its directory."""
    txns = toy_transactions(seed=seed)
    table = pd.concat([txns[list(KEY_COLUMNS)], build_features(txns)], axis=1)
    frame = sparkov_frame(table)
    roles = assign_roles(frame.event_ts, split_bounds(frame.event_ts))
    return build_release(
        frame,
        roles,
        arm=make_arms(TINY_CONFIG)["A1"],
        params=TINY_CONFIG.base_params,
        config=TINY_CONFIG,
        costs=CostConfig(Costs(4.0, 0.02, 15.0), 0.05),
        categories=CATEGORIES,
        device="cpu",
        models_dir=models_dir,
        raw_dir=models_dir / "raw-not-used",
    )


if __name__ == "__main__":  # pragma: no cover
    print(build_toy_bundle(Path(sys.argv[1] if len(sys.argv) > 1 else "models")))
