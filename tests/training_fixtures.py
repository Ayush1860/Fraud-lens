"""Small synthetic datasets for training tests: fast to fit, with a learnable signal."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fraudlens.features.batch import KEY_COLUMNS, build_features
from fraudlens.training.config import TrainingConfig, TuningConfig
from fraudlens.training.data import Frame, sparkov_frame
from helpers import make_transactions

TINY_CONFIG = TrainingConfig(
    seeds=(0, 1),
    max_rounds=30,
    early_stopping_rounds=5,
    n_boot=20,
    alert_budgets=(0.005, 0.01),
    undersample_negatives_per_positive=10.0,
    smote_minority_ratio=0.1,
    smote_k_neighbors=5,
    base_params={"max_depth": 3, "learning_rate": 0.3, "subsample": 1.0},
    tuning=TuningConfig(
        n_trials=2,
        n_folds=2,
        seed=0,
        space={
            "max_depth": {"type": "int", "low": 2, "high": 4},
            "learning_rate": {"type": "float", "low": 0.1, "high": 0.3, "log": True},
        },
    ),
)


def feature_table(n_txns: int = 12_000, *, seed: int = 0, span_days: int = 240) -> pd.DataFrame:
    """Features plus keys and labels; fraud is likelier for large night-time purchases."""
    txns = make_transactions(n_cards=60, n_txns=n_txns, seed=seed, span_days=span_days)
    features = build_features(txns)
    rng = np.random.default_rng(seed)
    logit = -6.5 + 1.2 * np.log1p(txns["amount"].to_numpy()) / 2 + 1.5 * features["is_night"]
    txns["is_fraud"] = (rng.random(len(txns)) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    table = pd.concat([txns[list(KEY_COLUMNS)], features], axis=1)
    return table.sort_values(["event_ts", "transaction_id"]).reset_index(drop=True)


def toy_frame(n_txns: int = 12_000, *, seed: int = 0) -> Frame:
    """A Sparkov-style frame built from :func:`feature_table`."""
    return sparkov_frame(feature_table(n_txns, seed=seed))


@dataclass
class Workspace:
    """Every file the training command line reads, in a temporary directory."""

    root: Path
    trainval: Path
    test: Path
    transactions: Path
    raw_dir: Path
    config: Path
    costs: Path
    results: Path
    models: Path


def make_workspace(root: Path) -> Workspace:
    """Write toy train/test feature tables, transactions, a ULB file and configs."""
    table = feature_table(16_000, seed=3, span_days=300)
    cut = table["event_ts"].quantile(0.8)
    processed = root / "processed"
    processed.mkdir()
    trainval, test = processed / "trainval.parquet", processed / "test.parquet"
    table[table["event_ts"] < cut].to_parquet(trainval, index=False)
    test_rows = table[table["event_ts"] >= cut]
    test_rows.to_parquet(test, index=False)

    rng = np.random.default_rng(3)
    transactions = root / "transactions.parquet"
    pd.DataFrame(
        {
            "transaction_id": test_rows["transaction_id"],
            "gender": rng.choice(["F", "M"], len(test_rows)),
            "partition": "test",
        }
    ).to_parquet(transactions, index=False)

    raw_dir = root / "raw"
    (raw_dir / "ulb").mkdir(parents=True)
    n = 6_000
    ulb = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172_000, n)).round()})
    signal = rng.normal(size=n)
    for i in range(1, 29):
        ulb[f"V{i}"] = rng.normal(size=n) + (signal if i == 14 else 0.0)
    ulb["Amount"] = rng.exponential(80, n).round(2)
    ulb["Class"] = (signal + rng.normal(0, 0.5, n) > 2.3).astype(np.int64)
    ulb.to_csv(raw_dir / "ulb" / "creditcard.csv", index=False)

    config = root / "training.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seeds": list(TINY_CONFIG.seeds),
                "max_rounds": TINY_CONFIG.max_rounds,
                "early_stopping_rounds": TINY_CONFIG.early_stopping_rounds,
                "n_boot": TINY_CONFIG.n_boot,
                "alert_budgets": list(TINY_CONFIG.alert_budgets),
                "undersample_negatives_per_positive": 10,
                "smote_minority_ratio": 0.1,
                "smote_k_neighbors": 5,
                "base_params": TINY_CONFIG.base_params,
                "tuning": {
                    "n_trials": 2,
                    "n_folds": 2,
                    "seed": 0,
                    "space": TINY_CONFIG.tuning.space,
                },
            }
        ),
        encoding="utf-8",
    )
    costs = root / "costs.yaml"
    costs.write_text(
        "review_cost: 4.0\nmargin_rate: 0.02\nfriction_cost: 15.0\nreview_capacity_share: 0.02\n",
        encoding="utf-8",
    )
    return Workspace(
        root=root,
        trainval=trainval,
        test=test,
        transactions=transactions,
        raw_dir=raw_dir,
        config=config,
        costs=costs,
        results=root / "results",
        models=root / "models",
    )
