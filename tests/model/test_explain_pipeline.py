"""The offline explanation checks and the golden-case proposal, end to end on toy data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudlens.data.splits import assign_roles, split_bounds
from fraudlens.explain import checks, golden
from fraudlens.features.batch import KEY_COLUMNS, build_features
from fraudlens.features.definitions import CATEGORIES
from fraudlens.training.arms import make_arms
from fraudlens.training.config import CostConfig
from fraudlens.training.data import sparkov_frame
from fraudlens.training.metrics import Costs
from fraudlens.training.release import build_release
from helpers import make_transactions
from training_fixtures import TINY_CONFIG


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Toy transactions, their feature table and a released bundle."""
    root = tmp_path_factory.mktemp("explain")
    txns = make_transactions(n_cards=60, n_txns=12_000, seed=5, span_days=240)
    features = build_features(txns)
    rng = np.random.default_rng(5)
    logit = -6.5 + 0.6 * np.log1p(txns["amount"].to_numpy()) + 1.5 * features["is_night"]
    txns["is_fraud"] = (rng.random(len(txns)) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    txns["partition"] = "train"

    transactions = root / "transactions.parquet"
    trainval = root / "trainval.parquet"
    txns.to_parquet(transactions, index=False)
    table = pd.concat([txns[list(KEY_COLUMNS)], features], axis=1)
    table.to_parquet(trainval, index=False)

    frame = sparkov_frame(table)
    roles = assign_roles(frame.event_ts, split_bounds(frame.event_ts))
    bundle = build_release(
        frame,
        roles,
        arm=make_arms(TINY_CONFIG)["A1"],
        params=TINY_CONFIG.base_params,
        config=TINY_CONFIG,
        costs=CostConfig(Costs(4.0, 0.02, 15.0), 0.05),
        categories=CATEGORIES,
        device="cpu",
        models_dir=root / "models",
        raw_dir=root / "raw",
    )
    return {"bundle": bundle, "trainval": trainval, "transactions": transactions, "root": root}


def test_checks_run_end_to_end(toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "STABILITY_ROWS", 30)

    results = checks.run_checks(
        toy["bundle"], trainval_path=toy["trainval"], transactions_path=toy["transactions"]
    )

    assert results["additivity"]["passed"]
    assert results["stability"]["rows"] == 30
    assert 0.0 <= results["stability"]["share_unchanged"] <= 1.0
    assert len(results["global_ranking"]) == 27
    assert {r["population"] for r in results["monthly_attribution"]} == {"all", "fraud"}


def test_checks_command_line_writes_json(
    toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = toy["root"] / "explainability.json"
    monkeypatch.setattr(checks, "STABILITY_ROWS", 10)
    original = checks.run_checks
    monkeypatch.setattr(
        checks,
        "run_checks",
        lambda bundle: original(
            bundle, trainval_path=toy["trainval"], transactions_path=toy["transactions"]
        ),
    )

    assert checks.main(["--bundle", str(toy["bundle"]), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["bundle"] == toy["bundle"].name


def test_golden_proposal_covers_the_worded_groups(
    toy: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = golden.propose(toy["bundle"], n=10, trainval_path=toy["trainval"])

    assert len(cases) == 10
    assert len({case["transaction_id"] for case in cases}) == 10
    for case in cases:
        assert case["decision"] in {"review", "block"}
        assert len(case["reasons"]) == len(case["reason_ids"]) <= 5

    out = toy["root"] / "proposal"
    monkeypatch.setattr(golden, "propose", lambda bundle: cases)
    assert golden.main(["--bundle", str(toy["bundle"]), "--out", str(out)]) == 0
    assert json.loads(out.with_suffix(".json").read_text()) == cases
    assert "| 10 |" in out.with_suffix(".md").read_text()
