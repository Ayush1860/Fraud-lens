"""The whole training command line, end to end, on toy data."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudlens.data import schema
from fraudlens.training.bundle import ModelBundle
from fraudlens.training.cli import main
from fraudlens.training.data import sparkov_frame
from fraudlens.training.evaluate import AlreadyEvaluatedError, final_test
from fraudlens.training.experiments import summarise
from training_fixtures import TINY_CONFIG, Workspace, make_workspace, toy_frame


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    return make_workspace(tmp_path_factory.mktemp("training"))


def cli(workspace: Workspace, step: str, *extra: str) -> int:
    return main(
        [
            step,
            "--config",
            str(workspace.config),
            "--costs",
            str(workspace.costs),
            "--results-dir",
            str(workspace.results),
            "--models-dir",
            str(workspace.models),
            "--trainval",
            str(workspace.trainval),
            "--test",
            str(workspace.test),
            "--transactions",
            str(workspace.transactions),
            "--raw-dir",
            str(workspace.raw_dir),
            *extra,
        ]
    )


def test_pipeline_runs_every_step_and_writes_its_artefacts(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    toy_ulb = pd.read_csv(workspace.raw_dir / "ulb" / "creditcard.csv")
    monkeypatch.setattr(schema, "ULB_ROWS", len(toy_ulb))
    monkeypatch.setattr(schema, "ULB_FRAUDS", int(toy_ulb["Class"].sum()))
    for step in ("experiments", "tune", "release", "evaluate", "ulb", "report"):
        assert cli(workspace, step) == 0, step

    study = json.loads((workspace.results / "experiments.json").read_text())
    assert [arm["arm"] for arm in study["arms"]] == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    for arm in study["arms"]:
        assert len(arm["per_seed"]) == len(TINY_CONFIG.seeds)
        low, high = arm["summary"]["average_precision"]["ci95"]
        assert low <= high

    tuning = json.loads((workspace.results / "tuning.json").read_text())
    assert tuning["n_trials"] == 2
    assert len(tuning["folds"]) == 2
    for fold in tuning["folds"]:
        assert fold["train_end_ts"] < fold["valid_start_ts"]

    (bundle_dir,) = list(workspace.models.iterdir())
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}\.1", bundle_dir.name)
    bundle = ModelBundle.load(bundle_dir)
    assert bundle.documents["training.json"]["arm"] == tuning["arm"]
    assert bundle.documents["training.json"]["tuned"] is True
    assert "test_metrics.json" in bundle.documents
    assert bundle.thresholds.t_review <= bundle.thresholds.t_block

    test = bundle.documents["test_metrics.json"]
    assert {row["group"] for row in test["fairness"]["gender"]} == {"F", "M"}
    assert len(test["fairness"]["age_band"]) == 5
    assert sum(row["rows"] for row in test["decisions"].values()) == test["rows"]

    ulb = json.loads((workspace.results / "ulb.json").read_text())
    assert [arm["arm"] for arm in ulb["arms"]] == ["A1", "A2", "A3", "A4", "A5", "A6"]

    report = (workspace.results / "report.md").read_text()
    for heading in ("Imbalance study", "Tuning", "Final test", "Fairness check", "ULB benchmark"):
        assert heading in report
    assert (workspace.results / "figures" / "pr_curves.png").exists()
    assert (workspace.results / "figures" / "reliability_test.png").exists()

    # The reproduction sample: a fresh load scores exactly what the release recorded.
    reproduction = bundle.documents["reproduction.json"]
    trainval = pd.read_parquet(workspace.trainval).set_index("transaction_id")
    rows = trainval.loc[reproduction["transaction_ids"]].reset_index()
    frame = sparkov_frame(rows)
    np.testing.assert_array_equal(bundle.score(frame.X), np.array(reproduction["score"]))

    # The test set is scored once only.
    with pytest.raises(AlreadyEvaluatedError):
        final_test(bundle_dir, test_path=workspace.test, transactions_path=workspace.transactions)


def test_release_refuses_tuning_for_a_different_arm(workspace: Workspace) -> None:
    (workspace.results).mkdir(exist_ok=True)
    tuning_path = workspace.results / "tuning.json"
    original = tuning_path.read_text() if tuning_path.exists() else None
    tuning_path.write_text(json.dumps({"arm": "A9", "best_params": {}}))
    try:
        with pytest.raises(SystemExit, match="A9"):
            cli(workspace, "release", "--arm", "A1")
    finally:
        if original is None:
            tuning_path.unlink()
        else:
            tuning_path.write_text(original)


def test_bootstrap_intervals_contain_the_point_estimate_for_a_stable_metric() -> None:
    frame = toy_frame()
    rng = np.random.default_rng(0)
    p = np.clip(frame.y * 0.6 + rng.random(len(frame)) * 0.5, 0, 1)

    summary = summarise([p, p], frame, (0.01,), n_boot=50)

    for name in ("average_precision", "recall_at_1pct", "brier"):
        low, high = summary[name]["ci95"]
        assert low <= summary[name]["mean"] <= high, name
        assert summary[name]["seed_std"] == 0.0


def test_the_test_features_are_read_only_by_final_test() -> None:
    sources = Path("src/fraudlens").rglob("*.py")
    readers = {
        path.as_posix()
        for path in sources
        if "TEST_FEATURES_PATH" in path.read_text(encoding="utf-8")
    }

    # paths.py defines it, batch.py writes it, evaluate.py reads it, cli.py passes it on.
    assert readers == {
        "src/fraudlens/common/paths.py",
        "src/fraudlens/features/batch.py",
        "src/fraudlens/training/evaluate.py",
        "src/fraudlens/training/cli.py",
    }
