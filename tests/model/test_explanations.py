"""Contributions are exact, independently confirmed, and worded faithfully."""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from shapley_oracle import predict_margin, shapley

from fraudlens.explain.checks import (
    additivity_error,
    amount_stability,
    global_ranking,
    monthly_attribution,
    run_checks,
)
from fraudlens.explain.contributions import Contributions, compute_contributions
from fraudlens.explain.reason_codes import (
    Templates,
    build_explanation,
    explain,
    load_templates,
    render,
    top_groups,
)
from fraudlens.features.definitions import CATEGORIES, FEATURE_NAMES, compute_features
from fraudlens.features.encoding import FEATURE_TYPES, encode_vector
from fraudlens.training.arms import fit_arm, make_arms
from fraudlens.training.bundle import ModelBundle
from training_fixtures import TINY_CONFIG, toy_frame

TEMPLATES = load_templates()


@pytest.fixture(scope="module")
def model() -> tuple[xgb.Booster, np.ndarray]:
    """A small booster with categorical splits, and its training rows."""
    frame = toy_frame(12_000)
    fitted = fit_arm(
        make_arms(TINY_CONFIG)["A1"],
        frame,
        frame,
        params={"max_depth": 3, "learning_rate": 0.3},
        seed=0,
        device="cpu",
        max_rounds=20,
        early_stopping_rounds=10,
    )
    assert fitted.booster is not None
    assert fitted.booster.get_score(importance_type="weight").get("category")
    return fitted.booster, frame.X


def test_contributions_add_up_to_the_margin_on_10000_rows(
    model: tuple[xgb.Booster, np.ndarray],
) -> None:
    booster, X = model
    rows = X[:10_000]

    contributions = compute_contributions(booster, rows, FEATURE_NAMES, FEATURE_TYPES)

    margin = booster.inplace_predict(rows, predict_type="margin")
    assert additivity_error(contributions, margin) <= 1e-4


def test_contributions_agree_with_an_independent_brute_force_shapley(
    model: tuple[xgb.Booster, np.ndarray],
) -> None:
    # shap.TreeExplainer would delegate straight back to XGBoost for this model, so the
    # second opinion is a from-scratch Shapley computation over the dumped trees.
    booster, X = model
    rows = X[:200]

    oracle_margin = predict_margin(booster, rows, FEATURE_NAMES)
    phi, bias = shapley(booster, rows, FEATURE_NAMES)
    contributions = compute_contributions(booster, rows, FEATURE_NAMES, FEATURE_TYPES)

    np.testing.assert_allclose(
        oracle_margin, booster.inplace_predict(rows, predict_type="margin"), atol=1e-5
    )
    np.testing.assert_allclose(contributions.values, phi, atol=1e-5)
    np.testing.assert_allclose(contributions.bias, bias, atol=1e-5)


def test_every_feature_is_in_exactly_one_group() -> None:
    members = [name for features in TEMPLATES.groups.values() for name in features]

    assert sorted(members) == sorted(FEATURE_NAMES)
    assert all(re.fullmatch(r"R\d{2}_[A-Z_]+", gid) for gid in TEMPLATES.groups)
    assert set(TEMPLATES.category_labels) == set(CATEGORIES)


def test_grouping_rejects_a_bad_partition() -> None:
    contributions = Contributions(("a", "b"), np.ones((1, 2)), np.zeros(1))

    with pytest.raises(ValueError, match="ungrouped"):
        contributions.grouped({"G1": ["a"]})
    with pytest.raises(ValueError, match="repeated"):
        contributions.grouped({"G1": ["a", "b"], "G2": ["b"]})


def features(**changes: float | int | str) -> dict[str, float | int | str]:
    base: dict[str, float | int | str] = dict.fromkeys(FEATURE_NAMES, 0)
    base.update(
        amount=336.10,
        category="shopping_net",
        hour=3,
        is_night=1,
        amt_mean_30d=41.0,
        amt_ratio_30d=8.2,
        n_prior_30d=12,
        txn_count_1h=0,
        txn_count_24h=1,
        txn_count_7d=4,
        amt_sum_24h=55.0,
        secs_since_prev=2_400.0,
        dist_prev_km=620.0,
        dist_home_km=80.0,
        is_new_merchant_30d=1,
        is_new_category_30d=0,
        n_merchants_24h=3,
    )
    base.update(changes)
    return base


@pytest.mark.parametrize(
    ("group", "changes", "expected"),
    [
        ("R01_AMOUNT_VS_PATTERN", {}, "Amount is 8.2 times this card's 30-day average of $41"),
        (
            "R01_AMOUNT_VS_PATTERN",
            {"amt_ratio_30d": math.nan, "n_prior_30d": 0},
            "Amount of $336.10 on a card with no purchases in the last 30 days",
        ),
        ("R01_AMOUNT_VS_PATTERN", {"amt_ratio_30d": 1.1}, "Amount of $336.10"),
        ("R02_VELOCITY", {"txn_count_1h": 6}, "6 other purchases on this card in the last hour"),
        (
            "R02_VELOCITY",
            {"txn_count_24h": 4, "amt_sum_24h": 912.4},
            "4 other purchases totalling $912 on this card in the last 24 hours",
        ),
        ("R02_VELOCITY", {}, "Previous purchase on this card 40 minutes earlier"),
        (
            "R02_VELOCITY",
            {"secs_since_prev": 90_000.0},
            "4 other purchases on this card in the last 7 days",
        ),
        ("R03_TRAVEL", {}, "620 km from the previous purchase 40 minutes earlier"),
        (
            "R03_TRAVEL",
            {"dist_prev_km": math.nan, "secs_since_prev": math.nan},
            "80 km from the cardholder's home",
        ),
        ("R04_NOVELTY", {}, "First purchase at this merchant in 30 days"),
        (
            "R04_NOVELTY",
            {"is_new_merchant_30d": 0, "is_new_category_30d": 1},
            "First online shopping purchase on this card in 30 days",
        ),
        (
            "R04_NOVELTY",
            {"is_new_merchant_30d": 0},
            "3 different merchants on this card in the last 24 hours",
        ),
        (
            "R05_TIME_AND_CATEGORY",
            {},
            "Online shopping purchase at night, in the 03:00 hour (UTC)",
        ),
        (
            "R05_TIME_AND_CATEGORY",
            {"hour": 14, "is_night": 0, "category": "travel"},
            "Travel purchase in the 14:00 hour (UTC)",
        ),
    ],
)
def test_templates_are_filled_with_the_transactions_own_values(
    group: str, changes: dict[str, float | int | str], expected: str
) -> None:
    assert render(group, features(**changes), TEMPLATES) == expected


def test_cardholder_context_is_never_worded() -> None:
    assert render("R06_CARDHOLDER_CONTEXT", features(), TEMPLATES) is None


def test_reasons_are_positive_ranked_capped_and_shared_fairly() -> None:
    group_ids = list(TEMPLATES.groups)
    values = np.array([1.5, 0.5, -0.8, 0.25, 0.75, 2.0])  # R06 context is the largest push
    names = list(FEATURE_NAMES)
    contributions = np.zeros(len(names))

    explanation = build_explanation(names, contributions, -6.0, values, features(), TEMPLATES)

    ids = [reason.id for reason in explanation.reasons]
    assert ids == ["R01_AMOUNT_VS_PATTERN", "R05_TIME_AND_CATEGORY", "R02_VELOCITY", "R04_NOVELTY"]
    assert "R06_CARDHOLDER_CONTEXT" not in ids
    upward = 1.5 + 0.5 + 0.25 + 0.75 + 2.0
    assert explanation.reasons[0].share == pytest.approx(1.5 / upward)
    assert sum(r.share for r in explanation.reasons) < 1.0
    assert explanation.mitigating is not None
    assert explanation.mitigating.id == group_ids[2]
    assert explanation.mitigating.contribution == -0.8


def test_at_most_five_reasons() -> None:
    templates = Templates(
        groups={f"R{i:02d}_G": (f"f{i}",) for i in range(1, 8)},
        titles={},
        wording={},
        category_labels={},
    )
    words = {gid: {"x": "text"} for gid in templates.groups}
    templates = Templates(templates.groups, {}, words, {})

    def fake_render(*_: object) -> str:
        return "text"

    import fraudlens.explain.reason_codes as module

    original = module.render
    module.render = fake_render  # type: ignore[assignment]
    try:
        explanation = build_explanation(
            [f"f{i}" for i in range(1, 8)],
            np.ones(7),
            0.0,
            np.arange(1, 8, dtype=float),
            features(),
            templates,
        )
    finally:
        module.render = original
    assert len(explanation.reasons) == 5
    assert [r.contribution for r in explanation.reasons] == [7.0, 6.0, 5.0, 4.0, 3.0]


def test_explain_end_to_end_adds_up(model: tuple[xgb.Booster, np.ndarray]) -> None:
    booster, _ = model
    from helpers import make_transactions, to_entry, to_transactions

    txns = to_transactions(make_transactions(n_cards=3, n_txns=60, seed=4))
    entries = [to_entry(t) for t in txns]
    rows = [compute_features(t, entries[:i]) for i, t in enumerate(txns)]
    encoded = np.vstack([encode_vector(row) for row in rows])

    explanations = explain(booster, encoded, rows, FEATURE_NAMES, FEATURE_TYPES)

    margin = booster.inplace_predict(encoded, predict_type="margin")
    for explanation, expected in zip(explanations, margin, strict=True):
        assert explanation.margin == pytest.approx(float(expected), abs=1e-4)
        assert explanation.bias + sum(explanation.groups.values()) == pytest.approx(
            explanation.margin, abs=1e-9
        )
        assert len(explanation.reasons) <= 5
        assert all(r.contribution > 0 for r in explanation.reasons)
        assert len(top_groups(explanation)) <= 3


def test_global_ranking_and_monthly_drift_tables() -> None:
    contributions = Contributions(("a", "b"), np.array([[1.0, -3.0], [-1.0, 1.0]]), np.zeros(2))

    ranking = global_ranking(contributions)
    drift = monthly_attribution(
        contributions, np.array([1_577_836_800, 1_580_515_200]), np.array([0, 1])
    )

    assert [r["feature"] for r in ranking] == ["b", "a"]
    assert ranking[0]["mean_abs_contribution"] == 2.0
    assert [(d["month"], d["population"]) for d in drift] == [
        ("2020-01", "all"),
        ("2020-02", "all"),
        ("2020-02", "fraud"),
    ]
    assert drift[1]["a"] == 1.0


def test_nothing_under_src_imports_shap() -> None:
    sources = Path("src").rglob("*.py")
    offenders = [
        path.as_posix()
        for path in sources
        if re.search(r"^\s*(import shap|from shap)", path.read_text(encoding="utf-8"), re.M)
    ]
    assert offenders == []


def newest_bundle() -> Path | None:
    bundles = sorted(Path("models").glob("*/manifest.json"))
    return bundles[-1].parent if bundles else None


@pytest.mark.data
def test_real_bundle_explanations_pass_every_check() -> None:
    bundle_dir = newest_bundle()
    if bundle_dir is None or not Path("data/processed/trainval.parquet").exists():
        pytest.skip("needs a local bundle and the built data")

    results = run_checks(bundle_dir)

    assert results["additivity"]["passed"], results["additivity"]
    assert results["stability"]["rows"] > 0
    top = [r["feature"] for r in results["global_ranking"][:5]]
    assert not any("id" in name for name in top)


@pytest.mark.data
def test_stability_on_a_handful_of_real_rows() -> None:
    bundle_dir = newest_bundle()
    path = Path("data/interim/transactions.parquet")
    if bundle_dir is None or not path.exists():
        pytest.skip("needs a local bundle and the built data")
    transactions = pd.read_parquet(path, filters=[("partition", "==", "train")])
    ids = transactions["transaction_id"].iloc[-20:].tolist()

    result = amount_stability(ModelBundle.load(bundle_dir), transactions, ids)

    assert result["rows"] == 20
    assert 0.0 <= result["share_unchanged"] <= 1.0


def test_golden_case_table_lists_every_case() -> None:
    from fraudlens.explain.golden import to_markdown

    case = {
        "transaction_id": "t1",
        "is_fraud": 1,
        "amount": 899.48,
        "category": "shopping_net",
        "score": 1.0,
        "decision": "block",
        "reason_ids": ["R01_AMOUNT_VS_PATTERN"],
        "reasons": ["Amount of $899.48 (53%)", "First purchase at this merchant in 30 days (1%)"],
        "mitigating": None,
    }

    table = to_markdown([case, {**case, "is_fraud": 0, "mitigating": "Dining purchase"}], "v1")

    assert "| 1 | yes | $899.48 | shopping_net | 1.000 | block |" in table
    assert "Amount of $899.48 (53%)<br>First purchase" in table
    assert "| 2 | no |" in table
    assert "`v1`" in table


GOLDEN = Path(__file__).with_name("golden_cases.json")


@pytest.mark.data
def test_golden_cases_keep_their_leading_reasons() -> None:
    bundle_dir = newest_bundle()
    if not GOLDEN.exists():
        pytest.skip("golden cases not frozen yet; see docs/results/golden_cases_proposed.md")
    if bundle_dir is None or not Path("data/processed/trainval.parquet").exists():
        pytest.skip("needs a local bundle and the built data")
    import json

    from fraudlens.explain.reason_codes import as_rows
    from fraudlens.features.encoding import encode_frame

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    table = pd.read_parquet("data/processed/trainval.parquet").set_index("transaction_id")
    rows = table.loc[[case["transaction_id"] for case in golden]].reset_index()
    bundle = ModelBundle.load(bundle_dir)

    explanations = explain(
        bundle.booster,
        encode_frame(rows),
        as_rows(rows[list(FEATURE_NAMES)]),
        FEATURE_NAMES,
        FEATURE_TYPES,
    )

    for case, explanation in zip(golden, explanations, strict=True):
        expected = case["expected_leading_reason_ids"]
        actual = [reason.id for reason in explanation.reasons][: len(expected)]
        assert actual == expected, case["transaction_id"]
