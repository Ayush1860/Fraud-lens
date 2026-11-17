"""Arms, calibration, thresholds and the bundle, each on its own."""

import json
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from fraudlens.features.definitions import CATEGORIES, FEATURE_NAMES, compute_features
from fraudlens.features.encoding import encode_frame, encode_vector
from fraudlens.training.arms import fit_arm, make_arms, prior_correction, resample_training
from fraudlens.training.bundle import (
    BundleIntegrityError,
    ModelBundle,
    next_version,
    verify_bundle,
    write_bundle,
)
from fraudlens.training.calibrate import Calibrator, fit_isotonic
from fraudlens.training.config import CostConfig, load_cost_config, load_training_config
from fraudlens.training.metrics import Costs, expected_cost_per_1000
from fraudlens.training.thresholds import (
    NEVER,
    Thresholds,
    candidates,
    choose_thresholds,
    decide,
    threshold_curve,
)
from helpers import make_transactions, to_entry, to_transactions
from training_fixtures import TINY_CONFIG, toy_frame

COSTS = CostConfig(Costs(review_cost=4.0, margin_rate=0.02, friction_cost=15.0), 0.02)


def test_repository_configs_load() -> None:
    config = load_training_config()
    costs = load_cost_config()

    assert config.seeds == (0, 1, 2, 3, 4)
    assert config.tuning.n_trials == 40
    assert costs.costs.review_cost > 0


def test_arms_cover_the_plan() -> None:
    arms = make_arms(load_training_config())

    assert list(arms) == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    assert arms["A0"].learner == "logreg"
    assert arms["A2"].weighting == "full"
    assert arms["A3"].weighting == "sqrt"
    assert arms["A4"].max_delta_step == 1.0
    assert arms["A5"].undersample_ratio == 10.0
    assert arms["A6"].smote_ratio == 0.1


@pytest.mark.parametrize(
    ("p_s", "beta", "expected"),
    [(0.5, 0.1, 0.05 / 0.55), (0.0, 0.2, 0.0), (1.0, 0.2, 1.0), (0.3, 1.0, 0.3)],
)
def test_prior_correction_by_hand(p_s: float, beta: float, expected: float) -> None:
    assert prior_correction(np.array([p_s]), beta)[0] == pytest.approx(expected)


def test_undersampling_keeps_every_fraud_and_the_requested_ratio() -> None:
    frame = toy_frame()
    arm = make_arms(TINY_CONFIG)["A5"]

    out = resample_training(arm, frame, seed=0)

    n_pos = int(frame.y.sum())
    assert int(out.y.sum()) == n_pos
    assert int((out.y == 0).sum()) == 10 * n_pos
    assert out.beta == pytest.approx(10 * n_pos / int((frame.y == 0).sum()))


def test_smote_adds_only_synthetic_frauds_and_keeps_every_real_row() -> None:
    frame = toy_frame()
    arm = make_arms(TINY_CONFIG)["A6"]

    out = resample_training(arm, frame, seed=0)

    n_neg = int((frame.y == 0).sum())
    assert np.array_equal(out.X[: len(frame)], frame.X, equal_nan=True)
    assert int((out.y == 0).sum()) == n_neg
    assert int(out.y.sum()) == int(np.ceil(0.1 * n_neg))
    synthetic = out.X[len(frame) :]
    assert not np.isnan(synthetic).any()
    category = FEATURE_NAMES.index("category")
    assert set(np.unique(synthetic[:, category])) <= set(range(len(CATEGORIES)))


def test_every_arm_fits_and_returns_probabilities() -> None:
    frame = toy_frame()
    train, valid = frame.subset(np.arange(9_000)), frame.subset(np.arange(9_000, len(frame)))

    for arm in make_arms(TINY_CONFIG).values():
        fitted = fit_arm(
            arm,
            train,
            valid,
            params=TINY_CONFIG.base_params,
            seed=0,
            device="cpu",
            max_rounds=20,
            early_stopping_rounds=5,
        )
        p = fitted.predict_proba(valid.X)
        assert p.shape == (len(valid),), arm.name
        assert ((p >= 0) & (p <= 1)).all(), arm.name


def test_encoding_matches_between_a_table_and_one_vector() -> None:
    txns = make_transactions(n_cards=2, n_txns=50)
    rows = to_transactions(txns)
    entries = [to_entry(txn) for txn in rows]
    vector = compute_features(rows[10], entries[:10])

    one = encode_vector(vector)
    table = encode_frame(pd.DataFrame([vector]))

    np.testing.assert_array_equal(one, table)
    assert one[0, FEATURE_NAMES.index("category")] == CATEGORIES.index(str(vector["category"]))


def test_inplace_prediction_matches_dmatrix_prediction_with_a_categorical_feature() -> None:
    frame = toy_frame()
    fitted = fit_arm(
        make_arms(TINY_CONFIG)["A1"],
        frame,
        frame,
        params=TINY_CONFIG.base_params,
        seed=0,
        device="cpu",
        max_rounds=20,
        early_stopping_rounds=5,
    )
    assert fitted.booster is not None
    matrix = xgb.DMatrix(
        frame.X,
        feature_names=list(frame.feature_names),
        feature_types=list(frame.feature_types),
        enable_categorical=True,
    )

    np.testing.assert_allclose(
        fitted.booster.inplace_predict(frame.X), fitted.booster.predict(matrix), rtol=0, atol=1e-7
    )


def test_calibrator_reproduces_isotonic_regression() -> None:
    rng = np.random.default_rng(0)
    raw = rng.random(2_000)
    y = (rng.random(2_000) < raw**2).astype(np.int64)
    reference = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw, y)

    calibrator = fit_isotonic(raw, y)
    probe = np.concatenate([rng.random(500), [-1.0, 0.0, 1.0, 2.0]])

    np.testing.assert_allclose(calibrator(probe), reference.predict(probe), rtol=0, atol=1e-12)
    assert Calibrator.from_json(json.loads(json.dumps(calibrator.to_json()))) == calibrator
    assert (np.diff(calibrator(np.sort(probe))) >= 0).all()


def brute_force(
    p: np.ndarray, y: np.ndarray, amount: np.ndarray, config: CostConfig
) -> tuple[float, float]:
    best = np.inf
    for t_review, t_block in product(candidates(p), repeat=2):
        if t_block < t_review:
            continue
        decision = decide(p, Thresholds(float(t_review), float(t_block)))
        if (decision == 1).mean() > config.review_capacity_share + 1e-12:
            continue
        best = min(best, expected_cost_per_1000(y, amount, decision, config.costs))
    return best, 0.0


def test_chosen_thresholds_are_the_cheapest_within_capacity() -> None:
    rng = np.random.default_rng(1)
    p = np.round(rng.random(400) ** 3, 2)
    y = (rng.random(400) < p).astype(np.int64)
    amount = rng.exponential(120, 400)

    thresholds, cost = choose_thresholds(p, y, amount, COSTS)

    expected, _ = brute_force(p, y, amount, COSTS)
    assert cost == pytest.approx(expected)
    decision = decide(p, thresholds)
    assert expected_cost_per_1000(y, amount, decision, COSTS.costs) == pytest.approx(cost)
    assert (decision == 1).mean() <= COSTS.review_capacity_share
    assert thresholds.t_review <= thresholds.t_block


def test_threshold_curve_prices_any_pair_like_the_cost_function() -> None:
    rng = np.random.default_rng(2)
    p = np.round(rng.random(300), 2)
    y = (rng.random(300) < p / 2).astype(np.int64)
    amount = rng.exponential(80, 300)

    curve = threshold_curve(p, y, amount, np.repeat(np.arange(10), 30), COSTS)
    points = {point["threshold"]: point for point in curve["points"]}
    t_review, t_block = sorted(points)[40], sorted(points)[70]
    r, b = points[t_review], points[t_block]
    costs = COSTS.costs
    from_curve = (
        (
            (curve["fraud_amount"] - r["fraud_amount"])
            + costs.review_cost * (r["count"] - b["count"])
            + costs.margin_rate * b["genuine_amount"]
            + costs.friction_cost * b["genuine"]
        )
        / curve["rows"]
        * 1000
    )

    decision = decide(p, Thresholds(t_review, t_block))
    assert from_curve == pytest.approx(expected_cost_per_1000(y, amount, decision, costs))
    assert points[NEVER]["count"] == 0
    assert curve["days"] == 10


def write_toy_bundle(directory: Path) -> tuple[Path, np.ndarray]:
    frame = toy_frame()
    fitted = fit_arm(
        make_arms(TINY_CONFIG)["A1"],
        frame,
        frame,
        params=TINY_CONFIG.base_params,
        seed=0,
        device="cpu",
        max_rounds=20,
        early_stopping_rounds=5,
    )
    assert fitted.booster is not None
    raw = fitted.booster.inplace_predict(frame.X)
    path = write_bundle(
        directory,
        booster=fitted.booster,
        calibrator=fit_isotonic(raw, frame.y),
        thresholds=Thresholds(0.2, 0.6),
        files={"metrics.json": {"note": "toy"}},
    )
    return path, frame.X


def test_bundle_verifies_reloads_and_scores_like_the_booster(tmp_path: Path) -> None:
    path, X = write_toy_bundle(tmp_path / "2026-11-17.1")

    bundle = ModelBundle.load(path)

    margin = bundle.margin(X)
    np.testing.assert_allclose(bundle.raw_probability(X), 1 / (1 + np.exp(-margin)))
    assert ((bundle.score(X) >= 0) & (bundle.score(X) <= 1)).all()
    assert bundle.thresholds == Thresholds(0.2, 0.6)
    assert set(np.unique(bundle.decide(bundle.score(X)))) <= {0, 1, 2}
    contributions = bundle.booster.predict(
        xgb.DMatrix(
            X,
            feature_names=list(FEATURE_NAMES),
            feature_types=["c" if n == "category" else "q" for n in FEATURE_NAMES],
            enable_categorical=True,
        ),
        pred_contribs=True,
    )
    np.testing.assert_allclose(contributions.sum(axis=1), margin, atol=1e-4)


@pytest.mark.parametrize("tamper", ["change", "delete", "add"])
def test_bundle_refuses_to_load_after_any_change(tmp_path: Path, tamper: str) -> None:
    path, _ = write_toy_bundle(tmp_path / "2026-11-17.1")
    if tamper == "change":
        (path / "thresholds.json").write_text('{"t_review": 0.9, "t_block": 0.95}')
    elif tamper == "delete":
        (path / "calibrator.json").unlink()
    else:
        (path / "extra.json").write_text("{}")

    with pytest.raises(BundleIntegrityError):
        ModelBundle.load(path)


def test_bundle_without_manifest_is_refused(tmp_path: Path) -> None:
    path, _ = write_toy_bundle(tmp_path / "2026-11-17.1")
    (path / "manifest.json").unlink()

    with pytest.raises(BundleIntegrityError, match="manifest"):
        verify_bundle(path)


def test_versions_count_up_within_a_day(tmp_path: Path) -> None:
    today = date(2026, 11, 17)
    assert next_version(tmp_path / "missing", today) == "2026-11-17.1"
    (tmp_path / "2026-11-17.1").mkdir()
    (tmp_path / "2026-11-17.2").mkdir()
    (tmp_path / "2026-11-16.9").mkdir()

    assert next_version(tmp_path, today) == "2026-11-17.3"
