"""Online features equal offline features, and stored decisions reproduce exactly.

2,000 synthetic transactions go through the running service one at a time, in time
order, as the replay client sends them. The features it stored in the audit table must
equal the batch builder's features for the same rows, and every stored decision must be
reproduced bit for bit from its stored feature vector and the bundle.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from conftest import ClientFactory

from fraudlens.features.batch import build_features
from fraudlens.features.definitions import FEATURE_NAMES
from fraudlens.features.encoding import encode_vector
from fraudlens.tools.replay import to_payload
from fraudlens.training.bundle import ModelBundle, logistic
from helpers import values_match
from toy_bundle import toy_transactions

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

N_REPLAYED = 2_000
N_AUDITED = 1_000


def stored_decisions(db: sa.Engine) -> pd.DataFrame:
    with db.connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT transaction_id, features, margin, score, decision, model_version "
                "FROM decisions"
            )
        ).mappings()
        return pd.DataFrame([dict(row) for row in rows]).set_index("transaction_id")


def as_feature_vector(stored: dict[str, Any] | str) -> dict[str, Any]:
    values = json.loads(stored) if isinstance(stored, str) else stored
    return {name: math.nan if value is None else value for name, value in values.items()}


async def test_online_features_match_offline_and_audit_reproduces(
    make_client: ClientFactory, db: sa.Engine, bundle_dir: Path
) -> None:
    txns = toy_transactions(n_txns=N_REPLAYED, seed=23, span_days=60)
    async with make_client() as client:
        for row in txns.to_dict(orient="records"):
            response = await client.post("/v1/transactions/score", json=to_payload(row))
            assert response.status_code == 200, response.text
            assert response.json()["degraded"] is False

    stored = stored_decisions(db)
    assert len(stored) == N_REPLAYED

    # Online features equal the offline batch builder's, row for row.
    offline = build_features(txns).set_index(txns["transaction_id"])
    wrong = [
        (tid, name, online[name], offline.at[tid, name])
        for tid, features in stored["features"].items()
        for online in [as_feature_vector(features)]
        for name in FEATURE_NAMES
        if not values_match(name, online[name], offline.at[tid, name])
    ]
    assert not wrong, wrong[:10]

    # The audit row alone reproduces the score: same features, same bundle, same number.
    bundle = ModelBundle.load(bundle_dir)
    sample = stored.sample(N_AUDITED, random_state=0)
    X = np.vstack([encode_vector(as_feature_vector(f)) for f in sample["features"]])
    margin = bundle.margin(X)
    score = bundle.calibrator(logistic(margin))
    assert (sample["model_version"] == bundle.version).all()
    np.testing.assert_array_equal(margin, sample["margin"].to_numpy())
    np.testing.assert_array_equal(score, sample["score"].to_numpy())
    decisions = np.array(["approve", "review", "block"])[bundle.decide(score)]
    np.testing.assert_array_equal(decisions, sample["decision"].to_numpy())
