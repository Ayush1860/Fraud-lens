"""Offline checks of the explanations, written to ``docs/results/explainability.json``.

* **Additivity:** contributions plus bias equal the margin, on 10,000 validation rows.
* **Stability:** raise each flagged transaction's amount by 1%, recompute its features
  through ``compute_features``, and count how often its top three reason groups stay the
  same.
* **Global ranking:** mean absolute contribution per feature, to spot a leak (an id-like
  feature near the top).
* **Explanation drift:** mean absolute contribution per feature per month, for all
  traffic and for fraud rows.

Only the public training partition is read. The test partition is never touched here.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudlens.common.logging import configure_logging
from fraudlens.common.paths import TRAINVAL_FEATURES_PATH, TRANSACTIONS_PATH
from fraudlens.data.splits import assign_roles, split_bounds
from fraudlens.explain.contributions import Contributions, compute_contributions
from fraudlens.explain.reason_codes import explain, load_templates, top_groups
from fraudlens.features.definitions import FEATURE_NAMES, HISTORY_WINDOW_S, compute_features
from fraudlens.features.encoding import FEATURE_TYPES, encode_frame, encode_vector
from fraudlens.features.records import to_history_entry, to_transaction
from fraudlens.training.bundle import ModelBundle, write_json

logger = logging.getLogger(__name__)

ADDITIVITY_ROWS = 10_000
ADDITIVITY_TOLERANCE = 1e-4
STABILITY_ROWS = 1_000
DRIFT_ROWS_PER_MONTH = 20_000


def additivity_error(contributions: Contributions, margin: np.ndarray) -> float:
    """Largest absolute gap between bias plus contributions and the model's margin."""
    return float(np.abs(contributions.margin - margin).max())


def global_ranking(contributions: Contributions) -> list[dict[str, Any]]:
    """Features by mean absolute contribution, largest first."""
    mean_abs = np.abs(contributions.values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [
        {"feature": contributions.feature_names[i], "mean_abs_contribution": float(mean_abs[i])}
        for i in order
    ]


def monthly_attribution(
    contributions: Contributions, event_ts: np.ndarray, y: np.ndarray
) -> list[dict[str, Any]]:
    """Mean absolute contribution per feature per calendar month (UTC).

    Returns:
        One record per month and population (``all`` or ``fraud``), with the row count and
        a value per feature.
    """
    month = pd.to_datetime(event_ts, unit="s").strftime("%Y-%m").to_numpy()
    records = []
    for value in sorted(set(month.tolist())):
        for population, mask in (("all", month == value), ("fraud", (month == value) & (y == 1))):
            if mask.any():
                mean_abs = np.abs(contributions.values[mask]).mean(axis=0)
                records.append(
                    {
                        "month": value,
                        "population": population,
                        "rows": int(mask.sum()),
                        **{
                            n: float(v)
                            for n, v in zip(contributions.feature_names, mean_abs, strict=True)
                        },
                    }
                )
    return records


def amount_stability(
    bundle: ModelBundle,
    transactions: pd.DataFrame,
    flagged_ids: Sequence[str],
    *,
    bump: float = 0.01,
    top: int = 3,
) -> dict[str, Any]:
    """Share of flagged rows whose top reason groups survive a small change in amount.

    Each row's features are recomputed twice through ``compute_features``, from the
    card's own history: once as recorded and once with the amount raised by ``bump``.

    Args:
        bundle: The model.
        transactions: Ingested transactions holding the flagged rows and their cards'
            earlier rows.
        flagged_ids: Transaction ids to test.
        bump: Relative change in amount (0.01 is +1%).
        top: How many leading groups must stay the same, in the same order.
    """
    wanted = set(flagged_ids)
    cards = transactions.loc[transactions["transaction_id"].isin(wanted), "card_id"].unique()
    pool = transactions[transactions["card_id"].isin(cards)].sort_values(
        ["card_id", "event_ts", "transaction_id"]
    )
    templates = load_templates()
    same = 0
    total = 0
    for _, card_rows in pool.groupby("card_id", sort=False):
        records: list[dict[str, Any]] = card_rows.to_dict(orient="records")  # type: ignore[assignment]
        entries = [to_history_entry(record) for record in records]
        for position, record in enumerate(records):
            if record["transaction_id"] not in wanted:
                continue
            oldest = int(record["event_ts"]) - HISTORY_WINDOW_S
            history = [e for e in entries[:position] if e.event_ts >= oldest]
            base = compute_features(to_transaction(record), history)
            moved = compute_features(
                to_transaction(record, amount=float(record["amount"]) * (1 + bump)), history
            )
            explanations = explain(
                bundle.booster,
                np.vstack([encode_vector(base), encode_vector(moved)]),
                [base, moved],
                FEATURE_NAMES,
                FEATURE_TYPES,
                templates,
            )
            total += 1
            same += top_groups(explanations[0], top) == top_groups(explanations[1], top)
    return {
        "rows": total,
        "amount_change": bump,
        "top_groups_compared": top,
        "share_unchanged": same / total if total else None,
    }


def run_checks(
    bundle_dir: Path,
    *,
    trainval_path: Path = TRAINVAL_FEATURES_PATH,
    transactions_path: Path = TRANSACTIONS_PATH,
    seed: int = 0,
) -> dict[str, Any]:
    """Run every check against a bundle, on the public training partition only."""
    bundle = ModelBundle.load(bundle_dir)
    table = pd.read_parquet(trainval_path)
    roles = assign_roles(table["event_ts"].to_numpy(), split_bounds(table["event_ts"].to_numpy()))
    rng = np.random.default_rng(seed)

    validation = table[np.isin(roles, ["valid_early", "valid_late"])]
    n_rows = min(ADDITIVITY_ROWS, len(validation))
    sample = validation.iloc[np.sort(rng.choice(len(validation), n_rows, replace=False))]
    X = encode_frame(sample)
    contributions = compute_contributions(bundle.booster, X, FEATURE_NAMES, FEATURE_TYPES)
    error = additivity_error(contributions, bundle.margin(X))

    late = table[roles == "valid_late"]
    late_scores = bundle.score(encode_frame(late))
    flagged = late.loc[late_scores >= bundle.thresholds.t_review, "transaction_id"].to_numpy()
    chosen = rng.choice(flagged, min(STABILITY_ROWS, len(flagged)), replace=False)
    transactions = pd.read_parquet(transactions_path, filters=[("partition", "==", "train")])
    stability = amount_stability(bundle, transactions, chosen.tolist())

    month = pd.to_datetime(table["event_ts"], unit="s").dt.strftime("%Y-%m")
    keep = np.zeros(len(table), dtype=bool)
    for value in month.unique():
        rows = np.flatnonzero((month == value).to_numpy())
        keep[rng.choice(rows, min(DRIFT_ROWS_PER_MONTH, len(rows)), replace=False)] = True
    keep |= table["is_fraud"].to_numpy() == 1
    drift_rows = table[keep]
    drift = compute_contributions(
        bundle.booster, encode_frame(drift_rows), FEATURE_NAMES, FEATURE_TYPES
    )

    return {
        "bundle": bundle.version,
        "additivity": {
            "rows": n_rows,
            "max_abs_error": error,
            "tolerance": ADDITIVITY_TOLERANCE,
            "passed": error <= ADDITIVITY_TOLERANCE,
        },
        "stability": stability,
        "global_ranking": global_ranking(contributions),
        "monthly_attribution": monthly_attribution(
            drift, drift_rows["event_ts"].to_numpy(), drift_rows["is_fraud"].to_numpy()
        ),
        "notes": {
            "rows": "validation rows of the public training partition; the test set is not read",
            "drift_sample": f"up to {DRIFT_ROWS_PER_MONTH} rows per month plus every fraud row",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m fraudlens.explain.checks``."""
    parser = argparse.ArgumentParser(prog="python -m fraudlens.explain.checks")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/results/explainability.json"))
    args = parser.parse_args(argv)
    configure_logging()
    write_json(args.out, run_checks(args.bundle))
    logger.info("explainability checks written", extra={"path": str(args.out)})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
