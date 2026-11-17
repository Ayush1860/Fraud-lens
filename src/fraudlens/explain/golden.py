"""Propose golden cases: real flagged transactions whose reasons a person signs off.

The proposal picks flagged transactions from the late validation slice so that, between
them, every worded reason group leads at least once, mixing confirmed frauds with genuine
purchases the model flagged. A person reads the table and agrees or disagrees with each
row's reasons; the approved rows are then frozen as ``tests/model/golden_cases.json`` and
a test holds every future model to the same leading reasons.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudlens.common.paths import TRAINVAL_FEATURES_PATH
from fraudlens.data.splits import assign_roles, split_bounds
from fraudlens.explain.reason_codes import as_rows, explain, load_templates
from fraudlens.features.definitions import FEATURE_NAMES
from fraudlens.features.encoding import FEATURE_TYPES, encode_frame
from fraudlens.training.bundle import ModelBundle

DECISIONS = ("approve", "review", "block")


def propose(
    bundle_dir: Path,
    *,
    n: int = 20,
    seed: int = 0,
    trainval_path: Path = TRAINVAL_FEATURES_PATH,
) -> list[dict[str, Any]]:
    """Pick ``n`` flagged late-validation transactions covering every reason group."""
    bundle = ModelBundle.load(bundle_dir)
    table = pd.read_parquet(trainval_path)
    ts = table["event_ts"].to_numpy()
    late = table[assign_roles(ts, split_bounds(ts)) == "valid_late"].reset_index(drop=True)
    scores = bundle.score(encode_frame(late))
    flagged = late[scores >= bundle.thresholds.t_review].copy()
    flagged["score"] = scores[scores >= bundle.thresholds.t_review]

    templates = load_templates()
    explanations = explain(
        bundle.booster,
        encode_frame(flagged),
        as_rows(flagged[list(FEATURE_NAMES)]),
        FEATURE_NAMES,
        FEATURE_TYPES,
        templates,
    )
    flagged["lead"] = [e.reasons[0].id if e.reasons else "" for e in explanations]

    rng = np.random.default_rng(seed)
    worded = [gid for gid, words in templates.wording.items() if words]
    picked: list[int] = []
    # One fraud and one genuine row led by each worded group, where they exist ...
    for gid in worded:
        for label in (1, 0):
            pool = np.flatnonzero(
                (flagged["lead"] == gid).to_numpy() & (flagged["is_fraud"] == label).to_numpy()
            )
            pool = np.setdiff1d(pool, picked)
            if len(pool):
                picked.append(int(rng.choice(pool)))
    # ... then fill up with a random mix of the remaining flagged rows.
    rest = np.setdiff1d(np.arange(len(flagged)), picked)
    picked += rng.choice(rest, max(0, n - len(picked)), replace=False).tolist()

    cases = []
    for position in sorted(
        picked[:n], key=lambda p: (flagged["lead"].iloc[p], -flagged["score"].iloc[p])
    ):
        row = flagged.iloc[position]
        explanation = explanations[position]
        cases.append(
            {
                "transaction_id": str(row["transaction_id"]),
                "is_fraud": int(row["is_fraud"]),
                "amount": float(row["amount"]),
                "category": str(row["category"]),
                "score": float(row["score"]),
                "decision": DECISIONS[int(bundle.decide(np.array([row["score"]]))[0])],
                "reason_ids": [r.id for r in explanation.reasons],
                "reasons": [f"{r.text} ({r.share:.0%})" for r in explanation.reasons],
                "mitigating": explanation.mitigating.text if explanation.mitigating else None,
            }
        )
    return cases


def to_markdown(cases: Sequence[dict[str, Any]], bundle: str) -> str:
    """Render the proposal as a review table."""
    lines = [
        "# Golden cases — proposed, awaiting review",
        "",
        f"Flagged transactions from the late validation slice, scored by bundle `{bundle}`. "
        "For each row, decide whether the reasons are the ones you would expect for that "
        "transaction. Approved rows are frozen with their leading reason ids.",
        "",
        "| # | Fraud? | Amount | Category | Score | Decision "
        "| Reasons (share of upward push) | Mitigating |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for number, case in enumerate(cases, 1):
        lines.append(
            f"| {number} | {'yes' if case['is_fraud'] else 'no'} | ${case['amount']:,.2f} "
            f"| {case['category']} | {case['score']:.3f} | {case['decision']} "
            f"| {'<br>'.join(case['reasons'])} | {case['mitigating'] or ''} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m fraudlens.explain.golden``."""
    parser = argparse.ArgumentParser(prog="python -m fraudlens.explain.golden")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/results/golden_cases_proposed"))
    args = parser.parse_args(argv)
    cases = propose(args.bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(to_markdown(cases, args.bundle.name), encoding="utf-8")
    print(args.out.with_suffix(".md"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
