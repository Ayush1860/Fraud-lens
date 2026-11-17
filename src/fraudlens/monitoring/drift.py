"""The drift report: what has moved in recent traffic, and what deserves an alarm.

Three comparisons, each with the population stability index:

* **Features against training.** Every feature over the newest window of decisions,
  against the reference histograms in the model bundle. Calendar features are reported
  but never alarm: a window of about a day cannot match a weekly cycle.
* **Amount within each category, newest window against the one before.** Overall amount
  hides a shift confined to one category: amounts in a category with 7.5% of traffic
  multiplied by 1.5 move overall amount PSI by about 0.01. Within the category the same
  shift gives about 0.34. The bundle has no per-category reference, so the previous window
  is the baseline; this catches sudden shifts, not slow ones.
* **Decision mix against validation.** The share of approvals, reviews and blocks against
  the shares on the validation slice. It replaces score PSI: the bundle's score histogram
  is degenerate (almost every validation score is near 0, so its decile edges collapse).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fraudlens.monitoring.psi import (
    expected_shares,
    psi,
    psi_between,
    shares_for_reference,
    status,
)

CYCLICAL: tuple[str, ...] = ("day_of_week", "is_weekend")
"""Features whose natural cycle is longer than the window; reported, never alarmed."""

MIN_SEGMENT_ROWS = 200
"""Rows a category needs in each window before its amount PSI is judged."""

DECISION_KINDS: tuple[str, ...] = ("approve", "review", "block")

Row = Mapping[str, object]


def _column(rows: Sequence[Row], name: str, *, categorical: bool) -> np.ndarray[Any, Any]:
    if categorical:
        return np.array([row.get(name) for row in rows], dtype=object)
    values = [row.get(name) for row in rows]
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)  # type: ignore[arg-type]


def _amounts(rows: Sequence[Row], category: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.array(
        [float(row["amount"]) for row in rows if row.get("category") == category],  # type: ignore[arg-type]
        dtype=np.float64,
    )


def drift_report(
    latest: Sequence[Row],
    previous: Sequence[Row],
    decisions: Sequence[str],
    *,
    reference: Mapping[str, Any],
    validation_mix: Mapping[str, float],
) -> dict[str, Any]:
    """Compare recent traffic with training, with itself, and with validation.

    Args:
        latest: Feature vectors of the newest decisions (the live window).
        previous: Feature vectors of the decisions just before them (the baseline window
            for the per-category comparison).
        decisions: Decisions in the live window: ``approve``, ``review`` or ``block``.
        reference: The bundle's ``reference.json``.
        validation_mix: Share of each decision on the validation slice.

    Returns:
        ``features``, ``segments`` and ``decision_mix``, each with PSI and status, and
        ``alarms``: everything over the alarm level except calendar features.
    """
    features = []
    for name, spec in reference["features"].items():
        value = None
        if latest:
            column = _column(latest, name, categorical=spec["kind"] == "categorical")
            value = psi(expected_shares(spec), shares_for_reference(column, spec))
        features.append(
            {"feature": name, "psi": value, "status": status(value), "cyclical": name in CYCLICAL}
        )

    categories = sorted(
        {str(row["category"]) for row in [*latest, *previous] if row.get("category") is not None}
    )
    segments = []
    for category in categories:
        live, base = _amounts(latest, category), _amounts(previous, category)
        value = psi_between(base, live) if min(len(live), len(base)) >= MIN_SEGMENT_ROWS else None
        segments.append(
            {
                "category": category,
                "psi": value,
                "status": status(value),
                "n_live": len(live),
                "n_previous": len(base),
            }
        )
    # Largest PSI first; categories too small to judge go last.
    segments.sort(key=lambda s: (s["psi"] is None, -float(s["psi"] or 0.0)))

    live_mix = dict.fromkeys(DECISION_KINDS, 0.0)
    mix_psi = None
    if decisions:
        for decision in decisions:
            live_mix[decision] += 1 / len(decisions)
        mix_psi = psi(
            np.array([validation_mix.get(kind, 0.0) for kind in DECISION_KINDS]),
            np.array([live_mix[kind] for kind in DECISION_KINDS]),
        )

    alarms = [
        {"kind": "feature", "name": f["feature"], "psi": f["psi"]}
        for f in features
        if f["status"] == "alarm" and not f["cyclical"]
    ]
    alarms += [
        {"kind": "segment", "name": f"amount in {s['category']}", "psi": s["psi"]}
        for s in segments
        if s["status"] == "alarm"
    ]
    if status(mix_psi) == "alarm":
        alarms.append({"kind": "decision_mix", "name": "decision mix", "psi": mix_psi})

    return {
        "window": len(latest),
        "previous_window": len(previous),
        "features": features,
        "segments": segments,
        "decision_mix": {
            "psi": mix_psi,
            "status": status(mix_psi),
            "live": live_mix,
            "validation": {kind: float(validation_mix.get(kind, 0.0)) for kind in DECISION_KINDS},
        },
        "alarms": alarms,
    }
