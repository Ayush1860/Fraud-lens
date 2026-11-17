"""Turn grouped contributions into plain-language reason codes.

A reason is a feature group whose summed contribution pushes the score up, worded with
the transaction's own values. Rules:

* Only groups with a positive contribution and a template become reasons, at most five,
  largest first. The cardholder-context group is never shown.
* Each reason's ``share`` is its part of the total upward push: its contribution divided
  by the sum of all positive group contributions. Contributions are in log-odds, so they
  do not add up in probability units; share is the honest unit to show.
* The largest negative group with a template is returned as the mitigating factor.

A reason says why the model scored high. It is not evidence that fraud happened.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

import numpy as np
import xgboost as xgb
import yaml

from fraudlens.explain.contributions import FloatArray, compute_contributions

FeatureValues = Mapping[str, float | int | str]

MAX_REASONS = 5
RATIO_WORTH_STATING = 1.5
"""Below this multiple of the 30-day average, the amount template states the amount only."""


@dataclass(frozen=True, slots=True)
class Templates:
    """Groups, their member features and wording, read from ``templates.yaml``."""

    groups: dict[str, tuple[str, ...]]
    titles: dict[str, str]
    wording: dict[str, dict[str, str]]
    category_labels: dict[str, str]


@cache
def load_templates() -> Templates:
    """Read ``templates.yaml`` shipped with the package."""
    raw: dict[str, Any] = yaml.safe_load(
        files("fraudlens.explain").joinpath("templates.yaml").read_text(encoding="utf-8")
    )
    groups = raw["groups"]
    return Templates(
        groups={gid: tuple(spec["features"]) for gid, spec in groups.items()},
        titles={gid: str(spec["title"]) for gid, spec in groups.items()},
        wording={gid: dict(spec.get("templates") or {}) for gid, spec in groups.items()},
        category_labels=dict(raw["category_labels"]),
    )


@dataclass(frozen=True, slots=True)
class ReasonCode:
    """One reason, as stored in the audit row and shown in the console.

    Attributes:
        id: Stable group id, for example ``R01_AMOUNT_VS_PATTERN``.
        text: Plain-language wording with the transaction's values.
        contribution: The group's summed contribution, in log-odds.
        share: Part of the total upward push (0 to 1); negative for a mitigating factor.
    """

    id: str
    text: str
    contribution: float
    share: float


@dataclass(frozen=True, slots=True)
class Explanation:
    """Everything needed to explain one scored transaction.

    Attributes:
        bias: The model's expected margin, the start of the waterfall.
        margin: Bias plus all contributions: the row's margin, in log-odds.
        contributions: Contribution of each feature, in log-odds.
        groups: Summed contribution of each group, in log-odds.
        reasons: Up to five reasons, largest first.
        mitigating: The largest factor pulling the score down, if any can be worded.
    """

    bias: float
    margin: float
    contributions: dict[str, float]
    groups: dict[str, float]
    reasons: tuple[ReasonCode, ...]
    mitigating: ReasonCode | None


def _number(features: FeatureValues, name: str) -> float:
    return float(features[name])


def render(group_id: str, features: FeatureValues, templates: Templates) -> str | None:
    """Word one group for one transaction, or ``None`` if the group has no wording.

    Args:
        group_id: The group, for example ``R02_VELOCITY``.
        features: The transaction's feature values, as ``compute_features`` or
            ``transaction_only_features`` returns them. When the history is unknown,
            pace and novelty are not worded, since nothing about them was observed.
        templates: Groups and wording.
    """
    words = templates.wording.get(group_id) or {}
    if not words:
        return None
    label = templates.category_labels.get(str(features["category"]), str(features["category"]))

    if group_id == "R01_AMOUNT_VS_PATTERN":
        ratio = _number(features, "amt_ratio_30d")
        if not math.isnan(ratio) and ratio >= RATIO_WORTH_STATING:
            return words["ratio"].format(ratio=ratio, average=_number(features, "amt_mean_30d"))
        if _number(features, "n_prior_30d") == 0:
            return words["no_history"].format(amount=_number(features, "amount"))
        return words["plain"].format(amount=_number(features, "amount"))

    history_unknown = math.isnan(_number(features, "n_prior_30d"))

    if group_id == "R02_VELOCITY":
        if history_unknown:  # degraded scoring: the card's pace was never read
            return None
        last_hour = int(_number(features, "txn_count_1h"))
        last_day = int(_number(features, "txn_count_24h"))
        since = _number(features, "secs_since_prev")
        if last_hour >= 2:
            return words["hour"].format(count=last_hour)
        if last_day >= 2:
            return words["day"].format(count=last_day, total=_number(features, "amt_sum_24h"))
        if not math.isnan(since) and since < 3600:
            return words["recent"].format(minutes=since / 60)
        return words["week"].format(count=int(_number(features, "txn_count_7d")))

    if group_id == "R03_TRAVEL":
        previous = _number(features, "dist_prev_km")
        since = _number(features, "secs_since_prev")
        if not math.isnan(previous) and not math.isnan(since):
            return words["previous"].format(distance=previous, minutes=since / 60)
        home = _number(features, "dist_home_km")
        return None if math.isnan(home) else words["home"].format(distance=home)

    if group_id == "R04_NOVELTY":
        if history_unknown:
            return None
        if _number(features, "is_new_merchant_30d") == 1:
            return words["new_merchant"]
        if _number(features, "is_new_category_30d") == 1:
            return words["new_category"].format(category=label.lower())
        return words["spread"].format(merchants=int(_number(features, "n_merchants_24h")))

    if group_id == "R05_TIME_AND_CATEGORY":
        hour = int(_number(features, "hour"))
        key = "night" if _number(features, "is_night") == 1 else "day"
        return words[key].format(category=label, hour=hour)

    raise KeyError(f"no wording rules for group {group_id}")


def build_explanation(
    feature_names: Sequence[str],
    contributions: FloatArray,
    bias: float,
    group_values: FloatArray,
    features: FeatureValues,
    templates: Templates,
    max_reasons: int = MAX_REASONS,
) -> Explanation:
    """Assemble the explanation of one row from its contributions."""
    group_ids = list(templates.groups)
    upward = float(group_values[group_values > 0].sum())
    worded = [
        (gid, float(value), render(gid, features, templates))
        for gid, value in zip(group_ids, group_values, strict=True)
    ]
    reasons = tuple(
        ReasonCode(gid, text, value, value / upward)
        for gid, value, text in sorted(worded, key=lambda item: -item[1])
        if value > 0 and text is not None
    )[:max_reasons]
    mitigating = None
    for gid, value, text in sorted(worded, key=lambda item: item[1]):
        if value < 0 and text is not None:
            mitigating = ReasonCode(gid, text, value, value / upward if upward else 0.0)
            break
    return Explanation(
        bias=bias,
        margin=bias + float(contributions.sum()),
        contributions={n: float(v) for n, v in zip(feature_names, contributions, strict=True)},
        groups={gid: float(v) for gid, v in zip(group_ids, group_values, strict=True)},
        reasons=reasons,
        mitigating=mitigating,
    )


def explain(
    booster: xgb.Booster,
    X: FloatArray,
    rows: Sequence[FeatureValues],
    feature_names: Sequence[str],
    feature_types: Sequence[str],
    templates: Templates | None = None,
) -> list[Explanation]:
    """Explain a batch of encoded rows.

    Args:
        booster: The model.
        X: Encoded features, as the model scores them.
        rows: The same rows' feature values, for wording; aligned with ``X``.
        feature_names: Feature names, in column order.
        feature_types: XGBoost feature types.
        templates: Groups and wording; defaults to the packaged ``templates.yaml``.
    """
    templates = templates or load_templates()
    contributions = compute_contributions(booster, X, feature_names, feature_types)
    grouped = contributions.grouped(templates.groups)
    return [
        build_explanation(
            feature_names,
            contributions.values[i],
            float(contributions.bias[i]),
            grouped[i],
            rows[i],
            templates,
        )
        for i in range(len(rows))
    ]


def top_groups(explanation: Explanation, n: int = 3) -> tuple[str, ...]:
    """Ids of the ``n`` largest positive groups, largest first (worded or not)."""
    ranked = sorted(
        ((gid, value) for gid, value in explanation.groups.items() if value > 0),
        key=lambda item: -item[1],
    )
    return tuple(gid for gid, _ in ranked[:n])


def as_rows(table: Any) -> list[dict[str, Any]]:
    """Convert a feature table (a pandas frame) to the per-row dicts ``explain`` words from."""
    records: list[dict[str, Any]] = table.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, np.generic):
                record[key] = value.item()
        record["category"] = str(record["category"])
    return records
