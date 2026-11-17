"""Choose the review and block thresholds by expected cost, within review capacity.

Decisions on a calibrated probability ``p``:

* approve when ``p < t_review``
* review when ``t_review <= p < t_block``
* block when ``p >= t_block``

The search tries every pair of candidate thresholds. Costs are computed from cumulative
totals above each candidate, so every pair costs O(1) and the full search is exact.
The same totals are exported as the threshold curve the console's threshold lab reads.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from fraudlens.training.config import CostConfig

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

NEVER = 1.01
"""A threshold above any probability: as a block threshold it means never block."""

MAX_CANDIDATES = 400


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Decision thresholds on the calibrated probability, ``t_review <= t_block``."""

    t_review: float
    t_block: float


def candidates(probability: FloatArray) -> FloatArray:
    """Candidate thresholds: distinct probabilities (at most 400 quantiles), plus ``NEVER``."""
    distinct = np.unique(probability)
    if len(distinct) > MAX_CANDIDATES:
        distinct = np.unique(np.quantile(distinct, np.linspace(0.0, 1.0, MAX_CANDIDATES)))
    values: FloatArray = np.append(distinct, NEVER)
    return values


def cumulative_totals(
    probability: FloatArray, y: IntArray, amount: FloatArray, thresholds: FloatArray
) -> dict[str, FloatArray]:
    """Totals over the rows with ``probability >= t``, for each threshold ``t``.

    Returns:
        ``count``, ``frauds``, ``fraud_amount``, ``genuine`` and ``genuine_amount``,
        each aligned with ``thresholds``.
    """
    order = np.argsort(probability)
    sorted_p = probability[order]
    fraud = (y[order] == 1).astype(np.float64)
    value = amount[order]

    def above(weights: FloatArray) -> FloatArray:
        # suffix[i] is the total over sorted rows i..n-1.
        suffix = np.concatenate([np.cumsum(weights[::-1])[::-1], [0.0]])
        start = np.searchsorted(sorted_p, thresholds, side="left")
        totals: FloatArray = suffix[start]
        return totals

    return {
        "count": above(np.ones_like(fraud)),
        "frauds": above(fraud),
        "fraud_amount": above(fraud * value),
        "genuine": above(1.0 - fraud),
        "genuine_amount": above((1.0 - fraud) * value),
    }


def pair_costs(
    totals: dict[str, FloatArray], n_rows: int, total_fraud_amount: float, config: CostConfig
) -> tuple[FloatArray, FloatArray]:
    """Expected cost per 1,000 transactions, and review share, for every threshold pair.

    Returns:
        Two ``(k, k)`` arrays indexed ``[review index, block index]``. Pairs with
        ``t_block < t_review`` are ``inf`` in the cost array.
    """
    costs = config.costs
    missed = total_fraud_amount - totals["fraud_amount"]  # frauds below t_review
    reviews = totals["count"][:, None] - totals["count"][None, :]
    blocked = costs.margin_rate * totals["genuine_amount"] + costs.friction_cost * totals["genuine"]
    total = missed[:, None] + costs.review_cost * reviews + blocked[None, :]
    k = len(totals["count"])
    total = np.where(np.triu(np.ones((k, k), dtype=bool)), total, np.inf)
    cost: FloatArray = total / n_rows * 1000.0
    share: FloatArray = reviews / n_rows
    return cost, share


def choose_thresholds(
    probability: FloatArray, y: IntArray, amount: FloatArray, config: CostConfig
) -> tuple[Thresholds, float]:
    """Return the cheapest thresholds whose review share fits the analysts' capacity.

    Args:
        probability: Calibrated probabilities on the threshold-selection rows.
        y: Labels, 1 for fraud.
        amount: Amounts in dollars.
        config: Costs and review capacity.

    Returns:
        The thresholds and their expected cost per 1,000 transactions on these rows.
    """
    grid = candidates(probability)
    totals = cumulative_totals(probability, y, amount, grid)
    cost, share = pair_costs(totals, len(y), float(amount[y == 1].sum()), config)
    cost = np.where(share <= config.review_capacity_share + 1e-12, cost, np.inf)
    review, block = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return Thresholds(float(grid[review]), float(grid[block])), float(cost[review, block])


def threshold_curve(
    probability: FloatArray,
    y: IntArray,
    amount: FloatArray,
    block: IntArray,
    config: CostConfig,
) -> dict[str, Any]:
    """Export what any threshold would do on these rows, for the threshold lab.

    For each candidate ``t`` the curve gives the totals over rows with ``p >= t``. From
    them the console can price any ``(t_review, t_block)`` pair without the raw rows:
    ``cost = missed(t_review) + review_cost * (count(t_review) - count(t_block))
    + margin_rate * genuine_amount(t_block) + friction_cost * genuine(t_block)``.
    Convenience columns give precision, recall, dollar recall and alerts per day when
    ``t`` is used as the alert threshold.
    """
    grid = candidates(probability)
    totals = cumulative_totals(probability, y, amount, grid)
    n_frauds = float((y == 1).sum())
    fraud_amount = float(amount[y == 1].sum())
    n_days = len(np.unique(block))
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(totals["count"] > 0, totals["frauds"] / totals["count"], np.nan)
    return {
        "rows": len(y),
        "days": n_days,
        "frauds": n_frauds,
        "fraud_amount": fraud_amount,
        "costs": {
            "review_cost": config.costs.review_cost,
            "margin_rate": config.costs.margin_rate,
            "friction_cost": config.costs.friction_cost,
            "review_capacity_share": config.review_capacity_share,
        },
        "points": [
            {
                "threshold": float(grid[i]),
                "count": float(totals["count"][i]),
                "frauds": float(totals["frauds"][i]),
                "fraud_amount": float(totals["fraud_amount"][i]),
                "genuine": float(totals["genuine"][i]),
                "genuine_amount": float(totals["genuine_amount"][i]),
                "precision": None if np.isnan(precision[i]) else float(precision[i]),
                "recall": float(totals["frauds"][i] / n_frauds) if n_frauds else None,
                "dollar_recall": (
                    float(totals["fraud_amount"][i] / fraud_amount) if fraud_amount else None
                ),
                "alerts_per_day": float(totals["count"][i] / n_days),
            }
            for i in range(len(grid))
        ],
    }


def decide(probability: FloatArray, thresholds: Thresholds) -> IntArray:
    """Return ``APPROVE`` (0), ``REVIEW`` (1) or ``BLOCK`` (2) for each probability."""
    decision: IntArray = (probability >= thresholds.t_review).astype(np.int64) + (
        probability >= thresholds.t_block
    ).astype(np.int64)
    return decision
