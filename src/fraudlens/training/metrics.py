"""Evaluation metrics for a rare-event classifier with a fixed review capacity.

Accuracy is deliberately absent: at roughly 0.5% fraud, a model that flags nothing is
more than 99% accurate. What matters is how much fraud, by count and by dollar value,
sits in the small share of transactions analysts can actually review, whether the
predicted probabilities can be trusted in a cost calculation, and what the decisions cost.

Confidence intervals resample whole days, because transactions within a day are
correlated and resampling single rows would make the intervals too narrow.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

ALERT_BUDGETS: tuple[float, ...] = (0.005, 0.01)
"""Shares of daily volume an analyst team could review: 0.5% and 1%."""

APPROVE, REVIEW, BLOCK = 0, 1, 2
"""Decision codes, in order of increasing risk."""

RELIABILITY_EDGES: tuple[float, ...] = (0.0, 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
"""Probability bin edges, finer near zero where almost every prediction lies."""


@dataclass(frozen=True, slots=True)
class Costs:
    """Business costs, all in dollars. The values are assumptions; see ``configs/costs.yaml``.

    Attributes:
        review_cost: Analyst cost of reviewing one alert.
        margin_rate: Share of the amount lost as margin when a genuine purchase is blocked.
        friction_cost: Fixed penalty for wrongly blocking a genuine purchase.
    """

    review_cost: float
    margin_rate: float
    friction_cost: float


def average_precision(y: npt.ArrayLike, score: npt.ArrayLike) -> float:
    """Return average precision, the area under the precision-recall curve.

    Args:
        y: Labels, 1 for fraud and 0 for genuine.
        score: Model scores; higher means more suspicious. Only their order matters.
    """
    return float(average_precision_score(y, score))


def daily_top_k(score: FloatArray, day: IntArray, budget: float) -> BoolArray:
    """Flag the highest-scoring share of each day's transactions.

    Args:
        score: Model scores, one per transaction.
        day: Day number of each transaction (any integer that is constant within a day).
        budget: Share of each day's volume to flag, for example 0.01 for 1%.

    Returns:
        A mask with ``ceil(budget * n)`` rows flagged on a day with ``n`` transactions.
        Ties in score are broken by row position.
    """
    n = len(score)
    order = np.lexsort((np.arange(n), -score, day))
    day_sorted = day[order]
    starts = np.flatnonzero(np.r_[True, day_sorted[1:] != day_sorted[:-1]])
    sizes = np.diff(np.r_[starts, n])
    rank = np.arange(n) - np.repeat(starts, sizes)
    quota = np.repeat(np.ceil(budget * sizes), sizes)
    flagged = np.zeros(n, dtype=np.bool_)
    flagged[order] = rank < quota
    return flagged


def alert_metrics(y: IntArray, amount: FloatArray, flagged: BoolArray) -> dict[str, float]:
    """Return recall, precision and dollar recall of a set of alerts.

    Args:
        y: Labels, 1 for fraud.
        amount: Transaction amounts in dollars.
        flagged: Which transactions were alerted.

    Returns:
        ``recall`` (share of frauds alerted), ``precision`` (share of alerts that are
        fraud) and ``dollar_recall`` (share of fraud value alerted). A ratio with a zero
        denominator is NaN.
    """
    fraud = y == 1
    caught = fraud & flagged
    fraud_value = float(amount[fraud].sum())
    return {
        "recall": _ratio(float(caught.sum()), float(fraud.sum())),
        "precision": _ratio(float(caught.sum()), float(flagged.sum())),
        "dollar_recall": _ratio(float(amount[caught].sum()), fraud_value),
    }


def brier_score(y: IntArray, probability: FloatArray) -> float:
    """Return the mean squared difference between predicted probability and outcome."""
    return float(np.mean((probability - y) ** 2))


def reliability_curve(
    y: IntArray, probability: FloatArray, edges: Sequence[float] = RELIABILITY_EDGES
) -> list[dict[str, float | int]]:
    """Compare predicted probability with observed fraud rate, bin by bin.

    Args:
        y: Labels, 1 for fraud.
        probability: Predicted fraud probabilities in [0, 1].
        edges: Increasing bin edges. Bins are ``[lower, upper)``; the last includes 1.

    Returns:
        One entry per non-empty bin with its bounds, row count, mean predicted
        probability and observed fraud rate. A calibrated model has the last two equal.
    """
    bins = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, len(edges) - 2)
    curve: list[dict[str, float | int]] = []
    for index in range(len(edges) - 1):
        members = bins == index
        count = int(members.sum())
        if count:
            curve.append(
                {
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": count,
                    "mean_predicted": float(probability[members].mean()),
                    "observed": float(y[members].mean()),
                }
            )
    return curve


def expected_cost_per_1000(
    y: IntArray, amount: FloatArray, decision: IntArray, costs: Costs
) -> float:
    """Return the cost of a set of decisions, in dollars per 1,000 transactions.

    * An approved fraud costs its amount.
    * A reviewed transaction costs the review, whatever it turns out to be; the analyst
      is assumed to stop the fraud and release the genuine purchase.
    * A blocked genuine purchase costs its lost margin plus the friction penalty.
    * A blocked fraud costs nothing.

    Args:
        y: Labels, 1 for fraud.
        amount: Transaction amounts in dollars.
        decision: ``APPROVE``, ``REVIEW`` or ``BLOCK`` per transaction.
        costs: The business costs.
    """
    fraud = y == 1
    missed = amount[(decision == APPROVE) & fraud].sum()
    reviews = costs.review_cost * (decision == REVIEW).sum()
    wrongly_blocked = (decision == BLOCK) & ~fraud
    blocked = costs.margin_rate * amount[wrongly_blocked].sum() + (
        costs.friction_cost * wrongly_blocked.sum()
    )
    return float((missed + reviews + blocked) / len(y) * 1000.0)


def day_blocks(day: IntArray) -> list[IntArray]:
    """Return the row positions of each day, days in increasing order."""
    order = np.argsort(day, kind="stable")
    boundaries = np.flatnonzero(np.diff(day[order])) + 1
    return [block.astype(np.int64) for block in np.split(order, boundaries)]


def bootstrap_days(blocks: Sequence[IntArray], n_boot: int, seed: int) -> Iterator[IntArray]:
    """Yield ``n_boot`` resamples of the rows, each drawing whole days with replacement.

    Args:
        blocks: Row positions per day, from :func:`day_blocks`.
        n_boot: Number of resamples.
        seed: Seed of the random generator; the same seed gives the same resamples.

    Yields:
        Row positions of one resample: as many days as the original, drawn with replacement.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        chosen = rng.integers(0, len(blocks), len(blocks))
        yield np.concatenate([blocks[index] for index in chosen])


def percentile_interval(values: npt.ArrayLike, level: float = 0.95) -> tuple[float, float]:
    """Return the central ``level`` percentile interval of bootstrap replicates, ignoring NaN."""
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.nanpercentile(np.asarray(values, dtype=np.float64), [tail, 100.0 - tail])
    return float(low), float(high)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")
