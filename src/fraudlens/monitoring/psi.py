"""The population stability index (PSI), and binning live values like the reference.

``PSI = sum((a - e) * ln(a / e))`` over bins, where ``e`` and ``a`` are the expected and
actual shares of rows in each bin. A common rule of thumb reads values under 0.1 as stable
and over 0.25 as worth investigating.

Where it misleads:

* It depends on the bins. Decile edges from the reference make each expected bin hold
  about 10%; a shift inside one bin is invisible.
* An empty bin makes the logarithm infinite, so shares are floored at 1e-4. The floor
  caps how loud a bin that empties completely can be.
* Small samples drift by chance. With ``n`` rows in each window, noise alone gives about
  ``(bins - 1) * 2 / n``; segments below a minimum size are not judged.
* A window shorter than a feature's natural cycle always looks drifted: 10,000
  transactions cover about a day, so day-of-week "drifts" with no change in behaviour.
"""

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

STABLE = 0.1
"""Below this, a distribution is read as stable."""

ALARM = 0.25
"""Above this, a distribution is read as shifted and worth investigating."""

FLOOR = 1e-4
"""Smallest share used in the logarithm, so an empty bin does not make PSI infinite."""

Status = Literal["stable", "watch", "alarm", "unknown"]


def psi(expected: FloatArray, actual: FloatArray) -> float:
    """PSI between two vectors of bin shares (each summing to at most 1)."""
    e = np.clip(np.asarray(expected, dtype=np.float64), FLOOR, None)
    a = np.clip(np.asarray(actual, dtype=np.float64), FLOOR, None)
    return float(((a - e) * np.log(a / e)).sum())


def status(value: float | None) -> Status:
    """``stable`` under 0.1, ``alarm`` over 0.25, ``watch`` between, ``unknown`` if absent."""
    if value is None:
        return "unknown"
    if value < STABLE:
        return "stable"
    return "alarm" if value > ALARM else "watch"


def expected_shares(spec: Mapping[str, Any]) -> FloatArray:
    """The reference's bin shares followed by its missing share, from ``reference.json``."""
    return np.append(np.asarray(spec["proportions"], dtype=np.float64), float(spec["missing"]))


def shares_for_reference(values: npt.NDArray[Any], spec: Mapping[str, Any]) -> FloatArray:
    """Live shares in the reference's bins, followed by the missing share.

    Numeric values are binned with the reference's inner edges using
    ``searchsorted(side="right")``, exactly as the reference was built at training time.
    Shares are of all rows, missing included, as in the reference.

    Args:
        values: One feature's live values; NaN (numeric) or ``None`` (categorical) when
            missing.
        spec: That feature's entry in the bundle's ``reference.json``.
    """
    n = max(1, len(values))
    if spec["kind"] == "categorical":
        categories = list(spec["categories"])
        counts = np.zeros(len(categories), dtype=np.float64)
        missing = 0
        for value in values:
            if value is None:
                missing += 1
            elif value in categories:
                counts[categories.index(value)] += 1
        return np.append(counts / n, missing / n)
    numeric = np.asarray(values, dtype=np.float64)
    absent = np.isnan(numeric)
    inner = np.asarray(spec["inner_edges"], dtype=np.float64)
    binned = np.bincount(
        np.searchsorted(inner, numeric[~absent], side="right"), minlength=len(inner) + 1
    )
    return np.append(binned / n, absent.sum() / n)


def psi_between(base: FloatArray, live: FloatArray, bins: int = 10) -> float:
    """PSI of ``live`` against ``base``, on decile edges taken from ``base``."""
    inner = np.unique(np.quantile(base, np.linspace(0.0, 1.0, bins + 1)[1:-1]))

    def shares(values: FloatArray) -> FloatArray:
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        result: FloatArray = counts / max(1, len(values))
        return result

    return psi(shares(base), shares(live))
