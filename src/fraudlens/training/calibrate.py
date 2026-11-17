"""Isotonic calibration, stored as plain numbers so serving needs no pickle.

``IsotonicRegression.predict`` is linear interpolation between its fitted breakpoints,
clipped at both ends. Storing the breakpoints and applying ``numpy.interp`` reproduces
it exactly, and the calibrator file stays human-readable and safe to load.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Calibrator:
    """A monotone map from the model's raw probability to a calibrated probability.

    Attributes:
        x: Raw-probability breakpoints, increasing.
        y: Calibrated probability at each breakpoint, non-decreasing, in [0, 1].
    """

    x: tuple[float, ...]
    y: tuple[float, ...]

    def __call__(self, raw: FloatArray) -> FloatArray:
        """Map raw probabilities to calibrated ones; values outside ``x`` are clipped."""
        calibrated: FloatArray = np.interp(raw, self.x, self.y)
        return calibrated

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {"method": "isotonic", "x": list(self.x), "y": list(self.y)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Calibrator":
        """Rebuild a calibrator from :meth:`to_json` output."""
        return cls(x=tuple(float(v) for v in data["x"]), y=tuple(float(v) for v in data["y"]))


def fit_isotonic(raw: FloatArray, y: npt.NDArray[np.int64]) -> Calibrator:
    """Fit isotonic regression of the label on the raw probability.

    Args:
        raw: Raw model probabilities on the calibration rows (the late validation slice).
        y: Labels of the same rows, 1 for fraud.
    """
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
    return Calibrator(
        x=tuple(float(v) for v in model.X_thresholds_),
        y=tuple(float(v) for v in model.y_thresholds_),
    )
