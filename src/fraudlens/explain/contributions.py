"""Exact per-feature contributions from XGBoost's own TreeSHAP.

``Booster.predict(pred_contribs=True)`` returns, for tree models, SHAP values on the
margin scale (log-odds for a logistic objective): one column per feature plus a final
bias column, and each row sums to the model's margin. This is path-dependent TreeSHAP,
whose background distribution is the training cover stored in the trees, so it needs no
data at serving time. The ``shap`` library is never imported here or in the service.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import xgboost as xgb

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class Contributions:
    """Per-feature contributions for a batch of rows, in log-odds.

    Attributes:
        feature_names: Column names of ``values``.
        values: ``(n, k)`` contribution of each feature to each row's margin.
        bias: ``(n,)`` the model's expected margin, the starting point of every row.
    """

    feature_names: tuple[str, ...]
    values: FloatArray
    bias: FloatArray

    @property
    def margin(self) -> FloatArray:
        """Bias plus every contribution: the model's margin for each row."""
        margin: FloatArray = self.values.sum(axis=1) + self.bias
        return margin

    def grouped(self, groups: Mapping[str, Sequence[str]]) -> FloatArray:
        """Sum contributions within each group.

        Args:
            groups: Group id to the features in it. Every feature must be in exactly one
                group.

        Returns:
            ``(n, len(groups))`` contributions, columns in the order of ``groups``.

        Raises:
            ValueError: if a feature is in no group, in two groups, or unknown.
        """
        position = {name: i for i, name in enumerate(self.feature_names)}
        members = [name for features in groups.values() for name in features]
        unknown = sorted(set(members) - set(position))
        missing = sorted(set(position) - set(members))
        repeated = sorted({name for name in members if members.count(name) > 1})
        if unknown or missing or repeated:
            raise ValueError(
                f"feature groups must cover each feature once: unknown {unknown}, "
                f"ungrouped {missing}, repeated {repeated}"
            )
        summed: FloatArray = np.stack(
            [
                self.values[:, [position[n] for n in features]].sum(axis=1)
                for features in groups.values()
            ],
            axis=1,
        )
        return summed


def compute_contributions(
    booster: xgb.Booster,
    X: FloatArray,
    feature_names: Sequence[str],
    feature_types: Sequence[str],
) -> Contributions:
    """Compute exact contributions for encoded rows.

    Args:
        booster: The model.
        X: Encoded features, columns in ``feature_names`` order.
        feature_names: Feature names, as the model was trained with.
        feature_types: XGBoost feature types (``c`` categorical, ``q`` numeric).
    """
    matrix = xgb.DMatrix(
        X,
        feature_names=list(feature_names),
        feature_types=list(feature_types),
        enable_categorical=True,
    )
    raw = np.asarray(booster.predict(matrix, pred_contribs=True), dtype=np.float64)
    return Contributions(feature_names=tuple(feature_names), values=raw[:, :-1], bias=raw[:, -1])
