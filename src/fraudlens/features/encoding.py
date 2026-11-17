"""Turn feature values into the numeric matrix the model reads.

This is encoding, not feature logic: every value comes from ``definitions.py``. The only
transformation is ``category``, which becomes its position in ``CATEGORIES`` so that
XGBoost can treat it as a native categorical feature. Training and serving both encode
through this module, so a category always has the same code.
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraudlens.features.definitions import CATEGORIES, FEATURE_NAMES, normalise_category

FloatArray = npt.NDArray[np.float64]

FEATURE_TYPES: tuple[str, ...] = tuple("c" if name == "category" else "q" for name in FEATURE_NAMES)
"""XGBoost feature types, aligned with ``FEATURE_NAMES``: ``c`` categorical, ``q`` numeric."""

CATEGORY_INDEX = FEATURE_NAMES.index("category")


def encode_frame(features: pd.DataFrame) -> FloatArray:
    """Encode a table of features into an ``(n, len(FEATURE_NAMES))`` float matrix.

    Args:
        features: One column per name in ``FEATURE_NAMES``; extra columns are ignored.

    Returns:
        Columns in ``FEATURE_NAMES`` order. ``category`` holds its code in ``CATEGORIES``
        (a category the model does not know becomes ``unknown``). Missing values are NaN.
    """
    out = np.empty((len(features), len(FEATURE_NAMES)), dtype=np.float64)
    for column, name in enumerate(FEATURE_NAMES):
        if name == "category":
            values = features[name].astype("str").map(normalise_category)
            codes = pd.Categorical(values, categories=list(CATEGORIES)).codes
            out[:, column] = codes.astype(np.float64)
        else:
            out[:, column] = features[name].to_numpy(np.float64)
    return out


def encode_vector(features: Mapping[str, float | int | str]) -> FloatArray:
    """Encode one feature vector, as returned by ``compute_features``, into a ``(1, k)`` matrix."""
    row = np.empty((1, len(FEATURE_NAMES)), dtype=np.float64)
    for column, name in enumerate(FEATURE_NAMES):
        value = features[name]
        if name == "category":
            row[0, column] = float(CATEGORIES.index(normalise_category(str(value))))
        else:
            row[0, column] = float(value)
    return row
