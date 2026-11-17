"""An independent, brute-force reference for path-dependent TreeSHAP.

It shares no code with XGBoost's SHAP routine: it reads the trees from the model's JSON
dump, evaluates each tree's path-dependent expectation for every subset of the features
the tree uses, and applies the Shapley formula directly. That makes it exponential in the
number of features per tree, so it is only for small test models, but it is a genuine
second opinion, including on categorical splits and on our column order.

The path-dependent value of a subset S at a node splitting on feature j: if j is in S,
follow the row's branch; otherwise take the cover-weighted average of both children.
"""

import json
import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import xgboost as xgb

Node = dict[str, Any]


def load_trees(booster: xgb.Booster) -> list[Node]:
    """Every tree of the booster, as nested dicts with cover statistics."""
    return [json.loads(text) for text in booster.get_dump(dump_format="json", with_stats=True)]


def base_margin(booster: xgb.Booster) -> float:
    """The margin every row starts from: the logit of the stored base score."""
    config = json.loads(booster.save_config())
    raw = config["learner"]["learner_model_param"]["base_score"].strip("[]")
    probability = float(raw)
    return math.log(probability / (1 - probability))


def _child(node: Node, nodeid: int) -> Node:
    return next(child for child in node["children"] if child["nodeid"] == nodeid)


def _branch(node: Node, value: float) -> Node:
    """The child a row with ``value`` on the split feature goes to."""
    if math.isnan(value):
        return _child(node, node["missing"])
    condition = node["split_condition"]
    if isinstance(condition, list):  # categorical: listed categories take the "yes" branch
        goes_yes = int(value) in condition
    else:
        # XGBoost compares in 32-bit floats; so must the reference.
        goes_yes = bool(np.float32(value) < np.float32(condition))
    return _child(node, node["yes"] if goes_yes else node["no"])


def _features_used(node: Node, names: dict[str, int], out: set[int]) -> set[int]:
    if "leaf" not in node:
        out.add(names[node["split"]])
        for child in node["children"]:
            _features_used(child, names, out)
    return out


def _value(node: Node, row: np.ndarray, names: dict[str, int], known: frozenset[int]) -> float:
    if "leaf" in node:
        return float(node["leaf"])
    feature = names[node["split"]]
    if feature in known:
        return _value(_branch(node, float(row[feature])), row, names, known)
    total = node["cover"]
    return sum(
        child["cover"] / total * _value(child, row, names, known) for child in node["children"]
    )


def predict_margin(booster: xgb.Booster, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Margins computed by walking the dumped trees; must equal the booster's own."""
    names = {name: i for i, name in enumerate(feature_names)}
    every = frozenset(range(len(feature_names)))
    trees = load_trees(booster)
    base = base_margin(booster)
    return np.array([base + sum(_value(t, row, names, every) for t in trees) for row in X])


def shapley(
    booster: xgb.Booster, X: np.ndarray, feature_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Exact path-dependent Shapley values by enumerating every feature subset per tree.

    Returns:
        ``(phi, bias)``: ``phi`` is ``(n, k)`` in log-odds, ``bias`` is ``(n,)``.
    """
    names = {name: i for i, name in enumerate(feature_names)}
    trees = load_trees(booster)
    base = base_margin(booster)
    phi = np.zeros((len(X), len(feature_names)))
    bias = np.full(len(X), base)
    for tree in trees:
        used = sorted(_features_used(tree, names, set()))
        m = len(used)
        weight = [
            math.factorial(s) * math.factorial(m - s - 1) / math.factorial(m) for s in range(m)
        ]
        for r, row in enumerate(X):
            value = {
                frozenset(subset): _value(tree, row, names, frozenset(subset))
                for size in range(m + 1)
                for subset in combinations(used, size)
            }
            bias[r] += value[frozenset()]
            for feature in used:
                others = [f for f in used if f != feature]
                phi[r, feature] += sum(
                    weight[size] * (value[frozenset(s) | {feature}] - value[frozenset(s)])
                    for size in range(m)
                    for s in combinations(others, size)
                )
    return phi, bias
