"""Versioned model bundles: write them, verify them, load them.

A release is a directory ``models/<version>/`` whose ``manifest.json`` lists the SHA-256
of every other file in it. :func:`verify_bundle` fails if a file is missing, extra or
changed, and :meth:`ModelBundle.load` verifies before it returns, so a service built on
it can never serve an unverified model.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from fraudlens.training.calibrate import Calibrator
from fraudlens.training.thresholds import Thresholds, decide

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

MANIFEST = "manifest.json"
BOOSTER = "booster.json"
_VERSION = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(\d+)$")


class BundleIntegrityError(RuntimeError):
    """A bundle's files do not match its manifest."""


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as indented, key-sorted JSON with a trailing newline."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(directory: Path) -> dict[str, str]:
    """Hash every file in ``directory`` except the manifest, and write the manifest."""
    files = {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != MANIFEST
    }
    write_json(directory / MANIFEST, {"files": files})
    return files


def verify_bundle(directory: Path) -> None:
    """Check every file in ``directory`` against its manifest.

    Raises:
        BundleIntegrityError: if the manifest is missing, a listed file is missing or
            changed, or a file is present that the manifest does not list.
    """
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise BundleIntegrityError(f"{directory} has no {MANIFEST}")
    expected: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    present = {p.name for p in directory.iterdir() if p.is_file() and p.name != MANIFEST}
    problems = [f"missing {name}" for name in sorted(set(expected) - present)]
    problems += [f"unlisted {name}" for name in sorted(present - set(expected))]
    problems += [
        f"changed {name}"
        for name in sorted(set(expected) & present)
        if sha256_file(directory / name) != expected[name]
    ]
    if problems:
        raise BundleIntegrityError(f"{directory}: " + ", ".join(problems))


def next_version(models_dir: Path, today: date) -> str:
    """Return ``YYYY-MM-DD.N``, one higher than any release already made on ``today``."""
    prefix = today.isoformat()
    taken = [
        int(match.group(2))
        for path in (models_dir.iterdir() if models_dir.exists() else [])
        if (match := _VERSION.match(path.name)) and match.group(1) == prefix
    ]
    return f"{prefix}.{max(taken, default=0) + 1}"


def latest_bundle(models_dir: Path) -> Path:
    """Return the newest release directory under ``models_dir``, by date then sequence.

    Raises:
        FileNotFoundError: if ``models_dir`` holds no release.
    """
    releases = [
        (match.group(1), int(match.group(2)), path)
        for path in (models_dir.iterdir() if models_dir.exists() else [])
        if path.is_dir() and (match := _VERSION.match(path.name))
    ]
    if not releases:
        raise FileNotFoundError(f"no model bundle under {models_dir}")
    return max(releases)[2]


def manifest_sha256(directory: Path) -> str:
    """Return the hex SHA-256 of a bundle's manifest, which pins every file in it."""
    return sha256_file(directory / MANIFEST)


def write_bundle(
    directory: Path,
    *,
    booster: xgb.Booster,
    calibrator: Calibrator,
    thresholds: Thresholds,
    files: Mapping[str, Any],
) -> Path:
    """Write a bundle and its manifest.

    Args:
        directory: New, empty bundle directory.
        booster: The model, cut to its final tree.
        calibrator: Raw probability to calibrated probability.
        thresholds: Decision thresholds on the calibrated probability.
        files: Other JSON documents to include, keyed by file name (``features.json``,
            ``metrics.json`` and so on).

    Raises:
        FileExistsError: if ``directory`` already exists, so a release is never overwritten.
    """
    directory.mkdir(parents=True, exist_ok=False)
    booster.save_model(directory / BOOSTER)
    write_json(directory / "calibrator.json", calibrator.to_json())
    write_json(
        directory / "thresholds.json",
        {"t_review": thresholds.t_review, "t_block": thresholds.t_block},
    )
    for name, content in files.items():
        write_json(directory / name, content)
    write_manifest(directory)
    return directory


@dataclass(eq=False)
class ModelBundle:
    """A verified release, ready to score encoded feature rows.

    Attributes:
        path: Bundle directory.
        version: Directory name, ``YYYY-MM-DD.N``.
        booster: The model, running on CPU.
        calibrator: Raw probability to calibrated probability.
        thresholds: Decision thresholds on the calibrated probability.
        documents: Every other JSON file in the bundle, keyed by file name.
    """

    path: Path
    version: str
    booster: xgb.Booster
    calibrator: Calibrator
    thresholds: Thresholds
    documents: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "ModelBundle":
        """Verify the bundle at ``path`` against its manifest, then load it.

        Raises:
            BundleIntegrityError: if any file fails verification.
        """
        verify_bundle(path)
        booster = xgb.Booster()
        booster.load_model(path / BOOSTER)
        booster.set_param({"device": "cpu"})
        documents = {
            p.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("*.json"))
            if p.name not in {BOOSTER, MANIFEST}
        }
        thresholds = documents["thresholds.json"]
        return cls(
            path=path,
            version=path.name,
            booster=booster,
            calibrator=Calibrator.from_json(documents["calibrator.json"]),
            thresholds=Thresholds(float(thresholds["t_review"]), float(thresholds["t_block"])),
            documents=documents,
        )

    def margin(self, X: FloatArray) -> FloatArray:
        """Return the model margin (log-odds) for encoded rows."""
        return booster_margin(self.booster, X)

    def raw_probability(self, X: FloatArray) -> FloatArray:
        """Return the uncalibrated fraud probability, the logistic of the margin."""
        return logistic(self.margin(X))

    def score(self, X: FloatArray) -> FloatArray:
        """Return the calibrated fraud probability, the service's ``risk_score``."""
        return self.calibrator(self.raw_probability(X))

    def decide(self, score: FloatArray) -> IntArray:
        """Return ``APPROVE`` (0), ``REVIEW`` (1) or ``BLOCK`` (2) for calibrated scores."""
        return decide(score, self.thresholds)


def booster_margin(booster: xgb.Booster, X: FloatArray) -> FloatArray:
    """Return a booster's margin (log-odds) for encoded rows, via ``inplace_predict``."""
    margin: FloatArray = np.asarray(
        booster.inplace_predict(X, predict_type="margin"), dtype=np.float64
    )
    return margin


def logistic(margin: FloatArray) -> FloatArray:
    """Map a margin in log-odds to a probability."""
    probability: FloatArray = 1.0 / (1.0 + np.exp(-margin))
    return probability
