"""Fetch the raw datasets from Kaggle into ``data/raw/`` and pin them by SHA-256.

The first run downloads each dataset and records one hash per file in
``data/raw/SHA256SUMS.json``. Later runs download nothing; they re-hash the files and
fail if any differs from the record, so every downstream artefact can name the exact
raw bytes it was built from.
"""

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fraudlens.common.logging import configure_logging
from fraudlens.common.paths import RAW_DIR

logger = logging.getLogger(__name__)

HASH_FILE = "SHA256SUMS.json"
_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """A Kaggle dataset and the files the pipeline needs from it."""

    handle: str
    files: tuple[str, ...]


DATASETS: Mapping[str, DatasetSpec] = {
    "sparkov": DatasetSpec(
        handle="kartik2112/fraud-detection",
        files=("fraudTrain.csv", "fraudTest.csv"),
    ),
    "ulb": DatasetSpec(
        handle="mlg-ulb/creditcardfraud",
        files=("creditcard.csv",),
    ),
}


class Fetch(Protocol):
    """Downloads every file of the dataset ``handle`` into ``output_dir``."""

    def __call__(self, handle: str, *, output_dir: str) -> str:
        """Download the dataset and return the directory it was written to."""
        ...


class RawDataChangedError(RuntimeError):
    """A raw file no longer matches the hash recorded when it was downloaded."""


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of the file at ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _kaggle_fetch(handle: str, *, output_dir: str) -> str:
    """Download with the official Kaggle client. Public datasets need no credentials."""
    import kagglehub

    return str(kagglehub.dataset_download(handle, output_dir=output_dir))


def download_all(
    raw_dir: Path = RAW_DIR,
    *,
    specs: Mapping[str, DatasetSpec] = DATASETS,
    fetch: Fetch = _kaggle_fetch,
) -> dict[str, str]:
    """Ensure every dataset in ``specs`` is present under ``raw_dir`` and hash-verified.

    Args:
        raw_dir: Directory holding one sub-directory per dataset.
        specs: Datasets to fetch, keyed by the sub-directory name.
        fetch: Downloader; replaced by a fake in tests.

    Returns:
        ``{"<dataset>/<file>": sha256_hex}`` for every expected file.

    Raises:
        FileNotFoundError: if a dataset was fetched but an expected file is absent.
        RawDataChangedError: if a file differs from its previously recorded hash.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    hash_path = raw_dir / HASH_FILE
    recorded: dict[str, str] = (
        json.loads(hash_path.read_text(encoding="utf-8")) if hash_path.exists() else {}
    )

    hashes: dict[str, str] = {}
    for name, spec in specs.items():
        target = raw_dir / name
        if not all((target / file).exists() for file in spec.files):
            target.mkdir(parents=True, exist_ok=True)
            logger.info("downloading dataset", extra={"dataset": name, "handle": spec.handle})
            returned = Path(fetch(spec.handle, output_dir=str(target)))
            # Some environments (Colab's Kaggle cache) ignore output_dir and return their
            # own directory instead; copy what is missing from there.
            for file in spec.files:
                if not (target / file).exists() and (returned / file).exists():
                    shutil.copyfile(returned / file, target / file)
        for file in spec.files:
            path = target / file
            if not path.exists():
                raise FileNotFoundError(f"{spec.handle} did not provide {file} in {target}")
            key = f"{name}/{file}"
            hashes[key] = sha256_file(path)
            if key in recorded and recorded[key] != hashes[key]:
                raise RawDataChangedError(
                    f"{key} has SHA-256 {hashes[key]} but {recorded[key]} was recorded in "
                    f"{hash_path}. Delete both to download again."
                )

    hash_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("raw data verified", extra={"files": len(hashes)})
    return hashes


def main() -> None:
    """Entry point for ``python -m fraudlens.data.download``."""
    configure_logging()
    download_all()


if __name__ == "__main__":  # pragma: no cover
    main()
