"""Locations of pipeline artefacts, relative to the repository root.

Everything under ``data/`` and ``models/`` is gitignored and rebuilt by the pipeline.
"""

from pathlib import Path

DATA_DIR = Path("data")

RAW_DIR = DATA_DIR / "raw"
"""Files exactly as downloaded, plus their recorded SHA-256 hashes."""

INTERIM_DIR = DATA_DIR / "interim"
"""Validated, PII-free transactions in parquet."""

PROCESSED_DIR = DATA_DIR / "processed"
"""Model-ready feature tables in parquet."""

TRANSACTIONS_PATH = INTERIM_DIR / "transactions.parquet"
"""Every Sparkov transaction after validation and PII handling, train and test partitions."""

TRAINVAL_FEATURES_PATH = PROCESSED_DIR / "trainval.parquet"
"""Features and labels for the public training partition (training and validation rows)."""

TEST_FEATURES_PATH = PROCESSED_DIR / "test.parquet"
"""Features and labels for the public test partition. Read only by ``evaluate.final_test``."""
