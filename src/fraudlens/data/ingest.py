"""Raw Sparkov CSVs to one validated, PII-free transaction table.

Each public file is read, checked against the raw schema and its known row count, and
passed through the PII policy. The result is checked against the clean schema, which
rejects any extra column, and written as parquet.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from fraudlens.common.logging import configure_logging
from fraudlens.common.paths import RAW_DIR, TRANSACTIONS_PATH
from fraudlens.common.settings import get_settings
from fraudlens.data.pii import clean_sparkov
from fraudlens.data.schema import SPARKOV_ROWS, validate_clean, validate_raw_sparkov
from fraudlens.data.splits import assert_time_ordered

logger = logging.getLogger(__name__)

SPARKOV_FILES: Mapping[str, str] = {"train": "fraudTrain.csv", "test": "fraudTest.csv"}


def ingest(
    raw_dir: Path = RAW_DIR,
    out_path: Path = TRANSACTIONS_PATH,
    *,
    hmac_key: bytes,
    expected_rows: Mapping[str, int] = SPARKOV_ROWS,
) -> dict[str, dict[str, int]]:
    """Validate both Sparkov files, strip PII and write the transaction table.

    Args:
        raw_dir: Directory holding ``sparkov/fraudTrain.csv`` and ``sparkov/fraudTest.csv``.
        out_path: Parquet file to write.
        hmac_key: Secret key for card tokens.
        expected_rows: Row count each partition must have.

    Returns:
        ``{partition: {"rows": ..., "frauds": ...}}``.

    Raises:
        RowCountError: if a file has the wrong number of rows.
        pandera.errors.SchemaErrors: if a file or the cleaned table breaks its schema.
        TimeOrderError: if the test partition does not start after the train partition ends.
    """
    frames: list[pd.DataFrame] = []
    for partition, file in SPARKOV_FILES.items():
        raw = pd.read_csv(raw_dir / "sparkov" / file)
        validate_raw_sparkov(raw, expected_rows=expected_rows[partition])
        frames.append(clean_sparkov(raw, partition=partition, hmac_key=hmac_key))
        logger.info("partition validated", extra={"partition": partition, "rows": len(raw)})

    table = validate_clean(pd.concat(frames, ignore_index=True))
    is_train = table["partition"] == "train"
    assert_time_ordered(
        table.loc[is_train, "event_ts"].to_numpy(), table.loc[~is_train, "event_ts"].to_numpy()
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    summary = {
        str(partition): {"rows": len(rows), "frauds": int(rows["is_fraud"].sum())}
        for partition, rows in table.groupby("partition", sort=False)
    }
    logger.info("transactions written", extra={"path": str(out_path), "summary": summary})
    return summary


def main() -> None:
    """Entry point for ``python -m fraudlens.data.ingest``."""
    settings = get_settings()
    configure_logging(settings.log_level)
    ingest(hmac_key=settings.card_hmac_key.get_secret_value().encode("utf-8"))


if __name__ == "__main__":  # pragma: no cover
    main()
