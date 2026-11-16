"""Raw CSVs to feature tables, end to end, on small files."""

from pathlib import Path

import pandas as pd
import pytest

from fraudlens.data import download, ingest
from fraudlens.data.ingest import ingest as run_ingest
from fraudlens.data.pii import CLEAN_COLUMNS, PII_COLUMNS
from fraudlens.data.schema import RowCountError
from fraudlens.data.splits import TimeOrderError
from fraudlens.features import batch
from fraudlens.features.batch import KEY_COLUMNS, write_feature_tables
from fraudlens.features.definitions import FEATURE_NAMES
from helpers import make_raw_sparkov

KEY = b"unit-test-key"
ROWS = {"train": 300, "test": 120}


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Two small raw files, the test partition starting after the train partition ends."""
    target = tmp_path / "raw" / "sparkov"
    target.mkdir(parents=True)
    make_raw_sparkov(300, seed=1, start="2019-01-01", span_days=100).to_csv(
        target / "fraudTrain.csv", index=False
    )
    make_raw_sparkov(120, seed=2, start="2019-06-01", span_days=40).to_csv(
        target / "fraudTest.csv", index=False
    )
    return tmp_path / "raw"


def test_ingest_writes_a_pii_free_table(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "interim" / "transactions.parquet"

    summary = run_ingest(raw_dir, out, hmac_key=KEY, expected_rows=ROWS)

    table = pd.read_parquet(out)
    assert tuple(table.columns) == CLEAN_COLUMNS
    assert set(PII_COLUMNS).isdisjoint(table.columns)
    assert summary["train"]["rows"] == 300
    assert summary["test"]["rows"] == 120
    assert summary["train"]["frauds"] == int(table.query("partition == 'train'")["is_fraud"].sum())


def test_ingest_refuses_a_file_with_the_wrong_row_count(raw_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(RowCountError):
        run_ingest(
            raw_dir, tmp_path / "t.parquet", hmac_key=KEY, expected_rows={"train": 299, "test": 120}
        )


def test_ingest_refuses_a_test_partition_that_overlaps_training(
    raw_dir: Path, tmp_path: Path
) -> None:
    make_raw_sparkov(120, seed=2, start="2019-02-01", span_days=40).to_csv(
        raw_dir / "sparkov" / "fraudTest.csv", index=False
    )

    with pytest.raises(TimeOrderError):
        run_ingest(raw_dir, tmp_path / "t.parquet", hmac_key=KEY, expected_rows=ROWS)


def test_feature_tables_are_written_per_partition(raw_dir: Path, tmp_path: Path) -> None:
    transactions = tmp_path / "transactions.parquet"
    run_ingest(raw_dir, transactions, hmac_key=KEY, expected_rows=ROWS)

    counts = write_feature_tables(
        transactions,
        tmp_path / "processed" / "trainval.parquet",
        tmp_path / "processed" / "test.parquet",
    )

    assert counts == ROWS
    trainval = pd.read_parquet(tmp_path / "processed" / "trainval.parquet")
    test = pd.read_parquet(tmp_path / "processed" / "test.parquet")
    assert tuple(trainval.columns) == KEY_COLUMNS + FEATURE_NAMES
    assert len(trainval) == 300
    assert len(test) == 120
    assert set(trainval["transaction_id"]).isdisjoint(test["transaction_id"])
    assert "gender" not in trainval.columns
    assert "partition" not in trainval.columns
    # The first test rows see history from the end of training.
    assert test["n_prior_30d"].max() > 0


def test_entry_points_call_their_pipeline_step(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "FRAUDLENS_API_KEY": "k",
        "FRAUDLENS_CARD_HMAC_KEY": "unit-test-key",
        "FRAUDLENS_DATABASE_URL": "postgresql+asyncpg://u:p@localhost/db",
        "FRAUDLENS_REDIS_URL": "redis://localhost:6379/0",
    }.items():
        monkeypatch.setenv(name, value)
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(download, "download_all", lambda **kw: calls.append(("download", kw)))
    monkeypatch.setattr(ingest, "ingest", lambda **kw: calls.append(("ingest", kw)))
    monkeypatch.setattr(batch, "write_feature_tables", lambda **kw: calls.append(("features", kw)))
    monkeypatch.setattr(ingest, "configure_logging", lambda *_: None)
    monkeypatch.setattr(download, "configure_logging", lambda *_: None)
    monkeypatch.setattr(batch, "configure_logging", lambda *_: None)
    ingest.get_settings.cache_clear()

    try:
        download.main()
        ingest.main()
        batch.main()
    finally:
        ingest.get_settings.cache_clear()

    # The card HMAC key reaches ingestion from settings, as bytes.
    assert calls == [("download", {}), ("ingest", {"hmac_key": KEY}), ("features", {})]
