"""Raw files are fetched once, hashed, and re-verified against the recorded hashes."""

import hashlib
import json
from pathlib import Path

import pytest

from fraudlens.data.download import (
    DATASETS,
    HASH_FILE,
    DatasetSpec,
    RawDataChangedError,
    download_all,
    sha256_file,
)

SPECS = {"toy": DatasetSpec(handle="someone/toy", files=("a.csv", "b.csv"))}


class FakeFetch:
    """Stands in for the Kaggle client: writes fixed bytes into ``output_dir``."""

    def __init__(self, contents: dict[str, bytes]) -> None:
        self.contents = contents
        self.calls: list[str] = []

    def __call__(self, handle: str, *, output_dir: str) -> str:
        self.calls.append(handle)
        for name, data in self.contents.items():
            Path(output_dir, name).write_bytes(data)
        return output_dir


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"fraudlens" * 300_000)

    assert sha256_file(path) == hashlib.sha256(b"fraudlens" * 300_000).hexdigest()


def test_download_records_a_hash_per_file(tmp_path: Path) -> None:
    fetch = FakeFetch({"a.csv": b"1,2\n", "b.csv": b"3,4\n"})

    hashes = download_all(tmp_path, specs=SPECS, fetch=fetch)

    assert hashes == {
        "toy/a.csv": hashlib.sha256(b"1,2\n").hexdigest(),
        "toy/b.csv": hashlib.sha256(b"3,4\n").hexdigest(),
    }
    assert json.loads((tmp_path / HASH_FILE).read_text(encoding="utf-8")) == hashes
    assert fetch.calls == ["someone/toy"]


def test_second_run_verifies_without_fetching_again(tmp_path: Path) -> None:
    fetch = FakeFetch({"a.csv": b"1,2\n", "b.csv": b"3,4\n"})
    first = download_all(tmp_path, specs=SPECS, fetch=fetch)

    second = download_all(tmp_path, specs=SPECS, fetch=fetch)

    assert second == first
    assert fetch.calls == ["someone/toy"]


def test_changed_raw_file_is_an_error_not_a_silent_rehash(tmp_path: Path) -> None:
    fetch = FakeFetch({"a.csv": b"1,2\n", "b.csv": b"3,4\n"})
    download_all(tmp_path, specs=SPECS, fetch=fetch)
    (tmp_path / "toy" / "a.csv").write_bytes(b"tampered\n")

    with pytest.raises(RawDataChangedError, match=r"toy/a\.csv"):
        download_all(tmp_path, specs=SPECS, fetch=fetch)


def test_missing_expected_file_after_fetch_is_an_error(tmp_path: Path) -> None:
    fetch = FakeFetch({"a.csv": b"1,2\n"})

    with pytest.raises(FileNotFoundError, match=r"b\.csv"):
        download_all(tmp_path, specs=SPECS, fetch=fetch)


def test_real_dataset_specs_name_the_expected_files() -> None:
    assert DATASETS["sparkov"].handle == "kartik2112/fraud-detection"
    assert DATASETS["sparkov"].files == ("fraudTrain.csv", "fraudTest.csv")
    assert DATASETS["ulb"].handle == "mlg-ulb/creditcardfraud"
    assert DATASETS["ulb"].files == ("creditcard.csv",)
