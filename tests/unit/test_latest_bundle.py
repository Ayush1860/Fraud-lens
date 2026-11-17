"""Picking the bundle the service loads by default."""

from pathlib import Path

import pytest

from fraudlens.training.bundle import latest_bundle


def test_the_newest_release_wins_by_date_then_sequence(tmp_path: Path) -> None:
    for name in ("2026-11-17.2", "2026-11-17.10", "2026-11-16.30", "notes", "2026-11-18"):
        (tmp_path / name).mkdir()
    (tmp_path / "2026-12-01.1").write_text("a file, not a release")

    assert latest_bundle(tmp_path).name == "2026-11-17.10"


def test_no_release_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        latest_bundle(tmp_path)
    with pytest.raises(FileNotFoundError):
        latest_bundle(tmp_path / "missing")
