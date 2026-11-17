"""Fixtures shared across test directories."""

from pathlib import Path

import pytest

from toy_bundle import build_toy_bundle


@pytest.fixture(scope="session")
def toy_bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A released toy bundle, trained once per session on first use."""
    return build_toy_bundle(tmp_path_factory.mktemp("toy-models"))


@pytest.fixture
def anyio_backend() -> str:
    """Run ``@pytest.mark.anyio`` tests on asyncio only."""
    return "asyncio"
