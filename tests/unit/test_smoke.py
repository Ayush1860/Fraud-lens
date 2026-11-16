"""Smoke test: the package installs and imports."""

import re

import fraudlens


def test_package_exposes_a_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", fraudlens.__version__)
