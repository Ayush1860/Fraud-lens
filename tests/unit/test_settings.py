"""Settings load from FRAUDLENS_* environment variables and keep secrets out of reprs."""

import os

import pytest
from pydantic import ValidationError

from fraudlens.common.settings import Settings, get_settings

REQUIRED = {
    "FRAUDLENS_API_KEY": "test-api-key",
    "FRAUDLENS_CARD_HMAC_KEY": "test-hmac-key",
    "FRAUDLENS_DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/fraudlens",
    "FRAUDLENS_REDIS_URL": "redis://localhost:6379/0",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Start from a clean FRAUDLENS_* environment with every required value set."""
    for name in list(os.environ):
        if name.startswith("FRAUDLENS_"):
            monkeypatch.delenv(name)
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def load() -> Settings:
    """Build settings from the process environment only, ignoring any local .env file."""
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_reads_prefixed_environment_variables(env: pytest.MonkeyPatch) -> None:
    env.setenv("FRAUDLENS_LOG_LEVEL", "DEBUG")
    env.setenv("FRAUDLENS_ENVIRONMENT", "demo")

    settings = load()

    assert settings.log_level == "DEBUG"
    assert settings.environment == "demo"
    assert settings.redis_url == REQUIRED["FRAUDLENS_REDIS_URL"]
    assert settings.database_url == REQUIRED["FRAUDLENS_DATABASE_URL"]
    assert settings.api_key.get_secret_value() == "test-api-key"
    assert settings.card_hmac_key.get_secret_value() == "test-hmac-key"


def test_defaults_are_local_and_info(env: pytest.MonkeyPatch) -> None:
    settings = load()

    assert settings.environment == "local"
    assert settings.log_level == "INFO"


def test_secrets_never_appear_in_repr(env: pytest.MonkeyPatch) -> None:
    text = repr(load())

    assert "test-api-key" not in text
    assert "test-hmac-key" not in text


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_every_required_value_must_be_set(env: pytest.MonkeyPatch, missing: str) -> None:
    env.delenv(missing)

    with pytest.raises(ValidationError):
        load()


def test_rejects_an_unknown_log_level(env: pytest.MonkeyPatch) -> None:
    env.setenv("FRAUDLENS_LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError):
        load()


def test_rejects_an_empty_secret(env: pytest.MonkeyPatch) -> None:
    env.setenv("FRAUDLENS_API_KEY", "")

    with pytest.raises(ValidationError):
        load()


def test_get_settings_is_cached(env: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
