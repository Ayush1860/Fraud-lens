"""Runtime configuration, read from ``FRAUDLENS_*`` environment variables.

Nothing here has a secret default: a missing key fails fast at start-up instead of
letting the service run with a guessable value.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

NonEmptySecret = Annotated[SecretStr, Field(min_length=1)]
NonEmptyStr = Annotated[str, Field(min_length=1)]


class Settings(BaseSettings):
    """Settings shared by the service, the pipeline and the tools.

    Values come from the process environment first, then from a ``.env`` file in the
    working directory. Unprefixed keys in that file (for example the ``POSTGRES_*``
    values Docker Compose reads) are ignored.
    """

    model_config = SettingsConfigDict(
        env_prefix="FRAUDLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Literal["local", "test", "demo"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    api_key: NonEmptySecret
    """Static key the scoring endpoint requires from its caller."""

    card_hmac_key: NonEmptySecret
    """Key for the HMAC-SHA256 that turns a card number into a ``card_id`` token."""

    database_url: NonEmptyStr
    """SQLAlchemy async URL of the Postgres audit store."""

    redis_url: NonEmptyStr
    """URL of the Redis online feature store."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first use.

    Raises:
        pydantic.ValidationError: if a required value is missing or invalid.
    """
    return Settings()
