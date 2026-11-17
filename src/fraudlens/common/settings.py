"""Runtime configuration, read from ``FRAUDLENS_*`` environment variables.

Nothing here has a secret default: a missing key fails fast at start-up instead of
letting the service run with a guessable value.
"""

from functools import lru_cache
from pathlib import Path
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

    bundle_dir: Path | None = None
    """Model bundle the service loads. When unset, the newest release under ``models_dir``."""

    models_dir: Path = Path("models")
    """Directory holding released bundles, ``models/<YYYY-MM-DD.N>/``."""

    degraded_review_amount: Annotated[float, Field(gt=0)] = 500.0
    """Dollars. Without card history, any larger amount goes to review at least."""

    redis_timeout_ms: Annotated[int, Field(gt=0)] = 50
    """Milliseconds allowed for each Redis call before the request takes the degraded path."""

    card_lock_timeout_ms: Annotated[int, Field(gt=0)] = 150
    """Milliseconds a request waits for its card's lock before taking the degraded path."""

    database_timeout_ms: Annotated[int, Field(gt=0)] = 2_000
    """Milliseconds allowed for an audit write before the request fails with 503."""

    idempotency_ttl_s: Annotated[int, Field(gt=0)] = 86_400
    """Seconds a decision stays in Redis for idempotent replays (Postgres keeps it forever)."""

    service_url: NonEmptyStr = "http://localhost:8000"
    """Base URL the replay client posts to."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first use.

    Raises:
        pydantic.ValidationError: if a required value is missing or invalid.
    """
    return Settings()
