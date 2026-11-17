"""Real Redis and Postgres for the service's integration tests.

The stores come from ``FRAUDLENS_TEST_REDIS_URL`` and ``FRAUDLENS_TEST_DATABASE_URL``;
the tests are skipped when either is unset. CI sets both to its service containers.
Every test starts from empty tables and an empty Redis database.
"""

import os
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

import httpx
import pytest
import redis
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from asgi_lifespan import LifespanManager

from fraudlens.common.settings import Settings
from fraudlens.service.app import create_app

API_KEY = "integration-test-key"
DEAD_REDIS_URL = "redis://127.0.0.1:1/0"
DEAD_DATABASE_URL = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none"

ClientFactory = Callable[..., AbstractAsyncContextManager[httpx.AsyncClient]]


@pytest.fixture(scope="session")
def redis_url() -> str:
    url = os.environ.get("FRAUDLENS_TEST_REDIS_URL")
    if not url:
        pytest.skip("FRAUDLENS_TEST_REDIS_URL is not set")
    return url


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("FRAUDLENS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("FRAUDLENS_TEST_DATABASE_URL is not set")
    return url


@pytest.fixture(scope="session")
def bundle_dir(toy_bundle_dir: Path) -> Path:
    return toy_bundle_dir


@pytest.fixture(scope="session")
def migrated(database_url: str) -> str:
    """Apply every migration from scratch, once per session."""
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.attributes["database_url"] = database_url
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    return database_url


def sync_url(database_url: str) -> str:
    """The same database through a synchronous driver, for test set-up and assertions."""
    return database_url.replace("+asyncpg", "+psycopg")


@pytest.fixture
def db(migrated: str) -> Iterator[sa.Engine]:
    """A synchronous engine on the test database, emptied before each test."""
    engine = sa.create_engine(sync_url(migrated))
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "TRUNCATE dispositions, alerts, decisions, model_versions RESTART IDENTITY CASCADE"
            )
        )
    yield engine
    engine.dispose()


@pytest.fixture
def redis_db(redis_url: str) -> Iterator[redis.Redis]:
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    client.flushdb()
    yield client
    client.close()


def settings_for(bundle: Path, *, redis_url: str, database_url: str, **extra: object) -> Settings:
    # Generous Redis timeouts: a slow CI runner must not turn a test degraded by accident.
    extra = {"redis_timeout_ms": 1_000, **extra}
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        environment="test",
        api_key=API_KEY,
        card_hmac_key="integration-test-hmac",
        database_url=database_url,
        redis_url=redis_url,
        bundle_dir=bundle,
        **extra,
    )


@pytest.fixture
def make_client(
    bundle_dir: Path, migrated: str, redis_url: str, db: sa.Engine, redis_db: redis.Redis
) -> ClientFactory:
    """Start the app with its lifespan and return an HTTP client bound to it."""

    @asynccontextmanager
    async def factory(
        *, redis: str = redis_url, database: str = migrated, **extra: object
    ) -> AsyncIterator[httpx.AsyncClient]:
        settings = settings_for(bundle_dir, redis_url=redis, database_url=database, **extra)
        app = create_app(settings)
        async with (
            LifespanManager(app) as manager,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://service",
                headers={"X-API-Key": API_KEY},
                timeout=30.0,
            ) as client,
        ):
            yield client

    return factory
