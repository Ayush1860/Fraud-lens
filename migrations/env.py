"""Alembic environment: run migrations on the asyncpg engine.

The URL is, in order: ``config.attributes["database_url"]`` (set by tests), ``-x url=...``
on the command line, then ``FRAUDLENS_DATABASE_URL`` through the service settings.
"""

import asyncio

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from fraudlens.common.settings import get_settings

config = context.config


def database_url() -> str:
    """The URL to migrate."""
    explicit = config.attributes.get("database_url") or context.get_x_argument(
        as_dictionary=True
    ).get("url")
    return str(explicit) if explicit else get_settings().database_url


def run_offline() -> None:
    """Emit the SQL instead of running it (``alembic upgrade head --sql``)."""
    context.configure(url=database_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(connection=connection, transaction_per_migration=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_online() -> None:
    """Run the migrations on a live database."""
    engine = create_async_engine(database_url())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run)
            await connection.commit()
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    asyncio.run(run_online())
