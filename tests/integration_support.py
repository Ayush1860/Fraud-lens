"""Names the integration tests share with their fixtures.

Kept out of ``conftest.py`` on purpose: there is more than one ``conftest`` module in
the test tree, so ``from conftest import ...`` would depend on import order.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

import httpx

API_KEY = "integration-test-key"
DEAD_REDIS_URL = "redis://127.0.0.1:1/0"
DEAD_DATABASE_URL = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none"

ClientFactory = Callable[..., AbstractAsyncContextManager[httpx.AsyncClient]]
"""Starts the app with its lifespan and yields an HTTP client bound to it."""
