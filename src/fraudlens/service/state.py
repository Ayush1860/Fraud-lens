"""What the routes share: settings, the bundle, the stores and the decision engine."""

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from fraudlens.common.settings import Settings
from fraudlens.explain.reason_codes import Templates
from fraudlens.service.decision import AuditLog, DecisionEngine, FeatureStore
from fraudlens.service.metrics import ServiceMetrics
from fraudlens.service.repository import (
    AlertRow,
    CardTransaction,
    DispositionRow,
    RecentDecision,
)
from fraudlens.training.bundle import ModelBundle


class ServiceStore(FeatureStore, Protocol):
    """The feature store plus the health check and the explanation cache."""

    async def ping(self) -> bool:
        """True if the store answers."""

    async def cache_get(self, key: str) -> Any | None:
        """A cached JSON document, or ``None``."""

    async def cache_set(self, key: str, value: Any, ttl_s: int) -> None:
        """Cache a JSON document, best effort."""


class ServiceAudit(AuditLog, Protocol):
    """The audit log plus the queries behind the read endpoints."""

    async def ping(self) -> bool:
        """True if the database answers."""

    async def list_alerts(
        self, *, status: str | None, decision: str | None, limit: int, cursor: str | None
    ) -> tuple[list[AlertRow], str | None]:
        """One page of alerts by priority, and the next cursor."""

    async def add_disposition(
        self, alert_id: int, *, label: str, analyst: str, note: str | None
    ) -> DispositionRow | None:
        """Record a label and close the alert; ``None`` if the alert does not exist."""

    async def record_model_version(
        self, version: str, *, manifest_hash: str, metrics: Mapping[str, Any]
    ) -> None:
        """Record a release's first activation."""

    async def recent_decisions(self, limit: int) -> list[RecentDecision]:
        """The newest decisions, newest first."""

    async def card_history(
        self, card_id: str, *, until: datetime, limit: int
    ) -> list[CardTransaction]:
        """The card's decisions at or before ``until``, newest first."""

    async def disposition_counts(self) -> dict[str, int]:
        """Alerts per label, counting each alert's latest label."""


@dataclass
class ServiceState:
    """Everything a request handler needs, created once in the app's lifespan."""

    settings: Settings
    bundle: ModelBundle
    manifest_hash: str
    store: ServiceStore
    audit: ServiceAudit
    engine: DecisionEngine
    metrics: ServiceMetrics
    templates: Templates


def get_state(request: Request) -> ServiceState:
    """The app's ``ServiceState``."""
    state: ServiceState = request.app.state.service
    return state


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, key: str | None = Security(_api_key_header)) -> None:
    """Reject the request with 401 unless it carries the configured API key."""
    expected = get_state(request).settings.api_key.get_secret_value().encode()
    if key is None or not hmac.compare_digest(key.encode(), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
