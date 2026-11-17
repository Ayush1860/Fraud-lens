"""The FastAPI application: lifespan, middleware, error mapping and routes.

Run it with ``uvicorn fraudlens.service.app:create_app --factory``.

At start-up the lifespan handler loads the model bundle and verifies every file against
its manifest. A bundle that fails verification raises, so the process never starts
serving and readiness never turns true.
"""

import asyncio
import contextlib
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from sqlalchemy.ext.asyncio import create_async_engine

from fraudlens.common.logging import configure_logging, request_id_var
from fraudlens.common.settings import Settings, get_settings
from fraudlens.explain.reason_codes import load_templates
from fraudlens.service.decision import AuditUnavailable, DecisionEngine, StillProcessing
from fraudlens.service.feature_store import RedisFeatureStore
from fraudlens.service.metrics import ServiceMetrics
from fraudlens.service.repository import Repository
from fraudlens.service.routes import alerts, model, transactions
from fraudlens.service.state import ServiceAudit, ServiceState, ServiceStore
from fraudlens.training.bundle import ModelBundle, latest_bundle, manifest_sha256

logger = logging.getLogger(__name__)

Stores = tuple[ServiceStore, ServiceAudit]
Connector = Callable[[Settings], AbstractAsyncContextManager[Stores]]

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@asynccontextmanager
async def connect_stores(settings: Settings) -> AsyncIterator[Stores]:
    """Open the Redis client and the Postgres pool; close both on exit.

    Neither connects eagerly, so the service starts while a store is down: Redis being
    down means degraded scoring, Postgres being down means readiness stays false.
    """
    redis_timeout = settings.redis_timeout_ms / 1000
    database_timeout = settings.database_timeout_ms / 1000
    client = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=redis_timeout,
        socket_connect_timeout=redis_timeout,
        retry=Retry(NoBackoff(), 0),
    )
    engine = create_async_engine(
        settings.database_url,
        pool_size=10,
        max_overflow=10,
        pool_timeout=database_timeout,
        pool_pre_ping=True,
        connect_args={"timeout": database_timeout},
    )
    store = RedisFeatureStore(
        client,
        timeout_s=redis_timeout,
        lock_wait_s=settings.card_lock_timeout_ms / 1000,
        idempotency_ttl_s=settings.idempotency_ttl_s,
    )
    try:
        yield store, Repository(engine, timeout_s=database_timeout)
    finally:
        await client.aclose()
        await engine.dispose()


async def _refresh_drift_forever(state: ServiceState, interval_s: float) -> None:
    """Recompute the drift report now and then every ``interval_s`` seconds."""
    while True:
        try:
            await state.refresh_drift()
        except Exception:  # a broken report must never take the service down
            logger.exception("drift refresh failed")
        await asyncio.sleep(interval_s)


def create_app(settings: Settings | None = None, *, connect: Connector = connect_stores) -> FastAPI:
    """Build the application.

    Args:
        settings: Configuration; read from the environment at start-up when ``None``.
        connect: Opens the feature store and the audit log. Tests pass in-memory ones.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = settings or get_settings()
        configure_logging(config.log_level)
        bundle_dir = config.bundle_dir or latest_bundle(config.models_dir)
        bundle = ModelBundle.load(bundle_dir)  # raises BundleIntegrityError: never serve it
        bundle.booster.set_param({"nthread": config.model_threads})
        manifest_hash = manifest_sha256(bundle_dir)
        metrics = ServiceMetrics()
        metrics.model.labels(version=bundle.version).set(1)
        templates = load_templates()
        async with connect(config) as (store, audit):
            try:
                await audit.record_model_version(
                    bundle.version,
                    manifest_hash=manifest_hash,
                    metrics=bundle.documents.get("metrics.json", {}),
                )
            except AuditUnavailable:
                logger.warning("model version not recorded: audit store unavailable")
            app.state.service = ServiceState(
                settings=config,
                bundle=bundle,
                manifest_hash=manifest_hash,
                store=store,
                audit=audit,
                engine=DecisionEngine(
                    bundle=bundle,
                    store=store,
                    audit=audit,
                    degraded_review_amount=config.degraded_review_amount,
                    observer=metrics,
                    templates=templates,
                ),
                metrics=metrics,
                templates=templates,
            )
            logger.info("model loaded", extra={"model_version": bundle.version})
            refresher = asyncio.create_task(
                _refresh_drift_forever(app.state.service, config.drift_interval_s)
            )
            try:
                yield
            finally:
                refresher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresher

    app = FastAPI(
        title="FraudLens scoring service",
        version="1.0.0",
        summary="Scores card transactions, explains flagged ones, and records every decision.",
        lifespan=lifespan,
    )
    app.include_router(transactions.router)
    app.include_router(alerts.router)
    app.include_router(model.router)
    app.include_router(model.health)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        route = request.scope.get("route")
        service: ServiceState | None = getattr(request.app.state, "service", None)
        if service is not None:
            service.metrics.request_seconds.labels(
                route=getattr(route, "path", "unmatched"),
                method=request.method,
                status=str(response.status_code),
            ).observe(time.perf_counter() - started)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(AuditUnavailable)
    async def audit_unavailable(request: Request, error: AuditUnavailable) -> JSONResponse:
        logger.error("audit store unavailable", extra={"error": str(error)})
        return JSONResponse(
            {"detail": "audit store unavailable; retry with the same transaction_id"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(StillProcessing)
    async def still_processing(request: Request, error: StillProcessing) -> JSONResponse:
        return JSONResponse(
            {"detail": "a request with this transaction_id is still being decided"},
            status_code=status.HTTP_409_CONFLICT,
            headers={"Retry-After": "1"},
        )

    return app
