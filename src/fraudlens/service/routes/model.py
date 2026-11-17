"""The active model, the monitoring summary, and health checks."""

from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, Response, status

from fraudlens.features.definitions import CATEGORIES, FEATURE_NAMES
from fraudlens.features.encoding import FEATURE_TYPES
from fraudlens.service.repository import RecentDecision
from fraudlens.service.schemas import (
    Health,
    ModelInfo,
    MonitoringSummary,
    Readiness,
    ThresholdLab,
)
from fraudlens.service.state import ServiceState, get_state, require_api_key
from fraudlens.training.thresholds import price_curve

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
health = APIRouter()

State = Annotated[ServiceState, Depends(get_state)]

SUMMARY_WINDOW = 10_000
SCORE_BINS = 20


@router.get("/model", response_model=ModelInfo)
async def get_model(state: State) -> Any:
    """The active release: version, manifest hash, thresholds, features and metrics."""
    bundle = state.bundle
    return {
        "version": bundle.version,
        "manifest_sha256": state.manifest_hash,
        "thresholds": {
            "t_review": bundle.thresholds.t_review,
            "t_block": bundle.thresholds.t_block,
        },
        "features": list(FEATURE_NAMES),
        "feature_types": list(FEATURE_TYPES),
        "categories": list(CATEGORIES),
        "metrics": bundle.documents.get("metrics.json", {}),
    }


@router.get("/model/threshold-curve", response_model=ThresholdLab)
async def get_threshold_curve(state: State) -> Any:
    """The validation threshold curve with every threshold pair priced.

    Prices come from the same function that chose the released thresholds.
    """
    bundle = state.bundle
    curve = bundle.documents["threshold_curve.json"]
    thresholds = [point["threshold"] for point in curve["points"]]
    return {
        "model_version": bundle.version,
        "rows": curve["rows"],
        "days": curve["days"],
        "frauds": curve["frauds"],
        "fraud_amount": curve["fraud_amount"],
        "costs": curve["costs"],
        "points": curve["points"],
        "pairs": price_curve(curve),
        "current_review": _nearest(thresholds, bundle.thresholds.t_review),
        "current_block": _nearest(thresholds, bundle.thresholds.t_block),
    }


def _nearest(values: Sequence[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


@router.get("/monitoring/summary", response_model=MonitoringSummary)
async def get_monitoring_summary(state: State) -> Any:
    """Decision mix, score histogram, latency and analyst labels.

    Everything except the labels covers the newest 10,000 decisions.
    """
    recent = await state.audit.recent_decisions(SUMMARY_WINDOW)
    labels = await state.audit.disposition_counts()
    return summarise(recent, model_version=state.bundle.version, labels=labels)


def summarise(
    recent: Sequence[RecentDecision],
    *,
    model_version: str,
    labels: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Summarise recent decisions; percentiles are ``None`` when there are none."""
    total = len(recent)
    kinds = [d.decision for d in recent]
    scores = np.array([d.score for d in recent], dtype=np.float64)
    latency = np.array([d.latency_ms for d in recent], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, SCORE_BINS + 1)
    counts, _ = np.histogram(scores, bins=edges)

    def percentile(q: float) -> float | None:
        return float(np.percentile(latency, q)) if total else None

    return {
        "window": total,
        "model_version": model_version,
        "decisions": {
            "total": total,
            "approve": kinds.count("approve"),
            "review": kinds.count("review"),
            "block": kinds.count("block"),
            "degraded": sum(d.degraded for d in recent),
        },
        "alert_rate": (total - kinds.count("approve")) / total if total else None,
        "score_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
        "latency_ms": {"p50": percentile(50), "p95": percentile(95), "p99": percentile(99)},
        "dispositions": _label_summary(labels or {}),
        "drift": None,
    }


def _label_summary(labels: dict[str, int]) -> dict[str, Any]:
    fraud, genuine = labels.get("fraud", 0), labels.get("genuine", 0)
    labelled = fraud + genuine
    return {
        "fraud": fraud,
        "genuine": genuine,
        "realised_precision": fraud / labelled if labelled else None,
    }


@health.get("/healthz", response_model=Health)
async def healthz() -> Any:
    """Liveness: the process is up."""
    return {"status": "ok"}


@health.get("/readyz", response_model=Readiness)
async def readyz(state: State, response: Response) -> Any:
    """Readiness: the bundle verified at start-up and Postgres answers.

    Redis being down does not make the service unready: it scores degraded instead.
    """
    postgres = await state.audit.ping()
    redis = await state.store.ping()
    if not postgres:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "ready": postgres,
        "model_version": state.bundle.version,
        "postgres": postgres,
        "redis": redis,
    }


@health.get("/metrics", include_in_schema=False)
async def metrics(state: State) -> Response:
    """Prometheus metrics in the text exposition format."""
    return Response(state.metrics.render(), media_type="text/plain; version=0.0.4")
