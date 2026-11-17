"""Prometheus metrics for ``/metrics``: request and per-stage latency, outcomes, scores.

Each app gets its own registry, so several apps in one process (as in tests) never
collide. All durations are in seconds.

With several uvicorn workers, each is a separate process with its own counters. Set
``PROMETHEUS_MULTIPROC_DIR`` (the Docker image does) and every worker writes its values to
files there; ``/metrics`` then sums all workers instead of reporting whichever one answered.
"""

import os
from collections.abc import Mapping
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.multiprocess import MultiProcessCollector

MULTIPROC_ENV = "PROMETHEUS_MULTIPROC_DIR"

STAGE_BUCKETS = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.002,
    0.004,
    0.008,
    0.016,
    0.032,
    0.064,
    0.128,
    0.256,
    0.512,
    1.024,
    2.048,
)
"""Stage timings, seconds: fine enough to resolve sub-millisecond stages such as features,
and wide enough to see a saturated stage."""

LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.015,
    0.025,
    0.05,
    0.1,
    0.15,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)
SCORE_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0)


class ServiceMetrics:
    """The service's metrics, registered on a private registry."""

    def __init__(self) -> None:
        """Create every metric on a new registry."""
        self.registry = CollectorRegistry()
        self.request_seconds = Histogram(
            "fraudlens_request_seconds",
            "Request latency by route, method and status code, in seconds.",
            ["route", "method", "status"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.stage_seconds = Histogram(
            "fraudlens_stage_seconds",
            "Latency of each stage of the scoring hot path, in seconds.",
            ["stage"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.decisions = Counter(
            "fraudlens_decisions",
            "Scoring responses by decision, replays included.",
            ["decision"],
            registry=self.registry,
        )
        self.degraded = Counter(
            "fraudlens_degraded_responses",
            "Decisions made without the card's history.",
            registry=self.registry,
        )
        self.replays = Counter(
            "fraudlens_idempotent_replays",
            "Requests answered with an earlier decision for the same transaction_id.",
            registry=self.registry,
        )
        self.scores = Histogram(
            "fraudlens_risk_score",
            "Calibrated risk scores of new decisions.",
            buckets=SCORE_BUCKETS,
            registry=self.registry,
        )
        self.drift_psi = Gauge(
            "fraudlens_drift_psi",
            "Population stability index from the latest drift report.",
            ["kind", "name"],
            registry=self.registry,
            multiprocess_mode="mostrecent",
        )
        self.drift_alarms = Gauge(
            "fraudlens_drift_alarms",
            "Number of alarms in the latest drift report.",
            registry=self.registry,
            multiprocess_mode="mostrecent",
        )
        self.model = Gauge(
            "fraudlens_model_info",
            "Always 1; the label names the active model version.",
            ["version"],
            registry=self.registry,
            multiprocess_mode="max",
        )

    def publish_drift(self, report: Mapping[str, Any]) -> None:
        """Expose a drift report: PSI per feature and segment, and the alarm count."""
        self.drift_psi.clear()
        for row in report["features"]:
            if row["psi"] is not None:
                self.drift_psi.labels(kind="feature", name=row["feature"]).set(row["psi"])
        for row in report["segments"]:
            if row["psi"] is not None:
                self.drift_psi.labels(kind="segment", name=row["category"]).set(row["psi"])
        if report["decision_mix"]["psi"] is not None:
            self.drift_psi.labels(kind="decision_mix", name="decision mix").set(
                report["decision_mix"]["psi"]
            )
        self.drift_alarms.set(len(report["alarms"]))

    def stage(self, name: str, seconds: float) -> None:
        """Record one stage of one request."""
        self.stage_seconds.labels(stage=name).observe(seconds)

    def decided(self, decision: str, *, degraded: bool, replayed: bool, score: float) -> None:
        """Record one scoring outcome. Replays are counted but not scored twice."""
        self.decisions.labels(decision=decision).inc()
        if replayed:
            self.replays.inc()
            return
        self.scores.observe(score)
        if degraded:
            self.degraded.inc()

    def render(self) -> bytes:
        """The metrics in the Prometheus text format, summed over workers if multiprocess."""
        directory = os.environ.get(MULTIPROC_ENV)
        if directory:
            registry = CollectorRegistry()
            MultiProcessCollector(registry, path=directory)  # type: ignore[no-untyped-call]
            return generate_latest(registry)
        return generate_latest(self.registry)
