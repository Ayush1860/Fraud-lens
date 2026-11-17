"""Prometheus metrics for ``/metrics``: request and per-stage latency, outcomes, scores.

Each app gets its own registry, so several apps in one process (as in tests) never
collide. All durations are in seconds.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.015, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0)
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
            buckets=LATENCY_BUCKETS,
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
        self.model = Gauge(
            "fraudlens_model_info",
            "Always 1; the label names the active model version.",
            ["version"],
            registry=self.registry,
        )

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
        """The metrics in the Prometheus text format."""
        return generate_latest(self.registry)
