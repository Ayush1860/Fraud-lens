"""Request and response bodies of the HTTP API (plan section 7.2).

Field names are the contract; the contract tests compare the OpenAPI document with a
committed snapshot so that a change to any of them is deliberate.
"""

import math
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from fraudlens.common.types import Transaction
from fraudlens.features.definitions import normalise_category

MAX_FUTURE_SKEW = timedelta(minutes=5)
"""How far ahead of the service clock an ``event_time`` may be."""

Latitude = Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0, allow_inf_nan=False)]
Decision = Literal["approve", "review", "block"]


class ScoreRequest(BaseModel):
    """One transaction to score.

    ``card_id`` must be an HMAC token (``c_`` and 16 hex characters), so a raw card number
    is rejected at the door. An unknown ``category`` is scored as ``unknown``. Optional
    fields that are missing are passed to the model as missing values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
    card_id: Annotated[str, Field(pattern=r"^c_[0-9a-f]{16}$")]
    event_time: AwareDatetime
    amount: Annotated[float, Field(gt=0.0, le=1_000_000.0, allow_inf_nan=False)]
    """Dollars."""
    category: Annotated[str, Field(min_length=1, max_length=64)]
    merchant_id: Annotated[str, Field(min_length=1, max_length=128)]
    merchant_lat: Latitude | None = None
    merchant_long: Longitude | None = None
    home_lat: Latitude | None = None
    home_long: Longitude | None = None
    city_pop: Annotated[int, Field(ge=0)] | None = None
    cardholder_age: Annotated[int, Field(ge=0, le=130)] | None = None

    @field_validator("event_time")
    @classmethod
    def _not_in_the_future(cls, value: datetime) -> datetime:
        if value > datetime.now(UTC) + MAX_FUTURE_SKEW:
            raise ValueError("event_time is more than 5 minutes in the future")
        return value

    def to_transaction(self) -> Transaction:
        """The value object the feature definition takes: whole UTC seconds, NaN if absent."""

        def number(value: float | None) -> float:
            return math.nan if value is None else float(value)

        return Transaction(
            transaction_id=self.transaction_id,
            card_id=self.card_id,
            event_ts=math.floor(self.event_time.timestamp()),
            amount=self.amount,
            category=normalise_category(self.category),
            merchant_id=self.merchant_id,
            merchant_lat=number(self.merchant_lat),
            merchant_long=number(self.merchant_long),
            home_lat=number(self.home_lat),
            home_long=number(self.home_long),
            city_pop=number(self.city_pop),
            cardholder_age=number(self.cardholder_age),
        )


class ReasonCodeOut(BaseModel):
    """One reason: a stable id, its wording, and its share of the upward push (0 to 1)."""

    id: str
    text: str
    share: float


class ScoreResponse(BaseModel):
    """The decision for one transaction."""

    transaction_id: str
    risk_score: float
    """Calibrated probability of fraud."""
    decision: Decision
    model_version: str
    degraded: bool
    """True when the card's history could not be read and the transaction was scored alone."""
    reason_codes: list[ReasonCodeOut]
    """Empty for an approval."""
    latency_ms: int
    """Milliseconds from the start of the decision to the audit write."""


class StoredReasonCode(ReasonCodeOut):
    """A reason as stored, with its contribution in log-odds."""

    contribution: float


class DecisionRecordOut(BaseModel):
    """A stored decision with the feature vector it was made from."""

    transaction_id: str
    card_id: str
    event_time: datetime
    amount: float
    category: str
    features: dict[str, float | int | str | None]
    """Feature values; ``null`` where the feature was missing."""
    margin: float
    """Model output in log-odds."""
    risk_score: float
    decision: Decision
    t_review: float
    t_block: float
    model_version: str
    reason_codes: list[StoredReasonCode]
    degraded: bool
    latency_ms: int
    created_at: datetime


class CardTransactionOut(BaseModel):
    """One decision on the same card, for the case view's history table."""

    transaction_id: str
    event_time: datetime
    amount: float
    category: str
    decision: Decision
    risk_score: float
    degraded: bool


class FeatureContribution(BaseModel):
    """One feature's value and its contribution to the margin, in log-odds."""

    feature: str
    value: float | int | str | None
    contribution: float


class GroupContribution(BaseModel):
    """One reason group's summed contribution, in log-odds."""

    id: str
    title: str
    contribution: float


class ExplanationOut(BaseModel):
    """The full explanation of a stored decision, recomputed from its stored features."""

    transaction_id: str
    model_version: str
    bias: float
    """The model's expected margin, where the waterfall starts, in log-odds."""
    margin: float
    risk_score: float
    contributions: list[FeatureContribution]
    """Every feature, largest absolute contribution first."""
    groups: list[GroupContribution]
    reason_codes: list[StoredReasonCode]
    mitigating: StoredReasonCode | None


class AlertOut(BaseModel):
    """One alert in the queue."""

    id: int
    transaction_id: str
    priority: float
    """Risk score times amount, in dollars at risk."""
    status: Literal["open", "closed"]
    decision: Decision
    risk_score: float
    amount: float
    category: str
    event_time: datetime
    created_at: datetime
    top_reason: str | None
    """Wording of the leading reason code, if one could be worded."""


class AlertPage(BaseModel):
    """One page of alerts and the cursor for the next."""

    items: list[AlertOut]
    next_cursor: str | None


class DispositionIn(BaseModel):
    """An analyst's label for an alert."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["fraud", "genuine"]
    analyst: Annotated[str, Field(min_length=1, max_length=64)]
    note: Annotated[str, Field(max_length=2_000)] | None = None


class DispositionOut(BaseModel):
    """A recorded label."""

    id: int
    alert_id: int
    label: Literal["fraud", "genuine"]
    analyst: str
    note: str | None
    created_at: datetime


class Thresholds(BaseModel):
    """Decision thresholds on the calibrated probability."""

    t_review: float
    t_block: float


class ModelInfo(BaseModel):
    """The active model release."""

    version: str
    manifest_sha256: str
    thresholds: Thresholds
    features: list[str]
    feature_types: list[str]
    categories: list[str]
    metrics: dict[str, object]
    """The bundle's validation metrics, as released."""


class CurvePoint(BaseModel):
    """What happens on the validation slice when ``threshold`` is the alert threshold."""

    threshold: float
    count: float
    """Transactions at or above the threshold."""
    frauds: float
    fraud_amount: float
    genuine: float
    genuine_amount: float
    precision: float | None
    recall: float | None
    dollar_recall: float | None
    alerts_per_day: float


class PricedPair(BaseModel):
    """One ``(t_review, t_block)`` pair, by index into ``points``, and its price."""

    review: int
    block: int
    cost_per_1000: float
    """Expected cost in dollars per 1,000 transactions on the validation slice."""
    review_share: float
    within_capacity: bool


class CostAssumptions(BaseModel):
    """The cost assumptions the thresholds were chosen with, in dollars."""

    review_cost: float
    margin_rate: float
    friction_cost: float
    review_capacity_share: float


class ThresholdLab(BaseModel):
    """Everything the threshold lab needs: the curve, every priced pair, the release."""

    model_version: str
    rows: int
    days: int
    frauds: float
    fraud_amount: float
    costs: CostAssumptions
    points: list[CurvePoint]
    pairs: list[PricedPair]
    current_review: int
    """Index into ``points`` of the released review threshold."""
    current_block: int


class Histogram(BaseModel):
    """Counts between consecutive edges; the last bin includes its right edge."""

    edges: list[float]
    counts: list[int]


class LatencySummary(BaseModel):
    """Decision latency percentiles, in milliseconds."""

    p50: float | None
    p95: float | None
    p99: float | None


class DecisionCounts(BaseModel):
    """How the recent decisions split."""

    total: int
    approve: int
    review: int
    block: int
    degraded: int


class DispositionSummary(BaseModel):
    """Analyst labels so far, counting each alert's latest label once."""

    fraud: int
    genuine: int
    realised_precision: float | None
    """Share of labelled alerts confirmed as fraud."""


DriftStatus = Literal["stable", "watch", "alarm", "unknown"]


class FeatureDrift(BaseModel):
    """One feature against its training histogram."""

    feature: str
    psi: float | None
    status: DriftStatus
    cyclical: bool
    """A calendar feature whose cycle is longer than the window: reported, never alarmed."""


class SegmentDrift(BaseModel):
    """Amount within one category, the live window against the window before it."""

    category: str
    psi: float | None
    """``None`` when either window has too few rows in this category to judge."""
    status: DriftStatus
    n_live: int
    n_previous: int


class DecisionMixDrift(BaseModel):
    """The live decision mix against the validation slice's."""

    psi: float | None
    status: DriftStatus
    live: dict[str, float]
    validation: dict[str, float]


class DriftAlarm(BaseModel):
    """One comparison over the alarm level (PSI above 0.25)."""

    kind: Literal["feature", "segment", "decision_mix"]
    name: str
    psi: float


class DriftReport(BaseModel):
    """The latest drift report, recomputed in the background."""

    computed_at: datetime
    model_version: str
    window: int
    """Decisions in the live window."""
    previous_window: int
    """Decisions in the baseline window just before it."""
    features: list[FeatureDrift]
    segments: list[SegmentDrift]
    decision_mix: DecisionMixDrift
    alarms: list[DriftAlarm]


class MonitoringSummary(BaseModel):
    """The recent behaviour of the service, for the console's monitoring page."""

    window: int
    """Number of most recent decisions summarised."""
    model_version: str
    decisions: DecisionCounts
    alert_rate: float | None
    """Share of decisions sent to review or block."""
    score_histogram: Histogram
    latency_ms: LatencySummary
    dispositions: DispositionSummary
    drift: DriftReport | None
    """The latest drift report; ``None`` until the first has been computed."""


class Health(BaseModel):
    """Liveness."""

    status: Literal["ok"]


class Readiness(BaseModel):
    """Readiness: true once the bundle is verified and the audit store answers."""

    ready: bool
    model_version: str
    postgres: bool
    redis: bool
    """False means scoring runs degraded; it does not make the service unready."""
