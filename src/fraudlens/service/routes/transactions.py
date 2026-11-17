"""Scoring, stored decisions and explanations."""

import logging
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status

from fraudlens.common.logging import mask_card_id
from fraudlens.explain.reason_codes import MAX_REASONS, ReasonCode, explain
from fraudlens.features.definitions import FEATURE_NAMES
from fraudlens.features.encoding import FEATURE_TYPES, encode_vector
from fraudlens.service.decision import HISTORY_UNAVAILABLE
from fraudlens.service.schemas import (
    CardTransactionOut,
    DecisionRecordOut,
    ExplanationOut,
    ScoreRequest,
    ScoreResponse,
)
from fraudlens.service.state import ServiceState, get_state, require_api_key
from fraudlens.training.bundle import logistic

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/transactions", dependencies=[Depends(require_api_key)])

State = Annotated[ServiceState, Depends(get_state)]
TransactionId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")]

EXPLANATION_CACHE_S = 86_400


def _reason(reason: ReasonCode) -> dict[str, Any]:
    return {
        "id": reason.id,
        "text": reason.text,
        "share": reason.share,
        "contribution": reason.contribution,
    }


@router.post("/score", response_model=ScoreResponse)
async def score_transaction(body: ScoreRequest, response: Response, state: State) -> Any:
    """Score one transaction and decide. Idempotent on ``transaction_id``.

    A retry with the same ``transaction_id`` returns the original decision, with the
    ``Idempotent-Replay: true`` header, and does not touch the card's history.
    """
    outcome = await state.engine.score(body.to_transaction())
    if outcome.replayed:
        response.headers["Idempotent-Replay"] = "true"
    logger.info(
        "transaction scored",
        extra={
            "transaction_id": body.transaction_id,
            "card": mask_card_id(body.card_id),
            "decision": outcome.body["decision"],
            "degraded": outcome.body["degraded"],
            "replayed": outcome.replayed,
        },
    )
    return outcome.body


@router.get("/{transaction_id}", response_model=DecisionRecordOut)
async def get_transaction(transaction_id: TransactionId, state: State) -> Any:
    """The stored decision for one transaction, with its feature vector."""
    record = await state.audit.get_decision(transaction_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no decision for this transaction")
    return {
        "transaction_id": record.transaction_id,
        "card_id": record.card_id,
        "event_time": datetime.fromtimestamp(record.event_ts, tz=UTC),
        "amount": record.amount,
        "category": record.category,
        "features": _json_features(record.features),
        "margin": record.margin,
        "risk_score": record.score,
        "decision": record.decision,
        "t_review": record.t_review,
        "t_block": record.t_block,
        "model_version": record.model_version,
        "reason_codes": [_reason(r) for r in record.reason_codes],
        "degraded": record.degraded,
        "latency_ms": record.latency_ms,
        "created_at": record.created_at,
    }


@router.get("/{transaction_id}/card-history", response_model=list[CardTransactionOut])
async def get_card_history(
    transaction_id: TransactionId,
    state: State,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Any:
    """The card's decisions up to and including this one, newest first."""
    record = await state.audit.get_decision(transaction_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no decision for this transaction")
    rows = await state.audit.card_history(
        record.card_id, until=datetime.fromtimestamp(record.event_ts, tz=UTC), limit=limit
    )
    return [asdict(row) for row in rows]


@router.get("/{transaction_id}/explanation", response_model=ExplanationOut)
async def get_explanation(transaction_id: TransactionId, state: State) -> Any:
    """Every feature's contribution and the reason codes, recomputed from stored features.

    Contributions are in log-odds. The result is cached per model version.
    """
    record = await state.audit.get_decision(transaction_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no decision for this transaction")
    bundle = state.bundle
    if record.model_version != bundle.version:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"scored by model {record.model_version}; the active model is {bundle.version}",
        )
    cache_key = f"expl:{transaction_id}:{bundle.version}"
    cached = await state.store.cache_get(cache_key)
    if cached is not None:
        return cached

    X = encode_vector(record.features)
    explanation = explain(
        bundle.booster, X, [record.features], FEATURE_NAMES, FEATURE_TYPES, state.templates
    )[0]
    ranked = sorted(explanation.contributions.items(), key=lambda item: -abs(item[1]))
    values = _json_features(record.features)
    body = {
        "transaction_id": transaction_id,
        "model_version": bundle.version,
        "bias": explanation.bias,
        "margin": explanation.margin,
        "risk_score": float(bundle.calibrator(logistic(bundle.margin(X)))[0]),
        "contributions": [
            {"feature": name, "value": values[name], "contribution": value}
            for name, value in ranked
        ],
        "groups": [
            {"id": gid, "title": state.templates.titles[gid], "contribution": value}
            for gid, value in explanation.groups.items()
        ],
        "reason_codes": [
            *(_reason(r) for r in record.reason_codes if r.id == HISTORY_UNAVAILABLE),
            *(_reason(r) for r in explanation.reasons),
        ][:MAX_REASONS],
        "mitigating": None if explanation.mitigating is None else _reason(explanation.mitigating),
    }
    await state.store.cache_set(cache_key, body, EXPLANATION_CACHE_S)
    return body


def _json_features(features: dict[str, float | int | str]) -> dict[str, float | int | str | None]:
    return {
        name: None if isinstance(value, float) and value != value else value  # NaN check
        for name, value in features.items()
    }
