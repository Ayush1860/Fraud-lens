"""The HTTP contract: field names from plan section 7.2, frozen in an OpenAPI snapshot.

A deliberate API change regenerates the snapshot:

    FRAUDLENS_UPDATE_SNAPSHOT=1 uv run pytest tests/contract
"""

import json
import os
from pathlib import Path

from fraudlens.service.app import create_app
from fraudlens.service.schemas import ScoreRequest, ScoreResponse

SNAPSHOT = Path(__file__).with_name("openapi.json")

REQUEST_FIELDS = {
    "transaction_id",
    "card_id",
    "event_time",
    "amount",
    "category",
    "merchant_id",
    "merchant_lat",
    "merchant_long",
    "home_lat",
    "home_long",
    "city_pop",
    "cardholder_age",
}
RESPONSE_FIELDS = {
    "transaction_id",
    "risk_score",
    "decision",
    "model_version",
    "degraded",
    "reason_codes",
    "latency_ms",
}


def test_request_and_response_fields_match_the_plan() -> None:
    assert set(ScoreRequest.model_fields) == REQUEST_FIELDS
    assert set(ScoreResponse.model_fields) == RESPONSE_FIELDS


def test_openapi_document_matches_the_snapshot() -> None:
    document = create_app().openapi()
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if os.environ.get("FRAUDLENS_UPDATE_SNAPSHOT") == "1":
        SNAPSHOT.write_text(rendered, encoding="utf-8")
    assert rendered == SNAPSHOT.read_text(encoding="utf-8"), (
        "the API contract changed; if that is deliberate, regenerate tests/contract/openapi.json"
    )


def test_every_documented_route_exists() -> None:
    paths = set(create_app().openapi()["paths"])

    assert paths == {
        "/v1/transactions/score",
        "/v1/transactions/{transaction_id}",
        "/v1/transactions/{transaction_id}/explanation",
        "/v1/transactions/{transaction_id}/card-history",
        "/v1/alerts",
        "/v1/alerts/{alert_id}/disposition",
        "/v1/model",
        "/v1/model/threshold-curve",
        "/v1/monitoring/summary",
        "/healthz",
        "/readyz",
    }
