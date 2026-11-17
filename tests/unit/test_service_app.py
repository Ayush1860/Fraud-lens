"""The HTTP layer with in-memory stores: auth, routes, error mapping, start-up checks."""

import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from fraudlens.common.settings import Settings
from fraudlens.service.app import create_app
from fraudlens.service.decision import PENDING
from fraudlens.service.routes.model import summarise
from fraudlens.training.bundle import BundleIntegrityError
from service_fakes import MemoryAudit, MemoryStore

pytestmark = pytest.mark.anyio

KEY = "unit-test-key"
BODY: dict[str, Any] = {
    "transaction_id": "tx0001",
    "card_id": "c_0123456789abcdef",
    "event_time": "2019-06-01T03:12:44Z",
    "amount": 42.5,
    "category": "grocery_pos",
    "merchant_id": "m_1",
    "merchant_lat": 40.1,
    "merchant_long": -74.2,
    "home_lat": 40.3,
    "home_long": -74.0,
    "city_pop": 1000,
    "cardholder_age": 40,
}


def settings(bundle: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        api_key=KEY,
        card_hmac_key="h",
        database_url="postgresql+asyncpg://unused/db",
        redis_url="redis://unused/0",
        bundle_dir=bundle,
        degraded_review_amount=100.0,
    )


@pytest.fixture
def stores() -> tuple[MemoryStore, MemoryAudit]:
    return MemoryStore(), MemoryAudit()


@pytest.fixture
async def client(
    toy_bundle_dir: Path, stores: tuple[MemoryStore, MemoryAudit]
) -> AsyncIterator[httpx.AsyncClient]:
    @asynccontextmanager
    async def connect(_: Settings) -> AsyncIterator[tuple[MemoryStore, MemoryAudit]]:
        yield stores

    app = create_app(settings(toy_bundle_dir), connect=connect)  # type: ignore[arg-type]
    async with (
        LifespanManager(app) as manager,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://service",
            headers={"X-API-Key": KEY},
        ) as http,
    ):
        yield http


async def test_health_readiness_and_metrics(
    client: httpx.AsyncClient, stores: tuple[MemoryStore, MemoryAudit]
) -> None:
    store, audit = stores
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True

    await client.post("/v1/transactions/score", json=BODY)
    metrics = (await client.get("/metrics")).text
    assert "fraudlens_decisions_total" in metrics
    assert 'fraudlens_stage_seconds_count{stage="model"} 1.0' in metrics
    assert "fraudlens_model_info" in metrics

    store.down = True
    audit.down = True
    ready = await client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json() | {"model_version": None} == {
        "ready": False,
        "model_version": None,
        "postgres": False,
        "redis": False,
    }


async def test_the_api_key_is_required(client: httpx.AsyncClient) -> None:
    missing = await client.post("/v1/transactions/score", json=BODY, headers={"X-API-Key": ""})
    wrong = await client.get("/v1/model", headers={"X-API-Key": "nope"})

    assert missing.status_code == wrong.status_code == 401
    assert wrong.headers["WWW-Authenticate"] == "ApiKey"


async def test_scoring_replays_and_request_ids(
    client: httpx.AsyncClient, stores: tuple[MemoryStore, MemoryAudit]
) -> None:
    first = await client.post(
        "/v1/transactions/score", json=BODY, headers={"X-Request-ID": "req-1"}
    )
    second = await client.post("/v1/transactions/score", json=BODY)

    assert first.status_code == 200
    assert first.headers["X-Request-ID"] == "req-1"
    assert len(second.headers["X-Request-ID"]) == 32
    assert second.json() == first.json()
    assert second.headers["Idempotent-Replay"] == "true"
    assert (
        await client.post("/v1/transactions/score", json={**BODY, "amount": -1})
    ).status_code == 422


async def test_audit_outage_is_503_and_a_stuck_retry_is_409(
    client: httpx.AsyncClient,
    stores: tuple[MemoryStore, MemoryAudit],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, audit = stores
    audit.down = True
    down = await client.post("/v1/transactions/score", json=BODY)
    assert down.status_code == 503
    assert down.headers["Retry-After"] == "1"

    audit.down = False
    monkeypatch.setattr("fraudlens.service.decision.DUPLICATE_WAIT_S", 0.01)
    store.keys["tx0002"] = PENDING
    stuck = await client.post("/v1/transactions/score", json={**BODY, "transaction_id": "tx0002"})
    assert stuck.status_code == 409


async def test_stored_decision_and_explanation(
    client: httpx.AsyncClient, stores: tuple[MemoryStore, MemoryAudit]
) -> None:
    store, audit = stores
    store.down = True  # degraded, so the large amount is flagged and explained
    await client.post("/v1/transactions/score", json={**BODY, "amount": 900.0})
    store.down = False

    stored = (await client.get("/v1/transactions/tx0001")).json()
    explanation = (await client.get("/v1/transactions/tx0001/explanation")).json()
    cached = (await client.get("/v1/transactions/tx0001/explanation")).json()

    assert stored["features"]["n_prior_30d"] is None
    assert stored["decision"] in {"review", "block"}
    assert explanation == cached
    assert len(store.cache) == 1
    assert len(explanation["contributions"]) == 27
    contributions = [abs(c["contribution"]) for c in explanation["contributions"]]
    assert contributions == sorted(contributions, reverse=True)
    assert explanation["bias"] + sum(c["contribution"] for c in explanation["contributions"]) == (
        pytest.approx(explanation["margin"])
    )
    assert explanation["risk_score"] == stored["risk_score"]
    assert (await client.get("/v1/transactions/missing")).status_code == 404
    assert (await client.get("/v1/transactions/missing/explanation")).status_code == 404

    from dataclasses import replace

    audit.decisions["old"] = replace(audit.decisions["tx0001"], model_version="2020-01-01.1")
    assert (await client.get("/v1/transactions/old/explanation")).status_code == 409


async def test_alerts_dispositions_model_and_summary(
    client: httpx.AsyncClient, stores: tuple[MemoryStore, MemoryAudit]
) -> None:
    store, _ = stores
    store.down = True
    for i, amount in enumerate((200.0, 300.0, 400.0)):
        body = {**BODY, "transaction_id": f"tx{i}", "amount": amount}
        await client.post("/v1/transactions/score", json=body)

    page = (await client.get("/v1/alerts", params={"limit": 2})).json()
    rest = (await client.get("/v1/alerts", params={"cursor": page["next_cursor"]})).json()
    assert len(page["items"]) == 2
    assert len(rest["items"]) == 1
    assert (await client.get("/v1/alerts", params={"cursor": "%%%"})).status_code == 400

    alert_id = page["items"][0]["id"]
    labelled = await client.post(
        f"/v1/alerts/{alert_id}/disposition", json={"label": "genuine", "analyst": "a1"}
    )
    assert labelled.status_code == 201
    assert labelled.json()["label"] == "genuine"
    closed = (await client.get("/v1/alerts", params={"status": "closed"})).json()["items"]
    assert [a["id"] for a in closed] == [alert_id]
    missing = await client.post(
        "/v1/alerts/99/disposition", json={"label": "fraud", "analyst": "a"}
    )
    assert missing.status_code == 404

    model = (await client.get("/v1/model")).json()
    assert model["features"][0] == "amount"
    assert len(model["manifest_sha256"]) == 64
    summary = (await client.get("/v1/monitoring/summary")).json()
    assert summary["decisions"]["total"] == 3
    assert summary["decisions"]["degraded"] == 3
    assert summary["alert_rate"] == 1.0
    assert sum(summary["score_histogram"]["counts"]) == 3


def test_an_empty_summary_has_no_percentiles() -> None:
    summary = summarise([], model_version="v")

    assert summary["alert_rate"] is None
    assert summary["latency_ms"] == {"p50": None, "p95": None, "p99": None}


async def test_a_tampered_bundle_stops_start_up(toy_bundle_dir: Path, tmp_path: Path) -> None:
    tampered = tmp_path / toy_bundle_dir.name
    shutil.copytree(toy_bundle_dir, tampered)
    (tampered / "thresholds.json").write_text('{"t_review": 0.0, "t_block": 0.0}\n')

    @asynccontextmanager
    async def connect(_: Settings) -> AsyncIterator[tuple[MemoryStore, MemoryAudit]]:
        yield MemoryStore(), MemoryAudit()

    app = create_app(settings(tampered), connect=connect)  # type: ignore[arg-type]
    with pytest.raises(BundleIntegrityError, match=r"changed thresholds.json"):
        async with LifespanManager(app):
            pass


async def test_card_history_threshold_lab_and_labels(
    client: httpx.AsyncClient, stores: tuple[MemoryStore, MemoryAudit]
) -> None:
    store, _ = stores
    store.down = True
    for i in range(4):
        body = {
            **BODY,
            "transaction_id": f"h{i}",
            "event_time": f"2019-06-01T03:1{i}:00Z",
            "amount": 150.0 + i,
        }
        await client.post("/v1/transactions/score", json=body)

    history = (await client.get("/v1/transactions/h2/card-history")).json()
    assert [row["transaction_id"] for row in history] == ["h2", "h1", "h0"]
    assert (await client.get("/v1/transactions/zz/card-history")).status_code == 404
    assert (await client.get("/v1/transactions/h2/card-history?limit=0")).status_code == 422

    lab = (await client.get("/v1/model/threshold-curve")).json()
    points = lab["points"]
    assert points[lab["current_review"]]["threshold"] <= points[lab["current_block"]]["threshold"]
    assert len(lab["pairs"]) == len(points) * (len(points) + 1) // 2

    alerts = (await client.get("/v1/alerts")).json()["items"]
    assert all("top_reason" in a for a in alerts)
    await client.post(
        f"/v1/alerts/{alerts[0]['id']}/disposition", json={"label": "fraud", "analyst": "a"}
    )
    summary = (await client.get("/v1/monitoring/summary")).json()
    assert summary["dispositions"] == {"fraud": 1, "genuine": 0, "realised_precision": 1.0}


def test_a_summary_without_labels_has_no_precision() -> None:
    assert summarise([], model_version="v")["dispositions"] == {
        "fraud": 0,
        "genuine": 0,
        "realised_precision": None,
    }
