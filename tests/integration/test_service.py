"""The scoring service's failure modes and guarantees, against real Redis and Postgres."""

import asyncio
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import redis
import sqlalchemy as sa

from fraudlens.tools.replay import to_payload
from integration_support import DEAD_DATABASE_URL, DEAD_REDIS_URL, ClientFactory
from toy_bundle import toy_transactions

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture(scope="module")
def txns() -> pd.DataFrame:
    return toy_transactions(n_txns=3_000, seed=11)


def payload(txns: pd.DataFrame, i: int, **override: Any) -> dict[str, Any]:
    body = to_payload(txns.iloc[i].to_dict())
    body.update(override)
    return body


async def score(client: httpx.AsyncClient, body: dict[str, Any]) -> httpx.Response:
    return await client.post("/v1/transactions/score", json=body)


def count(db: sa.Engine, table: str) -> int:
    with db.connect() as conn:
        return int(conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one())  # noqa: S608


def history_length(redis_db: redis.Redis, card_id: str) -> int:
    return int(redis_db.llen(f"hist:{card_id}"))


async def test_same_transaction_twice_returns_the_original_decision(
    make_client: ClientFactory, txns: pd.DataFrame, db: sa.Engine, redis_db: redis.Redis
) -> None:
    body = payload(txns, 0)
    async with make_client() as client:
        first = await score(client, body)
        second = await score(client, body)

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()
    assert second.headers["Idempotent-Replay"] == "true"
    assert "Idempotent-Replay" not in first.headers
    assert count(db, "decisions") == 1
    assert history_length(redis_db, body["card_id"]) == 1


async def test_replay_after_the_redis_key_expires_is_caught_by_postgres(
    make_client: ClientFactory, txns: pd.DataFrame, db: sa.Engine, redis_db: redis.Redis
) -> None:
    body = payload(txns, 1)
    async with make_client() as client:
        first = await score(client, body)
        redis_db.delete(f"idem:{body['transaction_id']}")
        second = await score(client, body)

    assert second.json() == first.json()
    assert second.headers["Idempotent-Replay"] == "true"
    assert count(db, "decisions") == 1
    assert history_length(redis_db, body["card_id"]) == 1


async def test_a_replayed_stream_does_not_double_count_velocity(
    make_client: ClientFactory, txns: pd.DataFrame, redis_db: redis.Redis
) -> None:
    card = txns["card_id"].value_counts().index[0]
    rows = txns.index[txns["card_id"] == card][:40]
    async with make_client() as client:
        first = [(await score(client, payload(txns, i))).json() for i in rows]
        stored = redis_db.lrange(f"hist:{card}", 0, -1)
        second = [(await score(client, payload(txns, i))).json() for i in rows]

    assert second == first
    assert redis_db.lrange(f"hist:{card}", 0, -1) == stored
    assert 0 < len(stored) <= len(rows)


async def test_redis_outage_degrades_instead_of_failing(
    make_client: ClientFactory, txns: pd.DataFrame, db: sa.Engine
) -> None:
    small = payload(txns, 2, amount=12.5)
    large = payload(txns, 3, amount=4_000.0)
    async with make_client(redis=DEAD_REDIS_URL, degraded_review_amount=1_000.0) as client:
        small_response = await score(client, small)
        large_response = await score(client, large)
        ready = await client.get("/readyz")

    assert small_response.status_code == large_response.status_code == 200
    assert small_response.json()["degraded"] is True
    assert large_response.json()["degraded"] is True
    assert large_response.json()["decision"] in {"review", "block"}
    assert ready.status_code == 200
    assert ready.json()["redis"] is False
    assert count(db, "decisions") == 2


async def test_degraded_requests_are_still_idempotent(
    make_client: ClientFactory, txns: pd.DataFrame, db: sa.Engine
) -> None:
    body = payload(txns, 4)
    async with make_client(redis=DEAD_REDIS_URL) as client:
        first = await score(client, body)
        second = await score(client, body)

    assert second.json() == first.json()
    assert second.headers["Idempotent-Replay"] == "true"
    assert count(db, "decisions") == 1


async def test_postgres_outage_fails_closed_and_releases_the_claim(
    make_client: ClientFactory, txns: pd.DataFrame, db: sa.Engine, redis_db: redis.Redis
) -> None:
    body = payload(txns, 5)
    async with make_client(database=DEAD_DATABASE_URL, database_timeout_ms=500) as client:
        failed = await score(client, body)
        ready = await client.get("/readyz")

    assert failed.status_code == 503
    assert failed.headers["Retry-After"] == "1"
    assert ready.status_code == 503
    assert history_length(redis_db, body["card_id"]) == 0

    async with make_client() as client:
        retried = await score(client, body)
    assert retried.status_code == 200
    assert "Idempotent-Replay" not in retried.headers
    assert count(db, "decisions") == 1


async def test_concurrent_requests_on_one_card_are_serialised(
    make_client: ClientFactory, txns: pd.DataFrame, db: sa.Engine, redis_db: redis.Redis
) -> None:
    template = payload(txns, 6)
    # Thirty transactions in the same second on one card. Whatever order they are
    # processed in, each must see exactly the ones processed before it that sort
    # earlier by transaction id. A lost or interleaved history update breaks that.
    bodies = [{**template, "transaction_id": f"concurrent{i:02d}"} for i in range(30)]
    async with make_client(card_lock_timeout_ms=5_000) as client:
        responses = await asyncio.gather(*(score(client, body) for body in bodies))

    assert all(response.status_code == 200 for response in responses)
    assert not any(response.json()["degraded"] for response in responses)
    with db.connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT transaction_id, (features->>'n_prior_30d')::int FROM decisions "
                "ORDER BY created_at"
            )
        ).all()
    processed: list[str] = []
    for transaction_id, n_prior in rows:
        assert n_prior == sum(earlier < transaction_id for earlier in processed)
        processed.append(transaction_id)
    assert history_length(redis_db, template["card_id"]) == 30


async def test_the_audit_tables_are_append_only(
    make_client: ClientFactory, txns: pd.DataFrame, db: sa.Engine
) -> None:
    async with make_client() as client:
        await score(client, payload(txns, 7))

    for statement in (
        "UPDATE decisions SET score = 0",
        "DELETE FROM decisions",
        "UPDATE model_versions SET manifest_hash = 'x'",
    ):
        with pytest.raises(sa.exc.DBAPIError, match="append-only"), db.begin() as conn:
            conn.execute(sa.text(statement))


async def test_alert_queue_disposition_and_explanation(
    make_client: ClientFactory, txns: pd.DataFrame, db: sa.Engine
) -> None:
    # Without Redis every amount above the limit is sent to review, so these three are
    # flagged whatever the toy model thinks of them.
    async with make_client(redis=DEAD_REDIS_URL, degraded_review_amount=100.0) as client:
        flagged: dict[str, dict[str, Any]] = {}
        for i, amount in enumerate((2_500.0, 3_000.0, 3_500.0)):
            body = payload(txns, 20 + i, amount=amount, transaction_id=f"flagged{i}")
            response = (await score(client, body)).json()
            assert response["decision"] in {"review", "block"}, response
            flagged[body["transaction_id"]] = response

        page = (await client.get("/v1/alerts", params={"limit": 2})).json()
        rest = (
            await client.get("/v1/alerts", params={"limit": 2, "cursor": page["next_cursor"]})
        ).json()
        alerts = page["items"] + rest["items"]
        assert {a["transaction_id"] for a in alerts} == set(flagged)
        assert [a["priority"] for a in alerts] == sorted(
            (a["risk_score"] * a["amount"] for a in alerts), reverse=True
        )
        assert rest["next_cursor"] is None
        bad_cursor = await client.get("/v1/alerts", params={"cursor": "not-a-cursor"})
        assert bad_cursor.status_code == 400

        alert_id = alerts[0]["id"]
        disposition = await client.post(
            f"/v1/alerts/{alert_id}/disposition",
            json={"label": "fraud", "analyst": "analyst-1", "note": "confirmed with cardholder"},
        )
        assert disposition.status_code == 201
        open_alerts = (await client.get("/v1/alerts", params={"status": "open"})).json()
        assert alert_id not in [a["id"] for a in open_alerts["items"]]
        assert len(open_alerts["items"]) == 2
        missing = await client.post(
            "/v1/alerts/999999/disposition", json={"label": "genuine", "analyst": "a"}
        )
        assert missing.status_code == 404

        top = alerts[0]["transaction_id"]
        stored = (await client.get(f"/v1/transactions/{top}")).json()
        explanation = (await client.get(f"/v1/transactions/{top}/explanation")).json()
        again = (await client.get(f"/v1/transactions/{top}/explanation")).json()
        summary = (await client.get("/v1/monitoring/summary")).json()
        unknown = await client.get("/v1/transactions/never-scored")

    assert stored["risk_score"] == flagged[top]["risk_score"]
    assert stored["degraded"] is True
    assert stored["features"]["txn_count_24h"] is None
    assert explanation == again
    assert [r["id"] for r in explanation["reason_codes"]] == [
        r["id"] for r in flagged[top]["reason_codes"]
    ]
    assert explanation["margin"] == pytest.approx(stored["margin"], abs=1e-4)
    assert explanation["risk_score"] == stored["risk_score"]
    assert summary["decisions"]["total"] == 3
    assert summary["decisions"]["degraded"] == 3
    assert unknown.status_code == 404
    assert count(db, "dispositions") == 1


async def test_card_history_top_reason_and_realised_precision(
    make_client: ClientFactory, txns: pd.DataFrame
) -> None:
    template = payload(txns, 30)
    base = int(pd.Timestamp(template["event_time"]).timestamp())
    async with make_client(redis=DEAD_REDIS_URL, degraded_review_amount=100.0) as client:
        for i in range(25):
            body = {
                **template,
                "transaction_id": f"card{i:02d}",
                "event_time": pd.Timestamp(base + 60 * i, unit="s", tz="UTC").isoformat(),
                "amount": 500.0 + i,
            }
            assert (await score(client, body)).status_code == 200
        other = {**payload(txns, 31), "transaction_id": "othercard", "amount": 900.0}
        await score(client, other)

        history = (await client.get("/v1/transactions/card10/card-history")).json()
        short = (
            await client.get("/v1/transactions/card10/card-history", params={"limit": 3})
        ).json()
        missing = await client.get("/v1/transactions/nothing/card-history")

        alerts = (await client.get("/v1/alerts", params={"limit": 200})).json()["items"]
        by_id = {a["transaction_id"]: a for a in alerts}
        for transaction_id, label in (
            ("card00", "fraud"),
            ("card01", "fraud"),
            ("card02", "genuine"),
        ):
            alert_id = by_id[transaction_id]["id"]
            await client.post(
                f"/v1/alerts/{alert_id}/disposition", json={"label": "genuine", "analyst": "a"}
            )
            # The latest label on an alert is the one that counts.
            await client.post(
                f"/v1/alerts/{alert_id}/disposition", json={"label": label, "analyst": "a"}
            )
        summary = (await client.get("/v1/monitoring/summary")).json()

    assert [t["transaction_id"] for t in history] == [f"card{i:02d}" for i in range(10, -1, -1)]
    assert [t["transaction_id"] for t in short] == ["card10", "card09", "card08"]
    assert history[0]["amount"] == 510.0
    assert missing.status_code == 404
    assert all(a["top_reason"] is None or isinstance(a["top_reason"], str) for a in alerts)
    assert any(a["top_reason"] for a in alerts)
    assert summary["dispositions"] == {"fraud": 2, "genuine": 1, "realised_precision": 2 / 3}


async def test_injected_drift_raises_a_visible_alarm(
    make_client: ClientFactory, txns: pd.DataFrame
) -> None:
    # Two categories so each has enough rows per window; amounts in one are tripled in
    # the newest window, as `replay --inject-drift` does.
    rows = txns.iloc[:1_200].reset_index(drop=True).copy()
    rows["category"] = np.where(np.arange(len(rows)) % 2 == 0, "shopping_net", "grocery_pos")
    late = (np.arange(len(rows)) >= 600) & (rows["category"] == "shopping_net").to_numpy()
    rows.loc[late, "amount"] = (rows.loc[late, "amount"] * 3).round(2)

    async with make_client(drift_window=600, drift_interval_s=0.5) as client:
        for i in range(len(rows)):
            body = to_payload(rows.iloc[i].to_dict())
            assert (await score(client, body)).status_code == 200
        # The background job recomputes every half second; wait for a report that covers
        # both full windows.
        drift: dict[str, Any] = {}
        for _ in range(60):
            drift = (await client.get("/v1/monitoring/summary")).json()["drift"] or {}
            if drift.get("previous_window") == 600:
                break
            await asyncio.sleep(0.5)
        assert drift.get("previous_window") == 600, drift

    names = [alarm["name"] for alarm in drift["alarms"]]
    assert "amount in shopping_net" in names
    segments = {s["category"]: s for s in drift["segments"]}
    assert segments["shopping_net"]["status"] == "alarm"
    assert segments["grocery_pos"]["status"] in {"stable", "watch"}
