"""The replay client: payloads, pacing, drift injection, ordering and its command line."""

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from fraudlens.common.hashing import card_token
from fraudlens.tools import replay
from helpers import make_raw_sparkov

ROW: dict[str, Any] = {
    "transaction_id": "abc123",
    "card_id": "c_0123456789abcdef",
    "event_ts": 1_592_741_665,
    "amount": 12.5,
    "category": "grocery_pos",
    "merchant_id": "fraud_Shop 1",
    "merchant_lat": 36.1,
    "merchant_long": -81.2,
    "home_lat": math.nan,
    "home_long": -81.0,
    "city_pop": 3495,
    "cardholder_age": 30,
}


def test_payload_uses_the_contract_field_names() -> None:
    body = replay.to_payload(ROW)

    assert body["event_time"] == "2020-06-21T12:14:25Z"
    assert "home_lat" not in body  # missing values are left out
    assert body["city_pop"] == 3495
    assert isinstance(body["cardholder_age"], int)
    assert "is_fraud" not in body


def test_drift_is_parsed_and_validated() -> None:
    assert replay.parse_drift("shopping_net=1.5") == ("shopping_net", 1.5)
    for bad in ("shopping_net", "=2", "shopping_net=0", "shopping_net=x"):
        with pytest.raises(argparse.ArgumentTypeError):
            replay.parse_drift(bad)


def test_drift_hits_one_category_in_the_later_part_only() -> None:
    stream = pd.DataFrame({"category": ["a", "b", "a", "a"], "amount": [10.0, 10.0, 10.0, 3.33]})

    drifted = replay.inject_drift(stream, "a", 1.5, after=0.5)

    assert drifted["amount"].tolist() == [10.0, 10.0, 15.0, 5.0]
    assert stream["amount"].tolist() == [10.0, 10.0, 10.0, 3.33]


def test_send_offsets() -> None:
    ts = np.array([100, 160, 400])

    assert replay.send_offsets(ts, rate=2.0, speed=None).tolist() == [0.0, 0.5, 1.0]
    assert replay.send_offsets(ts, rate=None, speed=60.0).tolist() == [0.0, 1.0, 5.0]
    assert replay.send_offsets(ts, rate=None, speed=None).tolist() == [0.0, 0.0, 0.0]


def test_warm_histories_take_the_days_before_the_start() -> None:
    day = 86_400
    rows = pd.DataFrame(
        [
            {**ROW, "transaction_id": f"t{i}", "event_ts": ts}
            for i, ts in enumerate([0, day, 2 * day])
        ]
    )

    histories = replay.warm_histories(rows, before=2 * day, days=1)

    assert [e.transaction_id for e in histories[ROW["card_id"]]] == ["t1"]


def test_load_stream_tokenises_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = make_raw_sparkov(50, seed=3)
    path = tmp_path / "fraudTest.csv"
    raw.to_csv(path, index=False)
    monkeypatch.setattr(replay, "SPARKOV_ROWS", {"test": 50})

    stream = replay.load_stream(path, hmac_key=b"k", start=None, limit=20)

    assert len(stream) == 20
    assert "cc_num" not in stream.columns
    assert set(stream["card_id"]) <= {card_token(c, b"k") for c in raw["cc_num"]}
    assert stream["event_ts"].is_monotonic_increasing
    late = replay.load_stream(
        path, hmac_key=b"k", start=int(stream["event_ts"].iloc[5]), limit=None
    )
    assert late["event_ts"].min() >= stream["event_ts"].iloc[5]


def fake_service(log: list[tuple[str, str]]) -> httpx.MockTransport:
    async def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        log.append((body["card_id"], body["transaction_id"]))
        await asyncio.sleep(0.01 if body["transaction_id"].endswith("0") else 0)
        if body["transaction_id"] == "bad":
            return httpx.Response(503)
        if body["transaction_id"] == "boom":
            raise httpx.ConnectError("refused")
        headers = {"Idempotent-Replay": "true"} if body["transaction_id"] == "again" else {}
        return httpx.Response(200, json={"decision": "approve", "degraded": True}, headers=headers)

    return httpx.MockTransport(handle)


@pytest.mark.anyio
async def test_replay_keeps_each_cards_order_and_counts_outcomes() -> None:
    ids = ["a0", "a1", "a2", "b0", "b1", "bad", "again", "boom"]
    cards = ["c_a", "c_a", "c_a", "c_b", "c_b", "c_b", "c_b", "c_b"]
    pairs = zip(ids, cards, strict=True)
    stream = pd.DataFrame([{**ROW, "transaction_id": t, "card_id": c} for t, c in pairs])
    log: list[tuple[str, str]] = []

    async with httpx.AsyncClient(transport=fake_service(log), base_url="http://s") as client:
        stats = await replay.replay(client, stream, [0.0] * len(stream), concurrency=4)

    for card in ("c_a", "c_b"):
        sent = [t for c, t in log if c == card]
        assert sent == [t for t, c in zip(ids, cards, strict=True) if c == card]
    summary = stats.summary()
    assert summary["sent"] == 8
    assert summary["statuses"] == {"0": 1, "200": 6, "503": 1}
    assert summary["degraded"] == 6
    assert summary["replayed"] == 1
    assert summary["client_latency_ms"]["p50"] >= 0


def test_main_replays_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in {
        "FRAUDLENS_API_KEY": "k",
        "FRAUDLENS_CARD_HMAC_KEY": "h",
        "FRAUDLENS_DATABASE_URL": "postgresql+asyncpg://u:p@h/db",
        "FRAUDLENS_REDIS_URL": "redis://h/0",
    }.items():
        monkeypatch.setenv(name, value)
    replay.get_settings.cache_clear()
    stream = pd.DataFrame(
        [{**ROW, "transaction_id": f"t{i}", "event_ts": ROW["event_ts"] + i} for i in range(3)]
    )
    monkeypatch.setattr(replay, "load_stream", lambda *a, **k: stream)
    monkeypatch.setattr(replay, "configure_logging", lambda *_: None)
    seeded: list[int] = []

    async def fake_seed(url: str, histories: dict[str, Any]) -> int:
        seeded.append(len(histories))
        return len(histories)

    monkeypatch.setattr(replay, "_seed", fake_seed)
    monkeypatch.setattr(
        replay.pd,
        "read_parquet",
        lambda *_: pd.DataFrame([{**ROW, "event_ts": ROW["event_ts"] - 10}]),
    )
    log: list[tuple[str, str]] = []
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        replay.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=fake_service(log), **kw),
    )

    code = replay.main(["--rate", "1000", "--warm-days", "1", "--inject-drift", "grocery_pos=2"])

    replay.get_settings.cache_clear()
    assert code == 0
    assert seeded == [1]
    assert json.loads(capsys.readouterr().out)["sent"] == 3
    assert len(log) == 3


def test_warm_only_seeds_history_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, value in {
        "FRAUDLENS_API_KEY": "k",
        "FRAUDLENS_CARD_HMAC_KEY": "h",
        "FRAUDLENS_DATABASE_URL": "postgresql+asyncpg://u:p@h/db",
        "FRAUDLENS_REDIS_URL": "redis://h/0",
    }.items():
        monkeypatch.setenv(name, value)
    replay.get_settings.cache_clear()
    stream = pd.DataFrame([{**ROW, "transaction_id": "t0"}])
    monkeypatch.setattr(replay, "load_stream", lambda *a, **k: stream)
    monkeypatch.setattr(replay, "configure_logging", lambda *_: None)
    monkeypatch.setattr(
        replay.pd,
        "read_parquet",
        lambda *_: pd.DataFrame([{**ROW, "event_ts": ROW["event_ts"] - 10}]),
    )
    seeded: list[int] = []

    async def fake_seed(url: str, histories: dict[str, Any]) -> int:
        seeded.append(len(histories))
        return len(histories)

    monkeypatch.setattr(replay, "_seed", fake_seed)
    monkeypatch.setattr(replay, "replay", None)  # any attempt to send would fail

    code = replay.main(["--warm-days", "1", "--warm-only"])

    replay.get_settings.cache_clear()
    assert code == 0
    assert seeded == [1]
    assert "warmed" in capsys.readouterr().out


def test_warm_only_needs_warm_days() -> None:
    with pytest.raises(SystemExit):
        replay.main(["--warm-only"])
