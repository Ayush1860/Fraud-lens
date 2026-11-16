"""Logs are one JSON object per line, carry the request id and never a full card token."""

import io
import json
import logging
from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest

from fraudlens.common.logging import (
    JsonFormatter,
    configure_logging,
    mask_card_id,
    request_id_var,
)


@pytest.fixture
def capture() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    """A private logger writing JSON lines into a string buffer."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("fraudlens.test")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield logger, stream
    logger.handlers = []
    logger.propagate = True


def last_line(stream: io.StringIO) -> dict[str, object]:
    payload = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert isinstance(payload, dict)
    return payload


def test_emits_one_json_object_with_core_fields(
    capture: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = capture

    logger.info("scored %s", "txn-1")

    record = last_line(stream)
    assert record["message"] == "scored txn-1"
    assert record["level"] == "INFO"
    assert record["logger"] == "fraudlens.test"
    timestamp = datetime.fromisoformat(str(record["timestamp"]))
    assert timestamp.utcoffset() == timedelta(0)


def test_includes_request_id_only_inside_a_request(
    capture: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = capture

    logger.info("outside")
    assert "request_id" not in last_line(stream)

    token = request_id_var.set("req-42")
    try:
        logger.info("inside")
    finally:
        request_id_var.reset(token)
    assert last_line(stream)["request_id"] == "req-42"


def test_extra_fields_become_top_level_keys(
    capture: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = capture

    logger.info("decision", extra={"decision": "review", "latency_ms": 14})

    record = last_line(stream)
    assert record["decision"] == "review"
    assert record["latency_ms"] == 14


def test_extra_fields_cannot_overwrite_core_fields(
    capture: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = capture

    logger.info("real message", extra={"level": "FAKE", "timestamp": "never"})

    record = last_line(stream)
    assert record["level"] == "INFO"
    assert record["timestamp"] != "never"


def test_unserialisable_extras_are_stringified(
    capture: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = capture

    logger.info("odd", extra={"when": datetime(2020, 8, 14, 3, 12, 44)})

    assert last_line(stream)["when"] == "2020-08-14 03:12:44"


def test_exceptions_are_rendered_into_the_record(
    capture: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, stream = capture

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")

    record = last_line(stream)
    assert "ValueError: boom" in str(record["exception"])
    assert len(stream.getvalue().strip().splitlines()) == 1


def test_configure_logging_installs_exactly_one_json_handler() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("WARNING")
        configure_logging("DEBUG")

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.DEBUG
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


@pytest.mark.parametrize(
    ("card_id", "expected"),
    [
        ("c_9f2e61b07a", "c_9f2e***"),
        ("abc", "abc***"),
        ("", "***"),
    ],
)
def test_mask_card_id_keeps_at_most_six_characters(card_id: str, expected: str) -> None:
    assert mask_card_id(card_id) == expected
