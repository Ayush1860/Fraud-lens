"""Structured JSON logging with a per-request id.

Every record is one JSON object on one line. Card tokens must pass through
:func:`mask_card_id` before they are logged; raw card numbers, names and addresses
are never logged at all.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
"""Id of the request being handled, or ``None`` outside a request."""

_CARD_ID_VISIBLE_CHARS = 6

# Attributes every LogRecord has; anything else on a record came from ``extra=``.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Format a log record as a single-line JSON object.

    Core keys are ``timestamp`` (UTC, ISO 8601), ``level``, ``logger`` and ``message``,
    plus ``request_id`` inside a request and ``exception`` when one is attached.
    Fields passed through ``extra=`` become top-level keys but cannot replace core keys.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as JSON. Values JSON cannot encode are passed through ``str``."""
        payload: dict[str, Any] = {
            key: value for key, value in vars(record).items() if key not in _STANDARD_ATTRS
        }
        payload.update(
            timestamp=datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
        )
        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Send all logs to stdout as JSON at ``level``, replacing any existing handlers.

    Safe to call more than once; the root logger always ends up with exactly one handler.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def mask_card_id(card_id: str) -> str:
    """Return a loggable form of a card token: its first six characters, then ``***``."""
    return f"{card_id[:_CARD_ID_VISIBLE_CHARS]}***"
