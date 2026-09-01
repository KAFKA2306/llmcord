from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from typing import Any, Mapping


EVENT_LOGGER_NAME = "llmcord.events"

# These fields must never be emitted verbatim. Keep this list explicit so metric names such
# as input_tokens/output_tokens remain valid while secrets and user-provided bodies are redacted.
_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "bot_token",
    "content",
    "message_content",
    "prompt",
    "request_body",
    "response_body",
    "response_text",
    "system_prompt",
    "attachment_body",
}
_REDACTED = "[REDACTED]"


def _event_logger() -> logging.Logger:
    logger = logging.getLogger(EVENT_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _sanitize_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def sanitize_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        key_text = str(key)
        if key_text.lower() in _SENSITIVE_KEYS:
            sanitized[key_text] = _REDACTED
        elif isinstance(value, Mapping):
            sanitized[key_text] = sanitize_fields(value)
        elif isinstance(value, (list, tuple)):
            sanitized[key_text] = [_sanitize_scalar(item) for item in value]
        else:
            sanitized[key_text] = _sanitize_scalar(value)
    return sanitized


def build_event(
    event: str,
    *,
    request_id: str | None = None,
    timestamp: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    if not isinstance(event, str) or not event.strip():
        raise ValueError("event must be a non-empty string")
    if request_id is not None and (not isinstance(request_id, str) or not request_id):
        raise ValueError("request_id must be a non-empty string when set")

    payload: dict[str, Any] = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    payload.update(sanitize_fields(fields))
    return payload


def emit_event(event: str, *, request_id: str | None = None, **fields: Any) -> None:
    """Emit one JSON object without allowing observability failures to break request handling."""

    try:
        payload = build_event(event, request_id=request_id, **fields)
        _event_logger().info(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    except Exception as exc:  # observability must not become a new availability dependency
        logging.getLogger(__name__).error(
            "structured observability emission failed event=%s error_class=%s",
            event if isinstance(event, str) else "invalid",
            type(exc).__name__,
        )
