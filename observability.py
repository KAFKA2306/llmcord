from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import re
from typing import Any, Mapping


EVENT_LOGGER_NAME = "llmcord.events"

_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "bot_token",
    "content",
    "message",
    "message_content",
    "prompt",
    "request_body",
    "response_body",
    "response_text",
    "system_prompt",
    "attachment_body",
}
_REDACTED = "[REDACTED]"

_REQUEST_ID_RE = re.compile(r"\brequest_id=([^\s]+)")
_RECEIVED_RE = re.compile(
    r"request_id=(?P<request_id>\S+) Message received \(user ID: \d+, attachments: (?P<attachments>\d+), conversation length: (?P<conversation>\d+)\):(?:\n.*)?$",
    re.DOTALL,
)
_ADMITTED_RE = re.compile(r"request_id=(?P<request_id>\S+) inference admitted queue_wait_seconds=(?P<wait>[0-9.]+)")
_BACKEND_UNAVAILABLE_RE = re.compile(
    r"request_id=(?P<request_id>\S+) backend unavailable state=(?P<state>\S+) failures=(?P<failures>\S+)"
)
_GENERATION_TIMEOUT_RE = re.compile(r"request_id=(?P<request_id>\S+) Generation timeout phase=(?P<phase>\S+)")
_CONTEXT_COMPACTED_RE = re.compile(
    r"Context compacted \(input tokens: (?P<before>\d+) -> (?P<after>\d+), current input compacted: (?P<current>True|False)\)"
)
_WATCHDOG_FAILURE_RE = re.compile(
    r"backend health failure state=(?P<state>\S+) count=(?P<count>\d+) threshold=(?P<threshold>\d+) error=(?P<error>.*)"
)
_WATCHDOG_RESTART_RE = re.compile(r"backend recovery restart attempt=(?P<attempt>\d+)")
_WATCHDOG_RECOVERED_RE = re.compile(r"backend recovery succeeded attempt=(?P<attempt>\d+)")
_PROBE_RE = re.compile(r"watchdog synthetic probe healthy latency_seconds=(?P<latency>[0-9.]+)")


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


def _request_id_from(message: str) -> str | None:
    match = _REQUEST_ID_RE.search(message)
    return match.group(1) if match else None


def classify_message(message: str, *, level: str, logger_name: str) -> dict[str, Any]:
    if message.startswith("structured_event="):
        try:
            raw = json.loads(message.removeprefix("structured_event="))
            if not isinstance(raw, dict) or not raw.get("event"):
                raise ValueError("structured event must be an object with event")
            return sanitize_fields(raw)
        except (json.JSONDecodeError, ValueError):
            return build_event("observability.invalid_structured_event", logger=logger_name, level=level)

    if match := _RECEIVED_RE.fullmatch(message):
        return build_event(
            "request.received",
            request_id=match.group("request_id"),
            attachments=int(match.group("attachments")),
            conversation_messages=int(match.group("conversation")),
        )
    if match := _ADMITTED_RE.fullmatch(message):
        return build_event(
            "queue.admitted",
            request_id=match.group("request_id"),
            queue_wait_seconds=float(match.group("wait")),
        )
    if "inference queue full" in message:
        return build_event("queue.rejected", request_id=_request_id_from(message), reason="queue_full")
    if "inference queue wait timed out" in message:
        return build_event("queue.timeout", request_id=_request_id_from(message), reason="queue_wait")
    if match := _BACKEND_UNAVAILABLE_RE.fullmatch(message):
        failures = match.group("failures")
        return build_event(
            "request.rejected",
            request_id=match.group("request_id"),
            reason="backend_unavailable",
            backend_state=match.group("state"),
            consecutive_failures=int(failures) if failures.isdigit() else failures,
        )
    if match := _CONTEXT_COMPACTED_RE.fullmatch(message):
        return build_event(
            "context.compacted",
            input_tokens_before=int(match.group("before")),
            input_tokens_after=int(match.group("after")),
            current_input_compacted=match.group("current") == "True",
        )
    if match := _GENERATION_TIMEOUT_RE.fullmatch(message):
        return build_event(
            "generation.timeout",
            request_id=match.group("request_id"),
            phase=match.group("phase"),
        )
    if "Generation stream ended before a usable response" in message:
        return build_event("generation.failure", request_id=_request_id_from(message), error_class="protocol_error")
    if "Backend API error while generating response" in message:
        return build_event("generation.failure", request_id=_request_id_from(message), error_class="backend_api_error")
    if match := _WATCHDOG_FAILURE_RE.fullmatch(message):
        return build_event(
            "watchdog.failure",
            backend_state=match.group("state"),
            consecutive_failures=int(match.group("count")),
            failure_threshold=int(match.group("threshold")),
            error_class="backend_health_failure",
        )
    if match := _WATCHDOG_RESTART_RE.fullmatch(message):
        return build_event("watchdog.restart", attempt=int(match.group("attempt")))
    if match := _WATCHDOG_RECOVERED_RE.fullmatch(message):
        return build_event("watchdog.recovered", attempt=int(match.group("attempt")))
    if match := _PROBE_RE.fullmatch(message):
        return build_event("probe.success", latency_seconds=float(match.group("latency")))

    # Fail closed for observability privacy: retain routing metadata, never an arbitrary
    # unclassified log body. This prevents a new log statement from silently leaking prompts,
    # attachments, provider error bodies, or credentials.
    return build_event(
        "log",
        request_id=_request_id_from(message),
        level=level,
        logger=logger_name,
        classified=False,
    )


class StructuredJsonFormatter(logging.Formatter):
    def __init__(self, *, static_fields: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.static_fields = sanitize_fields(static_fields or {})

    def format(self, record: logging.LogRecord) -> str:
        try:
            payload = classify_message(record.getMessage(), level=record.levelname, logger_name=record.name)
            for key, value in self.static_fields.items():
                payload.setdefault(key, value)
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except Exception as exc:
            fallback = build_event(
                "observability.error",
                error_class=type(exc).__name__,
                logger=record.name,
                level=record.levelname,
            )
            return json.dumps(fallback, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def install_structured_logging(*, provider: str | None = None, model: str | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    static_fields = {key: value for key, value in {"provider": provider, "model": model}.items() if value is not None}
    handler.setFormatter(StructuredJsonFormatter(static_fields=static_fields))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def emit_event(event: str, *, request_id: str | None = None, **fields: Any) -> None:
    """Emit one event without allowing observability failures to break request handling."""

    try:
        payload = build_event(event, request_id=request_id, **fields)
        logging.getLogger(EVENT_LOGGER_NAME).info(
            "structured_event=%s",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        )
    except Exception as exc:
        logging.getLogger(__name__).error(
            "structured observability emission failed event=%s error_class=%s",
            event if isinstance(event, str) else "invalid",
            type(exc).__name__,
        )
