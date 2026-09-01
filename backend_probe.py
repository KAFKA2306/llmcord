from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any

import httpx


@dataclass(frozen=True)
class BackendProbeConfig:
    base_url: str
    model: str
    timeout_seconds: float = 15.0
    expected_text: str = "PONG"
    max_tokens: int = 8
    prompt: str = "Reply exactly: PONG"
    headers: dict[str, str] = field(default_factory=dict)
    query: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not self.expected_text.strip():
            raise ValueError("expected_text must not be empty")


@dataclass(frozen=True)
class BackendProbeResult:
    healthy: bool
    elapsed_seconds: float
    error_class: str | None = None
    detail: str | None = None
    response_text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response does not contain choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("first choice must be an object")

    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("first choice does not contain a message")

    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("message content must be a string")

    return content.strip()


def _chat_completions_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    return f"{root}/chat/completions" if root.endswith("/v1") else f"{root}/v1/chat/completions"


async def probe_generation(
    config: BackendProbeConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> BackendProbeResult:
    """Run a real OpenAI-compatible generation request as the health authority.

    Process liveness, an open TCP port, `/health`, and `/v1/models` are useful
    diagnostics but are intentionally not sufficient to return healthy=True.
    """

    started = time.monotonic()
    timeout = httpx.Timeout(config.timeout_seconds)

    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            response = await client.post(
                _chat_completions_url(config.base_url),
                headers=config.headers,
                params=config.query,
                json={
                    "model": config.model,
                    "messages": [
                        {"role": "user", "content": config.prompt}
                    ],
                    "max_tokens": config.max_tokens,
                    "temperature": 0,
                    "stream": False,
                },
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                return BackendProbeResult(
                    healthy=False,
                    elapsed_seconds=time.monotonic() - started,
                    error_class="invalid_json",
                    detail=str(exc),
                )

            try:
                text = _extract_text(payload)
            except ValueError as exc:
                return BackendProbeResult(
                    healthy=False,
                    elapsed_seconds=time.monotonic() - started,
                    error_class="protocol_error",
                    detail=str(exc),
                )

            expected = config.expected_text.strip()
            if text != expected:
                return BackendProbeResult(
                    healthy=False,
                    elapsed_seconds=time.monotonic() - started,
                    error_class="unexpected_output",
                    detail=f"expected {expected!r}, got {text!r}",
                    response_text=text,
                )

            return BackendProbeResult(
                healthy=True,
                elapsed_seconds=time.monotonic() - started,
                response_text=text,
            )

    except httpx.TimeoutException as exc:
        return BackendProbeResult(
            healthy=False,
            elapsed_seconds=time.monotonic() - started,
            error_class="timeout",
            detail=str(exc),
        )
    except httpx.HTTPStatusError as exc:
        return BackendProbeResult(
            healthy=False,
            elapsed_seconds=time.monotonic() - started,
            error_class="http_status",
            detail=f"HTTP {exc.response.status_code}",
        )
    except httpx.HTTPError as exc:
        return BackendProbeResult(
            healthy=False,
            elapsed_seconds=time.monotonic() - started,
            error_class="transport_error",
            detail=str(exc),
        )
