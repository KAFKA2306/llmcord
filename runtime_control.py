from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import inspect
from typing import Any, AsyncIterator, Awaitable, Callable


class RuntimeControlError(RuntimeError):
    """Base error for deterministic runtime admission and timeout controls."""


class InferenceQueueFull(RuntimeControlError):
    """Raised when accepting another request would exceed the bounded queue."""


class InferenceQueueTimeout(RuntimeControlError):
    """Raised when an admitted request waits too long for an inference slot."""


class InferenceUnavailable(RuntimeControlError):
    """Raised when health control has closed admission to the inference backend."""


class GenerationTimeout(RuntimeControlError):
    def __init__(self, phase: str) -> None:
        super().__init__(f"generation timed out during {phase}")
        self.phase = phase


class GenerationProtocolError(RuntimeControlError):
    """Raised when a streaming backend ends before producing a generation signal."""


@dataclass(frozen=True)
class RuntimePolicy:
    max_concurrency: int
    max_queue_size: int
    queue_wait_timeout_seconds: float
    connect_timeout_seconds: float
    first_token_timeout_seconds: float
    total_generation_timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.max_concurrency, int) or isinstance(self.max_concurrency, bool) or self.max_concurrency <= 0:
            raise RuntimeControlError("max_concurrency must be a positive integer")
        if not isinstance(self.max_queue_size, int) or isinstance(self.max_queue_size, bool) or self.max_queue_size < 0:
            raise RuntimeControlError("max_queue_size must be a non-negative integer")

        positive_seconds = {
            "queue_wait_timeout_seconds": self.queue_wait_timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "first_token_timeout_seconds": self.first_token_timeout_seconds,
            "total_generation_timeout_seconds": self.total_generation_timeout_seconds,
        }
        for name, value in positive_seconds.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise RuntimeControlError(f"{name} must be a positive number")

        if self.first_token_timeout_seconds > self.total_generation_timeout_seconds:
            raise RuntimeControlError("first_token_timeout_seconds must not exceed total_generation_timeout_seconds")


@dataclass
class InferenceLease:
    gate: "InferenceGate"
    queue_wait_seconds: float
    _released: bool = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self.gate.release()

    async def __aenter__(self) -> "InferenceLease":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.release()


class InferenceGate:
    """Bound accepted work and expose one health-controlled admission switch."""

    def __init__(self, policy: RuntimePolicy) -> None:
        self.policy = policy
        self._semaphore = asyncio.Semaphore(policy.max_concurrency)
        self._state_lock = asyncio.Lock()
        self._in_system = 0
        self._accepting = True

    async def set_accepting(self, accepting: bool) -> None:
        if not isinstance(accepting, bool):
            raise RuntimeControlError("accepting must be a boolean")
        async with self._state_lock:
            self._accepting = accepting

    async def acquire(self) -> InferenceLease:
        async with self._state_lock:
            if not self._accepting:
                raise InferenceUnavailable("inference admission is closed by backend health control")
            capacity = self.policy.max_concurrency + self.policy.max_queue_size
            if self._in_system >= capacity:
                raise InferenceQueueFull(f"inference capacity {capacity} is full")
            self._in_system += 1

        started = asyncio.get_running_loop().time()
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.policy.queue_wait_timeout_seconds,
            )
        except TimeoutError as exc:
            async with self._state_lock:
                self._in_system -= 1
            raise InferenceQueueTimeout("timed out waiting for an inference slot") from exc
        except BaseException:
            async with self._state_lock:
                self._in_system -= 1
            raise

        async with self._state_lock:
            if not self._accepting:
                self._in_system -= 1
                self._semaphore.release()
                raise InferenceUnavailable("inference admission closed while this request was queued")

        return InferenceLease(
            gate=self,
            queue_wait_seconds=asyncio.get_running_loop().time() - started,
        )

    async def release(self) -> None:
        self._semaphore.release()
        async with self._state_lock:
            self._in_system -= 1
            if self._in_system < 0:
                self._in_system = 0
                raise RuntimeControlError("inference gate release imbalance")

    async def snapshot(self) -> dict[str, int | bool]:
        async with self._state_lock:
            active_or_waiting = self._in_system
            accepting = self._accepting
        return {
            "in_system": active_or_waiting,
            "capacity": self.policy.max_concurrency + self.policy.max_queue_size,
            "max_concurrency": self.policy.max_concurrency,
            "max_queue_size": self.policy.max_queue_size,
            "accepting": accepting,
        }


async def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
    if close is None:
        return
    with suppress(Exception):
        result = close()
        if inspect.isawaitable(result):
            await result


async def stream_with_timeouts(
    stream_factory: Callable[[], Awaitable[Any]],
    *,
    first_signal: Callable[[Any], bool],
    first_token_timeout_seconds: float,
    total_generation_timeout_seconds: float,
) -> AsyncIterator[Any]:
    """Yield a stream with separate first-generation-signal and total deadlines.

    Empty protocol chunks before the first real generation signal are intentionally discarded.
    The stream is closed on completion, timeout, cancellation, or error.
    """

    stream = None
    first_seen = False
    try:
        try:
            async with asyncio.timeout(total_generation_timeout_seconds):
                try:
                    async with asyncio.timeout(first_token_timeout_seconds):
                        stream = await stream_factory()
                        iterator = stream.__aiter__()
                        while True:
                            item = await anext(iterator)
                            if first_signal(item):
                                first_seen = True
                                break
                except StopAsyncIteration as exc:
                    raise GenerationProtocolError("stream ended before the first generation signal") from exc
                except TimeoutError as exc:
                    raise GenerationTimeout("first_token") from exc

                yield item
                async for item in iterator:
                    yield item
        except TimeoutError as exc:
            phase = "total" if first_seen else "first_token"
            raise GenerationTimeout(phase) from exc
    finally:
        if stream is not None:
            await _close_stream(stream)
