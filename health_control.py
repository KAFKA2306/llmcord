from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
import logging
from typing import Awaitable, Callable, Optional


Probe = Callable[[], Awaitable[None]]
RestartAction = Callable[[], Awaitable[None]]
GpuCheck = Callable[[], Awaitable[None]]
IdleCheck = Callable[[], Awaitable[bool]]


class HealthControlError(RuntimeError):
    """Raised when backend health cannot be established safely."""


class BackendState(StrEnum):
    STARTING = "starting"
    HEALTHY = "healthy"
    SUSPECT = "suspect"
    RECOVERING = "recovering"
    DEGRADED = "degraded"
    STOPPED = "stopped"


@dataclass(frozen=True)
class WatchdogPolicy:
    probe_interval_seconds: float
    failure_threshold: int
    restart_cooldown_seconds: float
    post_restart_grace_seconds: float
    max_restart_attempts: int

    def __post_init__(self) -> None:
        positive_seconds = {
            "probe_interval_seconds": self.probe_interval_seconds,
            "restart_cooldown_seconds": self.restart_cooldown_seconds,
            "post_restart_grace_seconds": self.post_restart_grace_seconds,
        }
        for name, value in positive_seconds.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise HealthControlError(f"{name} must be a positive number")
        for name, value in {
            "failure_threshold": self.failure_threshold,
            "max_restart_attempts": self.max_restart_attempts,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise HealthControlError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class WatchdogSnapshot:
    state: BackendState
    accepting: bool
    consecutive_failures: int
    restart_attempts: int
    last_error: Optional[str]


class BackendWatchdog:
    """Deterministic health state machine for one production inference backend.

    Health is established by a real generation probe plus the optional GPU check.
    A shallow HTTP health endpoint is intentionally not authoritative.
    """

    def __init__(
        self,
        policy: WatchdogPolicy,
        *,
        probe: Probe,
        idle: IdleCheck,
        restart: Optional[RestartAction] = None,
        gpu_check: Optional[GpuCheck] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.policy = policy
        self._probe = probe
        self._idle = idle
        self._restart = restart
        self._gpu_check = gpu_check
        self._logger = logger or logging.getLogger(__name__)

        self._state = BackendState.STARTING
        self._consecutive_failures = 0
        self._restart_attempts = 0
        self._last_error: Optional[str] = None
        self._state_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()

    async def snapshot(self) -> WatchdogSnapshot:
        async with self._state_lock:
            state = self._state
            return WatchdogSnapshot(
                state=state,
                accepting=state in {BackendState.HEALTHY, BackendState.SUSPECT},
                consecutive_failures=self._consecutive_failures,
                restart_attempts=self._restart_attempts,
                last_error=self._last_error,
            )

    async def is_accepting(self) -> bool:
        return (await self.snapshot()).accepting

    async def report_success(self) -> None:
        async with self._state_lock:
            self._state = BackendState.HEALTHY
            self._consecutive_failures = 0
            self._restart_attempts = 0
            self._last_error = None

    async def report_failure(self, error: BaseException | str) -> None:
        message = str(error)
        async with self._state_lock:
            self._consecutive_failures += 1
            self._last_error = message
            if self._consecutive_failures >= self.policy.failure_threshold:
                self._state = BackendState.DEGRADED
                self._wake.set()
            elif self._state != BackendState.RECOVERING:
                self._state = BackendState.SUSPECT
            failures = self._consecutive_failures
        self._logger.warning(
            "backend health failure count=%s threshold=%s error=%s",
            failures,
            self.policy.failure_threshold,
            message,
        )

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        async with self._state_lock:
            self._state = BackendState.STOPPED

    async def _verify(self) -> None:
        if self._gpu_check is not None:
            await self._gpu_check()
        await self._probe()

    async def _attempt_recovery(self) -> None:
        async with self._state_lock:
            if self._state != BackendState.DEGRADED:
                return
            if self._restart is None:
                return
            if self._restart_attempts >= self.policy.max_restart_attempts:
                return
            self._restart_attempts += 1
            attempt = self._restart_attempts
            self._state = BackendState.RECOVERING

        self._logger.warning("backend recovery restart attempt=%s", attempt)
        try:
            await self._restart()
            await asyncio.sleep(self.policy.post_restart_grace_seconds)
            await self._verify()
        except Exception as exc:
            self._logger.exception("backend recovery failed attempt=%s", attempt)
            async with self._state_lock:
                self._state = BackendState.DEGRADED
                self._last_error = str(exc)
            return

        self._logger.info("backend recovery succeeded attempt=%s", attempt)
        await self.report_success()

    async def run(self) -> None:
        """Run until stop() is called.

        Synthetic probes run only while normal user inference is idle. Request-path failures
        feed the same state machine, so the watchdog does not add competing GPU work during
        healthy active inference.
        """

        try:
            while not self._stop.is_set():
                snapshot = await self.snapshot()
                if snapshot.state == BackendState.DEGRADED:
                    await self._attempt_recovery()
                    snapshot = await self.snapshot()
                    if snapshot.state == BackendState.DEGRADED:
                        try:
                            await asyncio.wait_for(self._wake.wait(), timeout=self.policy.restart_cooldown_seconds)
                        except TimeoutError:
                            pass
                        self._wake.clear()
                        if await self._idle():
                            try:
                                await self._verify()
                            except Exception as exc:
                                await self.report_failure(exc)
                            else:
                                await self.report_success()
                        continue

                if await self._idle():
                    try:
                        await self._verify()
                    except Exception as exc:
                        await self.report_failure(exc)
                    else:
                        await self.report_success()

                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.policy.probe_interval_seconds)
                except TimeoutError:
                    pass
                self._wake.clear()
        finally:
            async with self._state_lock:
                self._state = BackendState.STOPPED
