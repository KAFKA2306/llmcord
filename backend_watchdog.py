from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
import logging
from typing import Awaitable, Callable

from backend_probe import BackendProbeResult


class WatchdogState(StrEnum):
    STARTING = "starting"
    HEALTHY = "healthy"
    SUSPECT = "suspect"
    UNAVAILABLE = "unavailable"
    RESTARTING = "restarting"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class WatchdogPolicy:
    probe_interval_seconds: float
    failure_threshold: int
    restart_cooldown_seconds: float
    restart_settle_seconds: float
    max_restart_attempts: int

    def __post_init__(self) -> None:
        if self.probe_interval_seconds <= 0:
            raise ValueError("probe_interval_seconds must be positive")
        if not isinstance(self.failure_threshold, int) or isinstance(self.failure_threshold, bool) or self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be a positive integer")
        if self.restart_cooldown_seconds < 0:
            raise ValueError("restart_cooldown_seconds must be non-negative")
        if self.restart_settle_seconds < 0:
            raise ValueError("restart_settle_seconds must be non-negative")
        if not isinstance(self.max_restart_attempts, int) or isinstance(self.max_restart_attempts, bool) or self.max_restart_attempts <= 0:
            raise ValueError("max_restart_attempts must be a positive integer")


@dataclass(frozen=True)
class WatchdogSnapshot:
    state: WatchdogState
    accepting: bool
    consecutive_failures: int
    restart_attempts_since_healthy: int
    total_restart_attempts: int
    last_probe_healthy: bool | None
    last_error_class: str | None


Probe = Callable[[], Awaitable[BackendProbeResult]]
RestartBackend = Callable[[], Awaitable[bool]]
SetAccepting = Callable[[bool], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]


class BackendWatchdog:
    """Deterministic health/recovery controller around a real generation probe.

    The watchdog owns no process manager. A caller supplies exactly one restart
    authority and one admission-control callback. LLM output never decides when
    to restart the backend.
    """

    def __init__(
        self,
        policy: WatchdogPolicy,
        *,
        probe: Probe,
        restart_backend: RestartBackend,
        set_accepting: SetAccepting,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.policy = policy
        self._probe = probe
        self._restart_backend = restart_backend
        self._set_accepting_callback = set_accepting
        self._sleep = sleep

        self.state = WatchdogState.STARTING
        self.consecutive_failures = 0
        self.restart_attempts_since_healthy = 0
        self.total_restart_attempts = 0
        self.last_probe: BackendProbeResult | None = None
        self._accepting: bool | None = None

    async def initialize(self) -> None:
        await self._set_accepting(False)

    async def _set_accepting(self, accepting: bool) -> None:
        if self._accepting == accepting:
            return
        await self._set_accepting_callback(accepting)
        self._accepting = accepting

    async def _run_probe(self) -> BackendProbeResult:
        try:
            result = await self._probe()
        except Exception as exc:
            logging.exception("backend watchdog probe raised unexpectedly")
            result = BackendProbeResult(
                healthy=False,
                elapsed_seconds=0.0,
                error_class=type(exc).__name__,
                detail=str(exc),
            )
        self.last_probe = result
        return result

    async def _mark_healthy(self, result: BackendProbeResult) -> None:
        self.last_probe = result
        self.consecutive_failures = 0
        self.restart_attempts_since_healthy = 0
        self.state = WatchdogState.HEALTHY
        await self._set_accepting(True)

    async def tick(self) -> WatchdogSnapshot:
        result = await self._run_probe()
        if result.healthy:
            await self._mark_healthy(result)
            return self.snapshot()

        self.consecutive_failures += 1
        if self.consecutive_failures < self.policy.failure_threshold:
            self.state = WatchdogState.SUSPECT
            return self.snapshot()

        await self._set_accepting(False)
        self.state = WatchdogState.UNAVAILABLE

        if self.restart_attempts_since_healthy >= self.policy.max_restart_attempts:
            self.state = WatchdogState.DEGRADED
            return self.snapshot()

        if self.policy.restart_cooldown_seconds:
            await self._sleep(self.policy.restart_cooldown_seconds)

        self.state = WatchdogState.RESTARTING
        self.restart_attempts_since_healthy += 1
        self.total_restart_attempts += 1

        try:
            restarted = await self._restart_backend()
        except Exception:
            logging.exception("backend restart authority raised unexpectedly")
            restarted = False

        if not restarted:
            self.state = (
                WatchdogState.DEGRADED
                if self.restart_attempts_since_healthy >= self.policy.max_restart_attempts
                else WatchdogState.UNAVAILABLE
            )
            return self.snapshot()

        if self.policy.restart_settle_seconds:
            await self._sleep(self.policy.restart_settle_seconds)

        recovery = await self._run_probe()
        if recovery.healthy:
            await self._mark_healthy(recovery)
            return self.snapshot()

        self.consecutive_failures += 1
        self.state = (
            WatchdogState.DEGRADED
            if self.restart_attempts_since_healthy >= self.policy.max_restart_attempts
            else WatchdogState.UNAVAILABLE
        )
        return self.snapshot()

    async def run_forever(self) -> None:
        await self.initialize()
        while True:
            snapshot = await self.tick()
            logging.info(
                "backend_watchdog state=%s accepting=%s consecutive_failures=%s restart_attempts=%s last_error=%s",
                snapshot.state,
                snapshot.accepting,
                snapshot.consecutive_failures,
                snapshot.restart_attempts_since_healthy,
                snapshot.last_error_class,
            )
            await self._sleep(self.policy.probe_interval_seconds)

    def snapshot(self) -> WatchdogSnapshot:
        return WatchdogSnapshot(
            state=self.state,
            accepting=bool(self._accepting),
            consecutive_failures=self.consecutive_failures,
            restart_attempts_since_healthy=self.restart_attempts_since_healthy,
            total_restart_attempts=self.total_restart_attempts,
            last_probe_healthy=None if self.last_probe is None else self.last_probe.healthy,
            last_error_class=None if self.last_probe is None else self.last_probe.error_class,
        )


async def restart_backend_command(command: list[str], *, timeout_seconds: float) -> bool:
    """Invoke one configured supervisor command without a shell.

    The command is intentionally argv-based so config cannot acquire shell
    interpolation semantics. stdout/stderr are not copied into bot logs because
    supervisor output can contain deployment-specific data.
    """

    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("restart_command must be a non-empty list of non-empty strings")
    if timeout_seconds <= 0:
        raise ValueError("restart timeout must be positive")

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        logging.exception("failed to start configured backend restart command")
        return False

    try:
        return_code = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        process.kill()
        await process.wait()
        return False

    return return_code == 0
