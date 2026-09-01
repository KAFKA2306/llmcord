from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
import logging
from typing import Awaitable, Callable, Optional, Sequence


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


async def run_restart_command(command: Sequence[str], timeout_seconds: float) -> None:
    """Execute one explicit argv restart action without a shell."""

    if not command or any(not isinstance(part, str) or not part for part in command):
        raise HealthControlError("restart_command must be a non-empty argv list")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise HealthControlError("restart command timeout must be a positive number")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise HealthControlError("restart command timed out") from exc

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
        raise HealthControlError(f"restart command failed with exit code {process.returncode}: {detail[:500]}")


async def check_nvidia_gpu(
    *,
    device_index: int,
    min_vram_used_mib: int,
    timeout_seconds: float,
    executable: str = "nvidia-smi",
) -> None:
    """Require a visible NVIDIA GPU and an expected minimum resident VRAM footprint.

    This is a coarse fail-closed check. Exact backend-process/GPU attribution remains a
    separate production acceptance requirement because unrelated GPU allocations can exist.
    """

    if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
        raise HealthControlError("GPU device_index must be a non-negative integer")
    if not isinstance(min_vram_used_mib, int) or isinstance(min_vram_used_mib, bool) or min_vram_used_mib < 0:
        raise HealthControlError("min_vram_used_mib must be a non-negative integer")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise HealthControlError("GPU check timeout must be a positive number")

    process = await asyncio.create_subprocess_exec(
        executable,
        f"--id={device_index}",
        "--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise HealthControlError("nvidia-smi timed out") from exc

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise HealthControlError(f"NVIDIA GPU check failed: {detail[:500]}")

    line = stdout.decode("utf-8", errors="replace").strip().splitlines()
    if len(line) != 1:
        raise HealthControlError("NVIDIA GPU check returned an unexpected device count")
    fields = [field.strip() for field in line[0].split(",")]
    if len(fields) != 4 or not fields[0]:
        raise HealthControlError("NVIDIA GPU check returned an unexpected payload")
    try:
        vram_used_mib = int(float(fields[1]))
        float(fields[2])
        float(fields[3])
    except ValueError as exc:
        raise HealthControlError("NVIDIA GPU metrics were not numeric") from exc

    if vram_used_mib < min_vram_used_mib:
        raise HealthControlError(
            f"GPU VRAM residency is below the production floor: {vram_used_mib} MiB < {min_vram_used_mib} MiB"
        )


class BackendWatchdog:
    """Deterministic health state machine for one production inference backend.

    Startup and recovery require a real generation probe plus the optional GPU check.
    Healthy-state user requests provide the normal health signal; no competing background
    generation is injected while the backend is accepting user inference.
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
            previous_state = self._state
            self._consecutive_failures += 1
            self._last_error = message
            if self._consecutive_failures >= self.policy.failure_threshold:
                self._state = BackendState.DEGRADED
                self._wake.set()
            elif previous_state == BackendState.STARTING:
                # Never admit user work before at least one real generation probe has succeeded.
                self._state = BackendState.STARTING
            elif previous_state != BackendState.RECOVERING:
                self._state = BackendState.SUSPECT
            failures = self._consecutive_failures
            state = self._state
        self._logger.warning(
            "backend health failure state=%s count=%s threshold=%s error=%s",
            state,
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

    async def _wait(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        except TimeoutError:
            pass
        self._wake.clear()

    async def run(self) -> None:
        """Run until stop() is called.

        Synthetic generation is used at startup and during degraded/recovery only. While
        HEALTHY or SUSPECT, the watchdog does not issue background generation, which keeps
        the production concurrency authority in the normal inference gate. Real user-request
        success/failure drives the healthy/suspect/degraded transitions during normal use.
        """

        try:
            while not self._stop.is_set():
                snapshot = await self.snapshot()

                if snapshot.state == BackendState.STARTING:
                    if await self._idle():
                        try:
                            await self._verify()
                        except Exception as exc:
                            await self.report_failure(exc)
                        else:
                            await self.report_success()
                    await self._wait(self.policy.probe_interval_seconds)
                    continue

                if snapshot.state == BackendState.DEGRADED:
                    await self._attempt_recovery()
                    snapshot = await self.snapshot()
                    if snapshot.state == BackendState.DEGRADED:
                        await self._wait(self.policy.restart_cooldown_seconds)
                        if await self._idle():
                            try:
                                await self._verify()
                            except Exception as exc:
                                await self.report_failure(exc)
                            else:
                                await self.report_success()
                    continue

                # HEALTHY/SUSPECT: user-generation outcomes are the health signal. Do not
                # race a background synthetic request against the max_concurrency=1 path.
                await self._wait(self.policy.probe_interval_seconds)
        finally:
            async with self._state_lock:
                self._state = BackendState.STOPPED
