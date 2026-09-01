from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from enum import StrEnum
import io
import logging
import os
import re
from typing import Awaitable, Callable, Optional, Sequence


Probe = Callable[[], Awaitable[None]]
RestartAction = Callable[[], Awaitable[None]]
GpuCheck = Callable[[], Awaitable[None]]
IdleCheck = Callable[[], Awaitable[bool]]

GPU_PROCESS_PATTERN_ENV = "LLMCORD_GPU_PROCESS_PATTERN"


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


@dataclass(frozen=True)
class NvidiaGpuSnapshot:
    uuid: str
    vram_used_mib: int
    utilization_percent: float
    temperature_c: float


@dataclass(frozen=True)
class NvidiaComputeProcess:
    pid: int
    process_name: str
    used_gpu_memory_mib: int | None


def parse_nvidia_gpu_snapshot(output: str) -> NvidiaGpuSnapshot:
    rows = list(csv.reader(io.StringIO(output.strip())))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise HealthControlError("NVIDIA GPU check returned an unexpected device payload")
    fields = [field.strip() for field in rows[0]]
    if not fields[0]:
        raise HealthControlError("NVIDIA GPU UUID is empty")
    try:
        return NvidiaGpuSnapshot(
            uuid=fields[0],
            vram_used_mib=int(float(fields[1])),
            utilization_percent=float(fields[2]),
            temperature_c=float(fields[3]),
        )
    except ValueError as exc:
        raise HealthControlError("NVIDIA GPU metrics were not numeric") from exc


def _parse_optional_gpu_memory(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"n/a", "[n/a]", "not supported", ""}:
        return None
    try:
        return int(float(value.strip()))
    except ValueError as exc:
        raise HealthControlError(f"NVIDIA compute-process GPU memory was invalid: {value!r}") from exc


def parse_nvidia_compute_processes(output: str) -> list[NvidiaComputeProcess]:
    if not output.strip():
        return []

    processes: list[NvidiaComputeProcess] = []
    for row in csv.reader(io.StringIO(output.strip())):
        if len(row) != 3:
            raise HealthControlError("NVIDIA compute-process query returned an unexpected payload")
        pid_text, process_name, memory_text = (field.strip() for field in row)
        if not process_name:
            raise HealthControlError("NVIDIA compute-process query returned an empty process name")
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise HealthControlError("NVIDIA compute-process PID was not numeric") from exc
        if pid <= 0:
            raise HealthControlError("NVIDIA compute-process PID must be positive")
        processes.append(
            NvidiaComputeProcess(
                pid=pid,
                process_name=process_name,
                used_gpu_memory_mib=_parse_optional_gpu_memory(memory_text),
            )
        )
    return processes


def require_expected_gpu_process(
    processes: Sequence[NvidiaComputeProcess],
    *,
    process_name_pattern: str,
) -> NvidiaComputeProcess:
    try:
        pattern = re.compile(process_name_pattern, re.IGNORECASE)
    except re.error as exc:
        raise HealthControlError(f"invalid GPU process_name_pattern: {exc}") from exc

    for process in processes:
        if pattern.search(process.process_name):
            return process

    raise HealthControlError(
        "expected backend compute process is not present on the selected GPU; "
        "possible CPU fallback, wrong GPU, or backend process loss"
    )


async def _run_nvidia_smi(
    executable: str,
    *arguments: str,
    timeout_seconds: float,
    failure_label: str,
) -> str:
    process = await asyncio.create_subprocess_exec(
        executable,
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise HealthControlError(f"{failure_label} timed out") from exc

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
        raise HealthControlError(f"{failure_label} failed: {detail[:500]}")

    return stdout.decode("utf-8", errors="replace")


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
    """Require the expected NVIDIA GPU state for the production backend.

    The base check requires a visible GPU and a calibrated total VRAM floor. When the
    production entrypoint provides LLMCORD_GPU_PROCESS_PATTERN, the selected GPU must also
    contain a matching active compute process. The compute-app query is filtered by the
    immutable GPU UUID obtained from the first query rather than depending on GPU index order.
    """

    if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
        raise HealthControlError("GPU device_index must be a non-negative integer")
    if not isinstance(min_vram_used_mib, int) or isinstance(min_vram_used_mib, bool) or min_vram_used_mib < 0:
        raise HealthControlError("min_vram_used_mib must be a non-negative integer")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise HealthControlError("GPU check timeout must be a positive number")

    gpu_output = await _run_nvidia_smi(
        executable,
        f"--id={device_index}",
        "--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
        timeout_seconds=timeout_seconds,
        failure_label="NVIDIA GPU query",
    )
    snapshot = parse_nvidia_gpu_snapshot(gpu_output)

    if snapshot.vram_used_mib < min_vram_used_mib:
        raise HealthControlError(
            f"GPU VRAM residency is below the production floor: "
            f"{snapshot.vram_used_mib} MiB < {min_vram_used_mib} MiB"
        )

    process_name_pattern = os.environ.get(GPU_PROCESS_PATTERN_ENV)
    if not process_name_pattern:
        return

    process_output = await _run_nvidia_smi(
        executable,
        f"--id={snapshot.uuid}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
        timeout_seconds=timeout_seconds,
        failure_label="NVIDIA compute-process query",
    )
    processes = parse_nvidia_compute_processes(process_output)
    require_expected_gpu_process(
        processes,
        process_name_pattern=process_name_pattern,
    )


class BackendWatchdog:
    """Deterministic health state machine for one production inference backend.

    Startup and recovery require a real generation probe plus the optional GPU check.
    User-generation success is accepted as healthy only when the optional GPU check also
    succeeds. HEALTHY/SUSPECT states perform a lightweight periodic GPU/process check but do
    not inject background model generation, preserving inference concurrency=1.
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

    async def _mark_healthy(self) -> None:
        async with self._state_lock:
            self._state = BackendState.HEALTHY
            self._consecutive_failures = 0
            self._restart_attempts = 0
            self._last_error = None

    async def report_success(self) -> None:
        """Report successful user generation, retaining GPU/process health as authority."""
        if self._gpu_check is not None:
            try:
                await self._gpu_check()
            except Exception as exc:
                await self.report_failure(exc, immediate=True)
                return
        await self._mark_healthy()

    async def report_failure(self, error: BaseException | str, *, immediate: bool = False) -> None:
        message = str(error)
        async with self._state_lock:
            previous_state = self._state
            next_failures = self._consecutive_failures + 1
            if immediate:
                next_failures = max(next_failures, self.policy.failure_threshold)
            self._consecutive_failures = next_failures
            self._last_error = message
            if immediate or self._consecutive_failures >= self.policy.failure_threshold:
                self._state = BackendState.DEGRADED
                if previous_state not in {BackendState.DEGRADED, BackendState.RECOVERING}:
                    self._wake.set()
            elif previous_state == BackendState.STARTING:
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

    async def _check_runtime_gpu(self) -> None:
        if self._gpu_check is None:
            return
        try:
            await self._gpu_check()
        except Exception as exc:
            # GPU/process disappearance or CPU fallback is a hard production-health failure,
            # not a transient generation failure that should remain accepting until threshold.
            await self.report_failure(exc, immediate=True)

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
        await self._mark_healthy()

    async def _wait(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        except TimeoutError:
            pass
        self._wake.clear()

    async def run(self) -> None:
        """Run until stop() is called.

        Synthetic generation is used at startup and during degraded/recovery only. While
        HEALTHY or SUSPECT, no background generation is injected. The watchdog still checks
        GPU/process presence periodically when inference is idle, and every successful user
        generation is followed by the same GPU/process check before health is reset to HEALTHY.
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
                            await self._mark_healthy()
                    await self._wait(self.policy.probe_interval_seconds)
                    continue

                if snapshot.state == BackendState.DEGRADED:
                    await self._attempt_recovery()
                    snapshot = await self.snapshot()
                    if snapshot.state == BackendState.DEGRADED:
                        await self._wait(self.policy.restart_cooldown_seconds)
                        if self._stop.is_set():
                            continue
                        if await self._idle():
                            try:
                                await self._verify()
                            except Exception as exc:
                                await self.report_failure(exc)
                            else:
                                await self._mark_healthy()
                    continue

                await self._wait(self.policy.probe_interval_seconds)
                if self._stop.is_set():
                    continue
                if self._gpu_check is not None and await self._idle():
                    await self._check_runtime_gpu()
        finally:
            async with self._state_lock:
                self._state = BackendState.STOPPED
