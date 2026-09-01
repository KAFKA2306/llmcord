from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from backend_probe import BackendProbeConfig, probe_generation
from health_control import GPU_PROCESS_PATTERN_ENV, check_nvidia_gpu, run_restart_command
from production_contract import (
    ProductionContract,
    ProductionContractError,
    load_resolved_config,
    validate_production_contract,
    verify_backend_executable,
)


BACKEND_SERVICE = "llmcord-llama-server.service"
BOT_SERVICE = "llmcord.service"


class RuntimeEvidenceError(RuntimeError):
    """Raised when production runtime evidence cannot be established safely."""


@dataclass(frozen=True)
class ServiceSnapshot:
    service: str
    active_state: str
    sub_state: str
    main_pid: int
    fragment_path: str

    @property
    def running(self) -> bool:
        return self.active_state == "active" and self.sub_state == "running" and self.main_pid > 0


@dataclass(frozen=True)
class ProbeSnapshot:
    healthy: bool
    elapsed_seconds: float
    error_class: str | None
    detail: str | None


@dataclass(frozen=True)
class PhaseEvidence:
    status: str
    backend: ServiceSnapshot | None = None
    bot: ServiceSnapshot | None = None
    probe: ProbeSnapshot | None = None
    gpu_healthy: bool | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RuntimeEvidence:
    status: str
    generated_at: str
    git_head: str
    git_branch: str
    git_clean: bool
    wsl2: bool
    contract: dict[str, Any]
    startup: PhaseEvidence
    restart: PhaseEvidence
    rollback: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(
    argv: Sequence[str],
    *,
    timeout_seconds: float = 30,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeEvidenceError(f"command failed to execute: {argv[0]}") from exc
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeEvidenceError(
            f"command failed ({completed.returncode}): {' '.join(argv)}: {detail[:500]}"
        )
    return completed


def _git_value(*args: str) -> str:
    return _run(("git", *args), timeout_seconds=10).stdout.strip()


def git_state() -> tuple[str, str, bool]:
    head = _git_value("rev-parse", "HEAD")
    branch = _git_value("branch", "--show-current")
    clean = not _run(("git", "status", "--porcelain"), timeout_seconds=10).stdout.strip()
    return head, branch, clean


def is_wsl2() -> bool:
    release = platform.release().lower()
    return "microsoft" in release and "wsl2" in release


def parse_systemd_show(service: str, output: str) -> ServiceSnapshot:
    values: dict[str, str] = {}
    for raw_line in output.splitlines():
        if not raw_line.strip() or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    try:
        pid = int(values.get("MainPID", "0"))
    except ValueError as exc:
        raise RuntimeEvidenceError(f"{service} MainPID is not numeric") from exc
    return ServiceSnapshot(
        service=service,
        active_state=values.get("ActiveState", ""),
        sub_state=values.get("SubState", ""),
        main_pid=pid,
        fragment_path=values.get("FragmentPath", ""),
    )


def service_snapshot(service: str) -> ServiceSnapshot:
    completed = _run(
        (
            "systemctl",
            "--user",
            "show",
            service,
            "--property=ActiveState,SubState,MainPID,FragmentPath",
            "--no-pager",
        ),
        timeout_seconds=10,
    )
    return parse_systemd_show(service, completed.stdout)


def _contract_identity(contract: ProductionContract) -> dict[str, Any]:
    return {
        "backend": contract.backend,
        "backend_release": contract.backend_release,
        "backend_version_or_commit": contract.backend_version_or_commit,
        "backend_executable": contract.backend_executable,
        "model": contract.model,
        "model_artifact": contract.model_artifact,
        "model_revision": contract.model_revision,
        "model_sha256": contract.model_sha256,
        "context_window_tokens": contract.context_window_tokens,
        "network_mode": contract.network_mode,
        "network_endpoint": contract.network_endpoint,
        "supervisor_kind": contract.supervisor_kind,
        "restart_command": list(contract.restart_command),
        "gpu_required": contract.gpu_required,
        "gpu_device_index": contract.gpu_device_index,
        "gpu_process_name_pattern": contract.gpu_process_name_pattern,
    }


def build_probe_config(config: Mapping[str, Any], contract: ProductionContract) -> BackendProbeConfig:
    health = config.get("health_control")
    if not isinstance(health, Mapping):
        raise RuntimeEvidenceError("health_control must be configured")
    model_alias = contract.model.split("/", 1)[1].removesuffix(":vision")
    return BackendProbeConfig(
        base_url=contract.network_endpoint,
        model=model_alias,
        timeout_seconds=float(health.get("probe_timeout_seconds", 30)),
        expected_text=str(health.get("probe_expected_text", "PONG")),
        max_tokens=int(health.get("probe_max_tokens", 8)),
        prompt=str(health.get("probe_prompt", "Reply exactly: PONG")),
    )


async def _probe(config: Mapping[str, Any], contract: ProductionContract) -> ProbeSnapshot:
    result = await probe_generation(build_probe_config(config, contract))
    return ProbeSnapshot(
        healthy=result.healthy,
        elapsed_seconds=result.elapsed_seconds,
        error_class=result.error_class,
        detail=result.detail,
    )


async def _gpu_check(config: Mapping[str, Any], contract: ProductionContract) -> bool:
    if not contract.gpu_required:
        return True
    health = config.get("health_control")
    if not isinstance(health, Mapping):
        raise RuntimeEvidenceError("health_control must be configured")
    gpu = health.get("nvidia_gpu")
    if not isinstance(gpu, Mapping):
        raise RuntimeEvidenceError("health_control.nvidia_gpu must be configured")
    if not contract.gpu_process_name_pattern:
        raise RuntimeEvidenceError("production GPU process identity is missing")
    os.environ[GPU_PROCESS_PATTERN_ENV] = contract.gpu_process_name_pattern
    await check_nvidia_gpu(
        device_index=int(gpu.get("device_index", 0)),
        min_vram_used_mib=int(gpu.get("min_vram_used_mib", 0)),
        timeout_seconds=float(gpu.get("timeout_seconds", 5)),
    )
    return True


async def _wait_until_healthy(
    config: Mapping[str, Any],
    contract: ProductionContract,
    *,
    timeout_seconds: float,
    expected_backend_pid_not: int | None = None,
    expected_bot_pid: int | None = None,
) -> PhaseEvidence:
    deadline = time.monotonic() + timeout_seconds
    last_reason = "runtime did not become healthy"
    while time.monotonic() < deadline:
        try:
            backend = service_snapshot(BACKEND_SERVICE)
            bot = service_snapshot(BOT_SERVICE)
            if not backend.running:
                last_reason = f"backend service not running: {backend.active_state}/{backend.sub_state}"
                await asyncio.sleep(1)
                continue
            if not bot.running:
                last_reason = f"bot service not running: {bot.active_state}/{bot.sub_state}"
                await asyncio.sleep(1)
                continue
            if expected_backend_pid_not is not None and backend.main_pid == expected_backend_pid_not:
                last_reason = "backend PID did not change after restart"
                await asyncio.sleep(1)
                continue
            if expected_bot_pid is not None and bot.main_pid != expected_bot_pid:
                return PhaseEvidence(
                    status="fail",
                    backend=backend,
                    bot=bot,
                    reason="bot PID changed during backend-only restart",
                )
            gpu_healthy = await _gpu_check(config, contract)
            probe = await _probe(config, contract)
            if not probe.healthy:
                last_reason = f"synthetic generation unhealthy: {probe.error_class or 'unknown'}"
                await asyncio.sleep(1)
                continue
            return PhaseEvidence(
                status="pass",
                backend=backend,
                bot=bot,
                probe=probe,
                gpu_healthy=gpu_healthy,
            )
        except Exception as exc:
            last_reason = str(exc)
            await asyncio.sleep(1)
    return PhaseEvidence(status="fail", reason=last_reason)


async def collect_runtime_evidence(
    *,
    config_path: str,
    exercise_restart: bool,
    startup_timeout_seconds: float,
    restart_timeout_seconds: float,
    rollback_commit: str | None,
) -> RuntimeEvidence:
    head, branch, clean = git_state()
    wsl2 = is_wsl2()
    config = load_resolved_config(config_path)
    try:
        contract = validate_production_contract(config, verify_artifact=True)
    except ProductionContractError as exc:
        raise RuntimeEvidenceError(f"production contract invalid: {exc}") from exc
    if contract is None:
        raise RuntimeEvidenceError("production contract is disabled")
    verify_backend_executable(contract)

    if branch != "main":
        raise RuntimeEvidenceError(f"runtime evidence must be collected from main, got {branch or 'detached'}")
    if not clean:
        raise RuntimeEvidenceError("runtime evidence requires a clean git worktree")
    if not wsl2:
        raise RuntimeEvidenceError("production contract requires WSL2")
    if contract.network_mode != "wsl" or contract.supervisor_kind != "systemd":
        raise RuntimeEvidenceError("runtime evidence collector requires WSL + systemd production contract")

    _run(("systemctl", "--user", "start", BACKEND_SERVICE, BOT_SERVICE), timeout_seconds=30)
    startup = await _wait_until_healthy(
        config,
        contract,
        timeout_seconds=startup_timeout_seconds,
    )

    restart = PhaseEvidence(status="unverified", reason="restart exercise not requested")
    if startup.status == "pass" and exercise_restart:
        assert startup.backend is not None
        assert startup.bot is not None
        health = config.get("health_control")
        if not isinstance(health, Mapping):
            raise RuntimeEvidenceError("health_control must be configured")
        await run_restart_command(
            contract.restart_command,
            timeout_seconds=float(health.get("restart_command_timeout_seconds", 30)),
        )
        restart = await _wait_until_healthy(
            config,
            contract,
            timeout_seconds=restart_timeout_seconds,
            expected_backend_pid_not=startup.backend.main_pid,
            expected_bot_pid=startup.bot.main_pid,
        )

    rollback: dict[str, Any]
    if rollback_commit:
        rollback = {
            "status": "unverified",
            "target_commit": rollback_commit,
            "reason": (
                "collector does not mutate git history automatically; execute the documented rollback "
                "procedure against this verified target, then collect a second runtime evidence file"
            ),
        }
    else:
        rollback = {
            "status": "unverified",
            "target_commit": None,
            "reason": "no previously verified rollback target supplied",
        }

    statuses = [startup.status, restart.status]
    if "fail" in statuses:
        overall = "fail"
    elif exercise_restart and all(status == "pass" for status in statuses):
        overall = "pass_with_rollback_unverified"
    else:
        overall = "unverified"

    return RuntimeEvidence(
        status=overall,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        git_head=head,
        git_branch=branch,
        git_clean=clean,
        wsl2=wsl2,
        contract=_contract_identity(contract),
        startup=startup,
        restart=restart,
        rollback=rollback,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect real WSL2 llmcord production runtime evidence")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exercise-restart", action="store_true")
    parser.add_argument("--startup-timeout-seconds", type=float, default=180)
    parser.add_argument("--restart-timeout-seconds", type=float, default=180)
    parser.add_argument("--rollback-commit")
    args = parser.parse_args(argv)

    try:
        evidence = asyncio.run(
            collect_runtime_evidence(
                config_path=args.config,
                exercise_restart=args.exercise_restart,
                startup_timeout_seconds=args.startup_timeout_seconds,
                restart_timeout_seconds=args.restart_timeout_seconds,
                rollback_commit=args.rollback_commit,
            )
        )
    except (RuntimeEvidenceError, OSError, ValueError) as exc:
        print(json.dumps({"status": "unverified", "error": str(exc)}, ensure_ascii=False))
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence.status, "output": str(output)}, ensure_ascii=False))
    if evidence.status == "fail":
        return 1
    if evidence.status == "pass_with_rollback_unverified":
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
