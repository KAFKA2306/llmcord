from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml


class ProductionContractError(RuntimeError):
    """Raised when the production deployment contract is incomplete or inconsistent."""


_PLACEHOLDER_RE = re.compile(r"[<>]|\b(todo|tbd|placeholder)\b", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_RE = re.compile(r"^v\d+\.\d+\.\d+$")
_ALLOWED_NETWORK_MODES = {"native", "wsl", "docker"}
_ALLOWED_SUPERVISORS = {"systemd", "docker-compose", "windows-service", "external"}


def resolve_env(node: Any) -> Any:
    if isinstance(node, dict):
        resolved: dict[str, Any] = {}
        for key, value in node.items():
            if key.endswith("_env"):
                resolved[key.removesuffix("_env")] = os.environ.get(str(value))
            else:
                resolved[key] = resolve_env(value)
        return resolved
    if isinstance(node, list):
        return [resolve_env(value) for value in node]
    return node


def load_resolved_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ProductionContractError("config root must be a mapping")
    return resolve_env(payload)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionContractError(f"{name} must be a mapping")
    return value


def _required_text(mapping: Mapping[str, Any], key: str, prefix: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProductionContractError(f"{prefix}.{key} must be a non-empty string")
    value = value.strip()
    if _PLACEHOLDER_RE.search(value):
        raise ProductionContractError(f"{prefix}.{key} contains a placeholder")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, prefix: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProductionContractError(f"{prefix}.{key} must be a positive integer")
    return value


def _argv(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ProductionContractError(f"{name} must be a non-empty argv list")
    parts = tuple(value)
    if any(not isinstance(part, str) or not part for part in parts):
        raise ProductionContractError(f"{name} must contain only non-empty strings")
    return parts


def _normalize_endpoint(value: str) -> str:
    return value.rstrip("/")


def _validate_endpoint(endpoint: str, network_mode: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or parsed.path.rstrip("/") != "/v1":
        raise ProductionContractError("production.network.endpoint must be an absolute http URL ending in /v1")
    host = parsed.hostname.lower()
    if host in {"0.0.0.0", "::"}:
        raise ProductionContractError("production.network.endpoint must not use a wildcard address")
    if network_mode in {"native", "wsl"} and host not in {"localhost", "127.0.0.1", "::1"}:
        raise ProductionContractError("native/WSL production endpoint must be loopback-only")
    if network_mode == "docker" and host in {"localhost", "127.0.0.1", "::1"}:
        raise ProductionContractError("Docker production endpoint must use a service/DNS name, not localhost")
    return _normalize_endpoint(endpoint)


@dataclass(frozen=True)
class ProductionContract:
    backend: str
    backend_release: str
    backend_version_or_commit: str
    backend_executable: str
    model: str
    model_upstream: str
    artifact_repo: str
    model_artifact: str
    model_revision: str
    artifact_url: str
    model_sha256: str
    quantization_or_dtype: str
    context_window_tokens: int
    network_mode: str
    network_endpoint: str
    supervisor_kind: str
    restart_command: tuple[str, ...]
    gpu_required: bool
    gpu_device_index: int | None
    min_vram_used_mib: int | None
    gpu_process_name_pattern: str | None
    artifact_path: str


def validate_production_contract(
    config: Mapping[str, Any],
    *,
    verify_artifact: bool = True,
) -> ProductionContract | None:
    production = config.get("production")
    if production is None:
        return None
    production = _mapping(production, "production")
    enabled = production.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ProductionContractError("production.enabled must be boolean")
    if not enabled:
        return None

    backend = _required_text(production, "backend", "production")
    backend_release = _required_text(production, "backend_release", "production")
    if _RELEASE_RE.fullmatch(backend_release) is None:
        raise ProductionContractError("production.backend_release must be an explicit vX.Y.Z release")
    backend_version = _required_text(production, "backend_version_or_commit", "production")
    if _GIT_COMMIT_RE.fullmatch(backend_version) is None:
        raise ProductionContractError("production.backend_version_or_commit must be a full commit SHA")
    backend_executable = _required_text(production, "backend_executable", "production")
    if backend == "llamacpp" and Path(os.path.expanduser(backend_executable)).name != "llama-server":
        raise ProductionContractError("llamacpp production.backend_executable must name llama-server")

    model = _required_text(production, "model", "production")
    if "/" not in model:
        raise ProductionContractError("production.model must use <provider>/<model> form")
    provider_name, _ = model.removesuffix(":vision").split("/", 1)
    if provider_name != backend:
        raise ProductionContractError("production.backend must equal the provider prefix of production.model")

    model_cfg = _mapping(production.get("model_artifact"), "production.model_artifact")
    model_upstream = _required_text(model_cfg, "upstream", "production.model_artifact")
    artifact_repo = _required_text(model_cfg, "artifact_repo", "production.model_artifact")
    artifact = _required_text(model_cfg, "artifact", "production.model_artifact")
    revision = _required_text(model_cfg, "revision", "production.model_artifact")
    if _GIT_REVISION_RE.fullmatch(revision) is None:
        raise ProductionContractError("production.model_artifact.revision must be a pinned git revision")
    artifact_url = _required_text(model_cfg, "artifact_url", "production.model_artifact")
    expected_url = f"https://huggingface.co/{artifact_repo}/resolve/{revision}/{artifact}"
    if artifact_url != expected_url:
        raise ProductionContractError("production.model_artifact.artifact_url must match repo/revision/artifact")
    sha256 = _required_text(model_cfg, "sha256", "production.model_artifact")
    if _SHA256_RE.fullmatch(sha256) is None:
        raise ProductionContractError("production.model_artifact.sha256 must be 64 lowercase hex characters")
    quantization = _required_text(model_cfg, "quantization_or_dtype", "production.model_artifact")
    context_window = _positive_int(model_cfg, "context_window_tokens", "production.model_artifact")
    artifact_path = _required_text(model_cfg, "verify_path", "production.model_artifact")
    if Path(os.path.expanduser(artifact_path)).name != artifact:
        raise ProductionContractError("production.model_artifact.verify_path must end with the pinned artifact filename")

    network = _mapping(production.get("network"), "production.network")
    network_mode = _required_text(network, "mode", "production.network").lower()
    if network_mode not in _ALLOWED_NETWORK_MODES:
        raise ProductionContractError(f"production.network.mode must be one of {sorted(_ALLOWED_NETWORK_MODES)}")
    endpoint = _validate_endpoint(_required_text(network, "endpoint", "production.network"), network_mode)

    supervisor = _mapping(production.get("supervisor"), "production.supervisor")
    supervisor_kind = _required_text(supervisor, "kind", "production.supervisor").lower()
    if supervisor_kind not in _ALLOWED_SUPERVISORS:
        raise ProductionContractError(f"production.supervisor.kind must be one of {sorted(_ALLOWED_SUPERVISORS)}")
    restart_command = _argv(supervisor.get("restart_command"), "production.supervisor.restart_command")

    gpu = _mapping(production.get("gpu"), "production.gpu")
    gpu_required = gpu.get("required")
    if not isinstance(gpu_required, bool):
        raise ProductionContractError("production.gpu.required must be boolean")
    gpu_device_index = None
    min_vram_used_mib = None
    gpu_process_name_pattern = None
    if gpu_required:
        gpu_device_index = gpu.get("device_index")
        min_vram_used_mib = gpu.get("min_vram_used_mib")
        gpu_process_name_pattern = _required_text(gpu, "process_name_pattern", "production.gpu")
        if not isinstance(gpu_device_index, int) or isinstance(gpu_device_index, bool) or gpu_device_index < 0:
            raise ProductionContractError("production.gpu.device_index must be a non-negative integer")
        if not isinstance(min_vram_used_mib, int) or isinstance(min_vram_used_mib, bool) or min_vram_used_mib <= 0:
            raise ProductionContractError("production.gpu.min_vram_used_mib must be a positive integer")
        try:
            pattern = re.compile(gpu_process_name_pattern, re.IGNORECASE)
        except re.error as exc:
            raise ProductionContractError(f"production.gpu.process_name_pattern is invalid: {exc}") from exc
        if backend == "llamacpp" and pattern.search(Path(os.path.expanduser(backend_executable)).name) is None:
            raise ProductionContractError("production GPU process pattern must match the pinned llama-server executable")

    providers = _mapping(config.get("providers"), "providers")
    if set(providers) != {backend}:
        raise ProductionContractError("production config must expose exactly one selected provider")
    provider = _mapping(providers.get(backend), f"providers.{backend}")
    configured_endpoint = _required_text(provider, "base_url", f"providers.{backend}")
    if _normalize_endpoint(configured_endpoint) != endpoint:
        raise ProductionContractError(f"providers.{backend}.base_url must exactly match production.network.endpoint")

    models = _mapping(config.get("models"), "models")
    if set(models) != {model}:
        raise ProductionContractError("production config must expose exactly one selected model")
    model_runtime_cfg = _mapping(models.get(model) or {}, f"models.{model}")
    context_cfg = _mapping(model_runtime_cfg.get("context_management"), f"models.{model}.context_management")
    if context_cfg.get("context_window_tokens") != context_window:
        raise ProductionContractError("runtime context_window_tokens must exactly match production model artifact context")

    runtime_control = _mapping(config.get("runtime_control"), "runtime_control")
    if runtime_control.get("max_concurrency") != 1:
        raise ProductionContractError("production runtime_control.max_concurrency must be 1")
    if config.get("max_images") != 0:
        raise ProductionContractError("production image input must remain disabled until an mmproj artifact is pinned")

    health = _mapping(config.get("health_control"), "health_control")
    if health.get("enabled") is not True or health.get("model") != model:
        raise ProductionContractError("health_control must be enabled for the selected production model")
    if tuple(health.get("restart_command") or ()) != restart_command:
        raise ProductionContractError("health_control.restart_command must exactly match production supervisor")
    gpu_health = _mapping(health.get("nvidia_gpu") or {}, "health_control.nvidia_gpu")
    if gpu_required:
        if gpu_health.get("enabled") is not True:
            raise ProductionContractError("health_control.nvidia_gpu.enabled must be true in GPU production")
        if gpu_health.get("device_index") != gpu_device_index:
            raise ProductionContractError("health GPU device must match production GPU device")
        if gpu_health.get("min_vram_used_mib") != min_vram_used_mib:
            raise ProductionContractError("health GPU VRAM floor must match production GPU contract")

    contract = ProductionContract(
        backend=backend,
        backend_release=backend_release,
        backend_version_or_commit=backend_version,
        backend_executable=backend_executable,
        model=model,
        model_upstream=model_upstream,
        artifact_repo=artifact_repo,
        model_artifact=artifact,
        model_revision=revision,
        artifact_url=artifact_url,
        model_sha256=sha256,
        quantization_or_dtype=quantization,
        context_window_tokens=context_window,
        network_mode=network_mode,
        network_endpoint=endpoint,
        supervisor_kind=supervisor_kind,
        restart_command=restart_command,
        gpu_required=gpu_required,
        gpu_device_index=gpu_device_index,
        min_vram_used_mib=min_vram_used_mib,
        gpu_process_name_pattern=gpu_process_name_pattern,
        artifact_path=artifact_path,
    )
    if verify_artifact:
        verify_model_artifact(contract)
    return contract


def verify_model_artifact(contract: ProductionContract) -> None:
    path = Path(os.path.expanduser(contract.artifact_path)).resolve()
    if not path.is_file():
        raise ProductionContractError(f"production model artifact is not accessible: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != contract.model_sha256:
        raise ProductionContractError(
            f"production model artifact hash mismatch: expected {contract.model_sha256}, got {actual}"
        )


def verify_backend_executable(contract: ProductionContract) -> None:
    path = Path(os.path.expanduser(contract.backend_executable)).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ProductionContractError(f"production backend executable is not runnable: {path}")
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProductionContractError("could not execute pinned backend --version") from exc
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise ProductionContractError("pinned backend --version returned non-zero")
    if contract.backend_release.removeprefix("v") not in output:
        raise ProductionContractError("backend release does not match production.backend_release")
    if contract.backend_version_or_commit[:9] not in output:
        raise ProductionContractError("backend commit does not match production.backend_version_or_commit")
