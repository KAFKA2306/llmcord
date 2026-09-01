from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml


class ProductionContractError(RuntimeError):
    """Raised when the production deployment contract is incomplete or inconsistent."""


_PLACEHOLDER_RE = re.compile(r"[<>]|\b(todo|tbd|placeholder)\b", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FORBIDDEN_FLOATING = {"latest", "main", "master", "head", "nightly"}
_ALLOWED_NETWORK_MODES = {"native", "wsl", "docker"}
_ALLOWED_SUPERVISORS = {"systemd", "docker-compose", "windows-service", "external"}


def resolve_env(node: Any) -> Any:
    """Resolve config keys ending in `_env` with the same semantics as llmcord.py."""
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


def _pinned_text(mapping: Mapping[str, Any], key: str, prefix: str) -> str:
    value = _required_text(mapping, key, prefix)
    if value.lower() in _FORBIDDEN_FLOATING or value.lower().endswith(":latest"):
        raise ProductionContractError(f"{prefix}.{key} must be pinned, not {value!r}")
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
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProductionContractError("production.network.endpoint must be an absolute http(s) URL")
    host = parsed.hostname.lower()
    if host in {"0.0.0.0", "::"}:
        raise ProductionContractError("production.network.endpoint must not use a wildcard listen address")
    if network_mode == "docker" and host in {"localhost", "127.0.0.1", "::1"}:
        raise ProductionContractError(
            "production.network.endpoint must not use localhost in Docker mode; use the backend service/DNS name"
        )
    return _normalize_endpoint(endpoint)


@dataclass(frozen=True)
class ProductionContract:
    backend: str
    backend_version_or_commit: str
    model: str
    model_upstream: str
    model_artifact: str
    model_revision: str
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
    artifact_path: str | None


def validate_production_contract(config: Mapping[str, Any]) -> ProductionContract | None:
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
    backend_version = _pinned_text(production, "backend_version_or_commit", "production")
    model = _required_text(production, "model", "production")
    if "/" not in model:
        raise ProductionContractError("production.model must use <provider>/<model> form")
    provider_name, _ = model.removesuffix(":vision").split("/", 1)
    if provider_name != backend:
        raise ProductionContractError("production.backend must equal the provider prefix of production.model")

    model_cfg = _mapping(production.get("model_artifact"), "production.model_artifact")
    model_upstream = _required_text(model_cfg, "upstream", "production.model_artifact")
    artifact = _required_text(model_cfg, "artifact", "production.model_artifact")
    revision = _pinned_text(model_cfg, "revision", "production.model_artifact")
    sha256 = _required_text(model_cfg, "sha256", "production.model_artifact")
    if not _SHA256_RE.fullmatch(sha256):
        raise ProductionContractError("production.model_artifact.sha256 must be a 64-character SHA-256 digest")
    quantization = _required_text(model_cfg, "quantization_or_dtype", "production.model_artifact")
    context_window = _positive_int(model_cfg, "context_window_tokens", "production.model_artifact")
    artifact_path_value = model_cfg.get("verify_path")
    if artifact_path_value is not None and (not isinstance(artifact_path_value, str) or not artifact_path_value.strip()):
        raise ProductionContractError("production.model_artifact.verify_path must be a non-empty string when set")
    artifact_path = artifact_path_value.strip() if isinstance(artifact_path_value, str) else None

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
    if gpu_required:
        gpu_device_index = gpu.get("device_index")
        min_vram_used_mib = gpu.get("min_vram_used_mib")
        if not isinstance(gpu_device_index, int) or isinstance(gpu_device_index, bool) or gpu_device_index < 0:
            raise ProductionContractError("production.gpu.device_index must be a non-negative integer")
        if not isinstance(min_vram_used_mib, int) or isinstance(min_vram_used_mib, bool) or min_vram_used_mib <= 0:
            raise ProductionContractError("production.gpu.min_vram_used_mib must be a positive integer")

    providers = _mapping(config.get("providers"), "providers")
    provider = _mapping(providers.get(backend), f"providers.{backend}")
    configured_endpoint = _required_text(provider, "base_url", f"providers.{backend}")
    if _normalize_endpoint(configured_endpoint) != endpoint:
        raise ProductionContractError(
            f"providers.{backend}.base_url must exactly match production.network.endpoint"
        )

    models = _mapping(config.get("models"), "models")
    if model not in models:
        raise ProductionContractError("production.model must exist in models")
    first_model = next(iter(models), None)
    if first_model != model:
        raise ProductionContractError(
            "production.model must be the first configured model because llmcord currently selects the first model at startup"
        )
    model_runtime_cfg = _mapping(models.get(model) or {}, f"models.{model}")
    context_cfg = _mapping(model_runtime_cfg.get("context_management"), f"models.{model}.context_management")
    runtime_context_window = context_cfg.get("context_window_tokens")
    if runtime_context_window not in {"auto", context_window}:
        raise ProductionContractError(
            f"models.{model}.context_management.context_window_tokens must be 'auto' or match production context_window_tokens"
        )

    health = _mapping(config.get("health_control"), "health_control")
    if health.get("enabled") is not True:
        raise ProductionContractError("health_control.enabled must be true in production")
    if health.get("model") != model:
        raise ProductionContractError("health_control.model must exactly match production.model")
    health_restart = tuple(health.get("restart_command") or ())
    if health_restart != restart_command:
        raise ProductionContractError(
            "health_control.restart_command must exactly match production.supervisor.restart_command"
        )

    gpu_health = _mapping(health.get("nvidia_gpu") or {}, "health_control.nvidia_gpu")
    if gpu_required:
        if gpu_health.get("enabled") is not True:
            raise ProductionContractError("health_control.nvidia_gpu.enabled must be true when production GPU is required")
        if gpu_health.get("device_index") != gpu_device_index:
            raise ProductionContractError("health_control.nvidia_gpu.device_index must match production.gpu.device_index")
        if gpu_health.get("min_vram_used_mib") != min_vram_used_mib:
            raise ProductionContractError(
                "health_control.nvidia_gpu.min_vram_used_mib must match production.gpu.min_vram_used_mib"
            )

    contract = ProductionContract(
        backend=backend,
        backend_version_or_commit=backend_version,
        model=model,
        model_upstream=model_upstream,
        model_artifact=artifact,
        model_revision=revision,
        model_sha256=sha256.lower(),
        quantization_or_dtype=quantization,
        context_window_tokens=context_window,
        network_mode=network_mode,
        network_endpoint=endpoint,
        supervisor_kind=supervisor_kind,
        restart_command=restart_command,
        gpu_required=gpu_required,
        gpu_device_index=gpu_device_index,
        min_vram_used_mib=min_vram_used_mib,
        artifact_path=artifact_path,
    )

    if artifact_path is not None:
        verify_model_artifact(contract)

    return contract


def verify_model_artifact(contract: ProductionContract) -> None:
    if contract.artifact_path is None:
        return
    path = Path(contract.artifact_path)
    if not path.is_file():
        raise ProductionContractError(f"production model artifact is not accessible: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != contract.model_sha256:
        raise ProductionContractError(
            f"production model artifact hash mismatch: expected {contract.model_sha256}, got {actual}"
        )
