from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml


class ProductionContractError(RuntimeError):
    """Raised when the production deployment contract is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ProductionContractError("config root must be a mapping")
    return config


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ProductionContractError(f"{key} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProductionContractError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProductionContractError(f"{name} must be a non-negative integer")
    return value


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionContractError(f"{name} must be a non-empty string")
    return value.strip()


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    production = _mapping(config, "production")
    deployment = _mapping(production, "deployment")
    backend = _mapping(production, "backend")
    model = _mapping(production, "model")
    gpu = _mapping(production, "gpu")

    if production.get("schema_version") != 1:
        raise ProductionContractError("production.schema_version must be 1")

    if deployment.get("platform") != "wsl2-native":
        raise ProductionContractError("production platform must be wsl2-native")
    if deployment.get("network_scope") != "loopback":
        raise ProductionContractError("production network_scope must be loopback")
    if deployment.get("api_host") != "127.0.0.1":
        raise ProductionContractError("production api_host must be 127.0.0.1")
    api_port = _positive_int(deployment.get("api_port"), "production.deployment.api_port")
    if api_port > 65535:
        raise ProductionContractError("production.deployment.api_port must be <= 65535")
    if deployment.get("supervisor") != "systemd-user":
        raise ProductionContractError("production supervisor must be systemd-user")
    backend_service = _non_empty_string(
        deployment.get("backend_service"), "production.deployment.backend_service"
    )
    if backend_service != "llmcord-llama-server.service":
        raise ProductionContractError("backend service authority must be llmcord-llama-server.service")

    if backend.get("name") != "llama.cpp":
        raise ProductionContractError("production backend must be llama.cpp")
    version = _non_empty_string(backend.get("version"), "production.backend.version")
    if version in {"latest", "main", "master"}:
        raise ProductionContractError("production backend version must be pinned")
    commit = _non_empty_string(backend.get("commit"), "production.backend.commit")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ProductionContractError("production backend commit must be a full 40-character SHA")
    _non_empty_string(backend.get("binary"), "production.backend.binary")
    context_window = _positive_int(
        backend.get("context_window_tokens"), "production.backend.context_window_tokens"
    )
    parallel = _positive_int(backend.get("parallel"), "production.backend.parallel")
    if parallel != 1:
        raise ProductionContractError("production backend parallel must be 1")
    if backend.get("n_gpu_layers") != "all":
        raise ProductionContractError("production backend must require all model layers on GPU")

    model_key = _non_empty_string(production.get("model_key"), "production.model_key")
    if not model_key.startswith("llamacpp/"):
        raise ProductionContractError("production.model_key must use the llamacpp provider")
    model_alias = _non_empty_string(model.get("alias"), "production.model.alias")
    if model_key != f"llamacpp/{model_alias}":
        raise ProductionContractError("production.model_key must match production.model.alias")
    _non_empty_string(model.get("upstream"), "production.model.upstream")
    _non_empty_string(model.get("artifact_repo"), "production.model.artifact_repo")
    revision = _non_empty_string(model.get("revision"), "production.model.revision")
    if revision in {"main", "master", "latest"} or re.fullmatch(r"[0-9a-f]{7,40}", revision) is None:
        raise ProductionContractError("production model revision must be a pinned git revision")
    filename = _non_empty_string(model.get("filename"), "production.model.filename")
    sha256 = _non_empty_string(model.get("sha256"), "production.model.sha256")
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ProductionContractError("production model sha256 must be 64 lowercase hex characters")
    artifact_url = _non_empty_string(model.get("artifact_url"), "production.model.artifact_url")
    if f"/resolve/{revision}/{filename}" not in artifact_url:
        raise ProductionContractError("production artifact_url must contain the pinned revision and filename")
    if model.get("quantization") != "Q6_K":
        raise ProductionContractError("production model quantization must be Q6_K")
    _non_empty_string(model.get("path"), "production.model.path")

    if gpu.get("vendor") != "nvidia":
        raise ProductionContractError("production GPU vendor must be nvidia")
    _non_negative_int(gpu.get("device_index"), "production.gpu.device_index")
    _positive_int(gpu.get("min_vram_used_mib"), "production.gpu.min_vram_used_mib")

    providers = _mapping(config, "providers")
    if set(providers) != {"llamacpp"}:
        raise ProductionContractError("production config must expose only the llamacpp provider")
    endpoint = f"http://127.0.0.1:{api_port}/v1"
    if _mapping(providers, "llamacpp").get("base_url") != endpoint:
        raise ProductionContractError("llamacpp base_url must match the production loopback endpoint")

    models = _mapping(config, "models")
    if set(models) != {model_key}:
        raise ProductionContractError("production config must expose exactly the pinned model")
    model_runtime = _mapping(models, model_key)
    context = _mapping(model_runtime, "context_management")
    if context.get("context_window_tokens") != context_window:
        raise ProductionContractError("context_management context window must match backend ctx-size")
    max_output = _positive_int(context.get("max_output_tokens"), "max_output_tokens")
    safety_margin = _positive_int(context.get("safety_margin_tokens"), "safety_margin_tokens")
    trigger = _positive_int(context.get("compaction_trigger_tokens"), "compaction_trigger_tokens")
    target = _positive_int(context.get("compaction_target_tokens"), "compaction_target_tokens")
    _positive_int(context.get("recent_messages"), "recent_messages")
    _positive_int(context.get("compaction_max_output_tokens"), "compaction_max_output_tokens")
    hard_input_limit = context_window - max_output - safety_margin
    if hard_input_limit <= 0:
        raise ProductionContractError("context output reservation leaves no input capacity")
    if trigger >= hard_input_limit:
        raise ProductionContractError("compaction trigger must be below the hard input limit")
    if target >= trigger:
        raise ProductionContractError("compaction target must be below the trigger")

    health = _mapping(config, "health_control")
    if health.get("enabled") is not True:
        raise ProductionContractError("production health_control must be enabled")
    if health.get("model") != model_key:
        raise ProductionContractError("health_control.model must match production.model_key")
    expected_restart = ["systemctl", "--user", "restart", backend_service]
    if health.get("restart_command") != expected_restart:
        raise ProductionContractError("watchdog restart_command must use the systemd user backend service")
    health_gpu = _mapping(health, "nvidia_gpu")
    if health_gpu.get("enabled") is not True:
        raise ProductionContractError("production NVIDIA health check must be enabled")
    if health_gpu.get("device_index") != gpu.get("device_index"):
        raise ProductionContractError("health GPU device must match production GPU device")
    if health_gpu.get("min_vram_used_mib") != gpu.get("min_vram_used_mib"):
        raise ProductionContractError("health GPU VRAM floor must match production GPU contract")

    return production


def expand_path(value: str) -> Path:
    return Path(os.path.expanduser(value)).resolve()


def backend_argv(config: dict[str, Any]) -> list[str]:
    production = validate_config(config)
    deployment = production["deployment"]
    backend = production["backend"]
    model = production["model"]
    binary = str(expand_path(backend["binary"]))
    model_path = str(expand_path(model["path"]))
    return [
        binary,
        "--model",
        model_path,
        "--alias",
        model["alias"],
        "--host",
        deployment["api_host"],
        "--port",
        str(deployment["api_port"]),
        "--ctx-size",
        str(backend["context_window_tokens"]),
        "--parallel",
        str(backend["parallel"]),
        "--n-gpu-layers",
        backend["n_gpu_layers"],
        "--jinja",
    ]


def verify_model_artifact(config: dict[str, Any]) -> None:
    production = validate_config(config)
    model = production["model"]
    path = expand_path(model["path"])
    if not path.is_file():
        raise ProductionContractError(f"production model artifact does not exist: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != model["sha256"]:
        raise ProductionContractError(
            f"production model SHA256 mismatch: expected {model['sha256']}, got {actual}"
        )


def verify_backend_binary(config: dict[str, Any]) -> None:
    production = validate_config(config)
    backend = production["backend"]
    path = expand_path(backend["binary"])
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ProductionContractError(f"production llama-server binary is not executable: {path}")

    try:
        completed = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProductionContractError("could not execute production llama-server --version") from exc

    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise ProductionContractError("production llama-server --version returned non-zero")
    if backend["version"].removeprefix("v") not in output:
        raise ProductionContractError("production llama-server version does not match pinned version")
    if backend["commit"][:9] not in output:
        raise ProductionContractError("production llama-server commit does not match pinned commit")


def check(config_path: str | Path, *, offline: bool) -> None:
    config = load_config(config_path)
    production = validate_config(config)
    if not offline:
        verify_backend_binary(config)
        verify_model_artifact(config)
    print(
        json.dumps(
            {
                "backend": production["backend"]["name"],
                "backend_version": production["backend"]["version"],
                "backend_commit": production["backend"]["commit"],
                "model": production["model_key"],
                "model_sha256": production["model"]["sha256"],
                "endpoint": f"http://{production['deployment']['api_host']}:{production['deployment']['api_port']}/v1",
                "supervisor": production["deployment"]["supervisor"],
                "offline": offline,
            },
            sort_keys=True,
        )
    )


def serve(config_path: str | Path) -> None:
    config = load_config(config_path)
    verify_backend_binary(config)
    verify_model_artifact(config)
    argv = backend_argv(config)
    os.execv(argv[0], argv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and start the pinned llmcord production backend")
    parser.add_argument("command", choices=("check", "serve"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--offline", action="store_true", help="validate static contract only")
    args = parser.parse_args()

    try:
        if args.command == "check":
            check(args.config, offline=args.offline)
        else:
            if args.offline:
                raise ProductionContractError("--offline is only valid with check")
            serve(args.config)
    except ProductionContractError as exc:
        raise SystemExit(f"production contract error: {exc}") from exc


if __name__ == "__main__":
    main()
