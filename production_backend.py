from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from production_contract import (
    ProductionContractError,
    load_resolved_config,
    validate_production_contract,
    verify_backend_executable,
)


def backend_argv() -> list[str]:
    config = load_resolved_config("config.yaml")
    contract = validate_production_contract(config, verify_artifact=True)
    if contract is None:
        raise ProductionContractError("production contract must be enabled for backend startup")
    if contract.backend != "llamacpp":
        raise ProductionContractError("production backend launcher currently requires llamacpp")
    if contract.network_mode != "wsl":
        raise ProductionContractError("production backend launcher requires WSL network mode")
    if contract.supervisor_kind != "systemd":
        raise ProductionContractError("production backend launcher requires systemd supervisor")

    verify_backend_executable(contract)
    endpoint = urlparse(contract.network_endpoint)
    if endpoint.hostname is None or endpoint.port is None:
        raise ProductionContractError("production endpoint must include host and port")

    executable = str(Path(os.path.expanduser(contract.backend_executable)).resolve())
    artifact = str(Path(os.path.expanduser(contract.artifact_path)).resolve())
    model_alias = contract.model.split("/", 1)[1].removesuffix(":vision")
    return [
        executable,
        "--model",
        artifact,
        "--alias",
        model_alias,
        "--host",
        endpoint.hostname,
        "--port",
        str(endpoint.port),
        "--ctx-size",
        str(contract.context_window_tokens),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "all",
        "--jinja",
    ]


def main() -> None:
    try:
        argv = backend_argv()
    except ProductionContractError as exc:
        raise SystemExit(f"production backend contract error: {exc}") from exc
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
