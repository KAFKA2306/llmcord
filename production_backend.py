from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from production_contract import (
    ProductionContract,
    ProductionContractError,
    load_resolved_config,
    validate_production_contract,
    verify_backend_executable,
)


def load_contract(*, verify_artifact: bool) -> ProductionContract:
    config = load_resolved_config("config.yaml")
    contract = validate_production_contract(config, verify_artifact=verify_artifact)
    if contract is None:
        raise ProductionContractError("production contract must be enabled")
    if contract.backend != "llamacpp":
        raise ProductionContractError("production backend launcher currently requires llamacpp")
    if contract.network_mode != "wsl":
        raise ProductionContractError("production backend launcher requires WSL network mode")
    if contract.supervisor_kind != "systemd":
        raise ProductionContractError("production backend launcher requires systemd supervisor")
    return contract


def backend_argv(contract: ProductionContract) -> list[str]:
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


def check(*, static: bool) -> None:
    contract = load_contract(verify_artifact=not static)
    if not static:
        verify_backend_executable(contract)
    print(json.dumps({
        "backend": contract.backend,
        "backend_release": contract.backend_release,
        "backend_commit": contract.backend_version_or_commit,
        "model": contract.model,
        "model_sha256": contract.model_sha256,
        "network_mode": contract.network_mode,
        "endpoint": contract.network_endpoint,
        "supervisor": contract.supervisor_kind,
        "static": static,
    }, sort_keys=True))


def serve() -> None:
    contract = load_contract(verify_artifact=True)
    verify_backend_executable(contract)
    argv = backend_argv(contract)
    os.execv(argv[0], argv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or start the pinned llmcord production backend")
    parser.add_argument("command", choices=("check", "serve"))
    parser.add_argument("--static", action="store_true", help="skip local binary/model file verification")
    args = parser.parse_args()
    try:
        if args.command == "check":
            check(static=args.static)
        elif args.static:
            raise ProductionContractError("--static is only valid with check")
        else:
            serve()
    except ProductionContractError as exc:
        raise SystemExit(f"production backend contract error: {exc}") from exc


if __name__ == "__main__":
    main()
