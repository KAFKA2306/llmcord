from __future__ import annotations

import logging
import os
from pathlib import Path
import runpy

from health_control import GPU_PROCESS_PATTERN_ENV
from observability import emit_event, install_structured_logging
from production_contract import ProductionContract, load_resolved_config, validate_production_contract


def configure_runtime_environment(contract: ProductionContract | None) -> None:
    if contract is not None and contract.gpu_required and contract.gpu_process_name_pattern:
        os.environ[GPU_PROCESS_PATTERN_ENV] = contract.gpu_process_name_pattern
    else:
        os.environ.pop(GPU_PROCESS_PATTERN_ENV, None)


def main() -> None:
    config_path = Path("config.yaml")
    config = load_resolved_config(str(config_path))
    contract = validate_production_contract(config)
    configure_runtime_environment(contract)

    install_structured_logging(
        provider=contract.backend if contract is not None else None,
        model=contract.model if contract is not None else None,
    )

    if contract is None:
        emit_event("production.startup", contract_enabled=False)
        logging.info("production contract disabled; starting llmcord without production validation")
    else:
        emit_event(
            "production.startup",
            contract_enabled=True,
            network_mode=contract.network_mode,
            supervisor=contract.supervisor_kind,
            gpu_required=contract.gpu_required,
        )
        logging.info(
            "production contract validated backend=%s model=%s network_mode=%s supervisor=%s gpu_required=%s",
            contract.backend,
            contract.model,
            contract.network_mode,
            contract.supervisor_kind,
            contract.gpu_required,
        )

    runpy.run_path(str(Path(__file__).with_name("llmcord.py")), run_name="__main__")


if __name__ == "__main__":
    main()
