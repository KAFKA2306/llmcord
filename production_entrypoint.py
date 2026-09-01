from __future__ import annotations

import logging
import os
from pathlib import Path
import runpy

from health_control import GPU_PROCESS_PATTERN_ENV
from production_contract import ProductionContract, load_resolved_config, validate_production_contract


def configure_runtime_environment(contract: ProductionContract | None) -> None:
    if contract is not None and contract.gpu_required and contract.gpu_process_name_pattern:
        os.environ[GPU_PROCESS_PATTERN_ENV] = contract.gpu_process_name_pattern
    else:
        os.environ.pop(GPU_PROCESS_PATTERN_ENV, None)


def main() -> None:
    config_path = Path("config.yaml")
    config = load_resolved_config(str(config_path))
    # Backend startup owns expensive binary/model artifact verification. The bot verifies
    # the canonical manifest and execution projections without hashing the GGUF again.
    contract = validate_production_contract(config, verify_artifact=False)
    configure_runtime_environment(contract)

    if contract is None:
        logging.info("production contract disabled; starting llmcord without production validation")
    else:
        logging.info(
            "production contract validated backend=%s release=%s model=%s network_mode=%s supervisor=%s gpu_required=%s",
            contract.backend,
            contract.backend_release,
            contract.model,
            contract.network_mode,
            contract.supervisor_kind,
            contract.gpu_required,
        )

    runpy.run_path(str(Path(__file__).with_name("llmcord.py")), run_name="__main__")


if __name__ == "__main__":
    main()
