from __future__ import annotations

import logging
import os
from pathlib import Path
import runpy

from health_control import GPU_PROCESS_PATTERN_ENV
from production_contract import load_resolved_config, validate_production_contract


def main() -> None:
    config_path = Path("config.yaml")
    config = load_resolved_config(str(config_path))
    contract = validate_production_contract(config)

    if contract is None:
        os.environ.pop(GPU_PROCESS_PATTERN_ENV, None)
        logging.info("production contract disabled; starting llmcord without production validation")
    else:
        if contract.gpu_required and contract.gpu_process_name_pattern:
            os.environ[GPU_PROCESS_PATTERN_ENV] = contract.gpu_process_name_pattern
        else:
            os.environ.pop(GPU_PROCESS_PATTERN_ENV, None)

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
