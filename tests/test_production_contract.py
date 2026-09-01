from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest

from production_contract import ProductionContractError, validate_production_contract


class ProductionContractTests(unittest.TestCase):
    def config(self):
        return {
            "production": {
                "enabled": True,
                "backend": "llamacpp",
                "backend_version_or_commit": "b1234567890abcdef",
                "model": "llamacpp/local-model",
                "model_artifact": {
                    "upstream": "example/model",
                    "artifact": "model-Q4_K_M.gguf",
                    "revision": "0123456789abcdef",
                    "sha256": "a" * 64,
                    "quantization_or_dtype": "Q4_K_M",
                    "context_window_tokens": 32768,
                },
                "network": {
                    "mode": "native",
                    "endpoint": "http://127.0.0.1:8080/v1",
                },
                "supervisor": {
                    "kind": "systemd",
                    "restart_command": ["systemctl", "restart", "llama-server.service"],
                },
                "gpu": {
                    "required": True,
                    "device_index": 0,
                    "min_vram_used_mib": 12000,
                    "process_name_pattern": r"llama-server(?:\.exe)?$",
                },
            },
            "providers": {
                "llamacpp": {"base_url": "http://127.0.0.1:8080/v1"},
            },
            "models": {
                "llamacpp/local-model": {
                    "context_management": {
                        "context_window_tokens": "auto",
                    }
                },
                "openai/other": {},
            },
            "health_control": {
                "enabled": True,
                "model": "llamacpp/local-model",
                "restart_command": ["systemctl", "restart", "llama-server.service"],
                "nvidia_gpu": {
                    "enabled": True,
                    "device_index": 0,
                    "min_vram_used_mib": 12000,
                },
            },
        }

    def test_valid_contract_is_accepted(self):
        contract = validate_production_contract(self.config())
        self.assertIsNotNone(contract)
        self.assertEqual("llamacpp", contract.backend)
        self.assertEqual("llamacpp/local-model", contract.model)
        self.assertTrue(contract.gpu_required)
        self.assertEqual(r"llama-server(?:\.exe)?$", contract.gpu_process_name_pattern)

    def test_disabled_contract_is_noop(self):
        config = self.config()
        config["production"]["enabled"] = False
        self.assertIsNone(validate_production_contract(config))

    def test_floating_backend_version_is_rejected(self):
        config = self.config()
        config["production"]["backend_version_or_commit"] = "latest"
        with self.assertRaisesRegex(ProductionContractError, "must be pinned"):
            validate_production_contract(config)

    def test_docker_localhost_is_rejected(self):
        config = self.config()
        config["production"]["network"]["mode"] = "docker"
        config["production"]["network"]["endpoint"] = "http://localhost:8080/v1"
        config["providers"]["llamacpp"]["base_url"] = "http://localhost:8080/v1"
        with self.assertRaisesRegex(ProductionContractError, "must not use localhost"):
            validate_production_contract(config)

    def test_projection_mismatch_is_rejected(self):
        config = self.config()
        config["health_control"]["restart_command"] = ["docker", "restart", "backend"]
        with self.assertRaisesRegex(ProductionContractError, "restart_command must exactly match"):
            validate_production_contract(config)

    def test_selected_model_must_be_startup_model(self):
        config = self.config()
        config["models"] = {
            "openai/other": {},
            "llamacpp/local-model": config["models"]["llamacpp/local-model"],
        }
        with self.assertRaisesRegex(ProductionContractError, "must be the first configured model"):
            validate_production_contract(config)

    def test_gpu_required_requires_process_identity(self):
        config = self.config()
        del config["production"]["gpu"]["process_name_pattern"]
        with self.assertRaisesRegex(ProductionContractError, "process_name_pattern"):
            validate_production_contract(config)

    def test_invalid_gpu_process_pattern_is_rejected(self):
        config = self.config()
        config["production"]["gpu"]["process_name_pattern"] = "["
        with self.assertRaisesRegex(ProductionContractError, "process_name_pattern is invalid"):
            validate_production_contract(config)

    def test_accessible_artifact_hash_is_verified(self):
        config = self.config()
        payload = b"fixed model artifact fixture"
        with tempfile.NamedTemporaryFile() as file:
            file.write(payload)
            file.flush()
            config["production"]["model_artifact"]["verify_path"] = file.name
            config["production"]["model_artifact"]["sha256"] = hashlib.sha256(payload).hexdigest()
            contract = validate_production_contract(config)
            self.assertEqual(file.name, contract.artifact_path)

            bad = copy.deepcopy(config)
            bad["production"]["model_artifact"]["sha256"] = "b" * 64
            with self.assertRaisesRegex(ProductionContractError, "hash mismatch"):
                validate_production_contract(bad)


if __name__ == "__main__":
    unittest.main()
