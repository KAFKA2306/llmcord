from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from production_contract import (
    ProductionContractError,
    load_resolved_config,
    validate_production_contract,
)


class ProductionContractTests(unittest.TestCase):
    def config(self):
        return {
            "max_images": 0,
            "runtime_control": {"max_concurrency": 1},
            "production": {
                "enabled": True,
                "backend": "llamacpp",
                "backend_release": "v0.3.0",
                "backend_version_or_commit": "c1d0e7a004015f23bc0233470b747b596f29b264",
                "backend_executable": "/opt/llama/bin/llama-server",
                "model": "llamacpp/local-model",
                "model_artifact": {
                    "upstream": "example/model",
                    "artifact_repo": "example/model-GGUF",
                    "artifact": "model-Q6_K.gguf",
                    "revision": "0123456",
                    "artifact_url": "https://huggingface.co/example/model-GGUF/resolve/0123456/model-Q6_K.gguf",
                    "sha256": "a" * 64,
                    "quantization_or_dtype": "Q6_K",
                    "context_window_tokens": 32768,
                    "verify_path": "/models/model-Q6_K.gguf",
                },
                "network": {"mode": "wsl", "endpoint": "http://127.0.0.1:8080/v1"},
                "supervisor": {
                    "kind": "systemd",
                    "restart_command": ["systemctl", "--user", "restart", "llmcord-llama-server.service"],
                },
                "gpu": {
                    "required": True,
                    "device_index": 0,
                    "min_vram_used_mib": 6000,
                    "process_name_pattern": r"llama-server(?:\.exe)?$",
                },
            },
            "providers": {"llamacpp": {"base_url": "http://127.0.0.1:8080/v1"}},
            "models": {
                "llamacpp/local-model": {
                    "context_management": {"context_window_tokens": 32768}
                }
            },
            "health_control": {
                "enabled": True,
                "model": "llamacpp/local-model",
                "restart_command": ["systemctl", "--user", "restart", "llmcord-llama-server.service"],
                "nvidia_gpu": {"enabled": True, "device_index": 0, "min_vram_used_mib": 6000},
            },
        }

    def test_repository_config_is_the_pinned_contract(self):
        contract = validate_production_contract(load_resolved_config(), verify_artifact=False)
        self.assertIsNotNone(contract)
        self.assertEqual("llamacpp", contract.backend)
        self.assertEqual("v0.3.0", contract.backend_release)
        self.assertEqual("c1d0e7a004015f23bc0233470b747b596f29b264", contract.backend_version_or_commit)
        self.assertEqual("llamacpp/ornith-1.5-9b-q6_k", contract.model)
        self.assertEqual("ornith-ai/Ornith-1.5-9B-GGUF", contract.artifact_repo)
        self.assertEqual("b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a", contract.model_sha256)
        self.assertEqual("wsl", contract.network_mode)
        self.assertEqual("http://127.0.0.1:8080/v1", contract.network_endpoint)

    def test_valid_contract_static_projection_is_accepted(self):
        contract = validate_production_contract(self.config(), verify_artifact=False)
        self.assertIsNotNone(contract)
        self.assertEqual(r"llama-server(?:\.exe)?$", contract.gpu_process_name_pattern)

    def test_disabled_contract_is_noop(self):
        config = self.config()
        config["production"]["enabled"] = False
        self.assertIsNone(validate_production_contract(config))

    def test_floating_or_partial_backend_version_is_rejected(self):
        for value in ("latest", "c1d0e7a"):
            config = self.config()
            config["production"]["backend_version_or_commit"] = value
            with self.assertRaises(ProductionContractError):
                validate_production_contract(config, verify_artifact=False)

    def test_network_provider_model_and_context_drift_fail_loudly(self):
        mutations = [
            lambda c: c["production"]["network"].__setitem__("endpoint", "http://0.0.0.0:8080/v1"),
            lambda c: c["providers"].__setitem__("ollama", {"base_url": "http://127.0.0.1:11434/v1"}),
            lambda c: c["models"].__setitem__("openai/other", {}),
            lambda c: c["models"]["llamacpp/local-model"]["context_management"].__setitem__("context_window_tokens", 65536),
            lambda c: c.__setitem__("max_images", 1),
        ]
        for mutate in mutations:
            config = copy.deepcopy(self.config())
            mutate(config)
            with self.assertRaises(ProductionContractError):
                validate_production_contract(config, verify_artifact=False)

    def test_supervisor_and_gpu_projection_must_match(self):
        config = self.config()
        config["health_control"]["restart_command"] = ["pkill", "llama-server"]
        with self.assertRaisesRegex(ProductionContractError, "restart_command"):
            validate_production_contract(config, verify_artifact=False)

        config = self.config()
        config["production"]["gpu"]["process_name_pattern"] = "python$"
        with self.assertRaisesRegex(ProductionContractError, "must match"):
            validate_production_contract(config, verify_artifact=False)

    def test_artifact_source_and_path_are_pinned(self):
        config = self.config()
        config["production"]["model_artifact"]["artifact_url"] = "https://example.invalid/model.gguf"
        with self.assertRaisesRegex(ProductionContractError, "artifact_url"):
            validate_production_contract(config, verify_artifact=False)

        config = self.config()
        config["production"]["model_artifact"]["verify_path"] = "/models/other.gguf"
        with self.assertRaisesRegex(ProductionContractError, "verify_path"):
            validate_production_contract(config, verify_artifact=False)

    def test_accessible_artifact_hash_is_verified(self):
        config = self.config()
        payload = b"fixed model artifact fixture"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model-Q6_K.gguf"
            path.write_bytes(payload)
            config["production"]["model_artifact"]["verify_path"] = str(path)
            config["production"]["model_artifact"]["sha256"] = hashlib.sha256(payload).hexdigest()
            contract = validate_production_contract(config)
            self.assertEqual(str(path), contract.artifact_path)

            bad = copy.deepcopy(config)
            bad["production"]["model_artifact"]["sha256"] = "b" * 64
            with self.assertRaisesRegex(ProductionContractError, "hash mismatch"):
                validate_production_contract(bad)

    def test_no_checked_in_docker_supervisor_path(self):
        self.assertFalse(Path("Dockerfile").exists())
        self.assertFalse(Path("docker-compose.yaml").exists())
        self.assertTrue(Path("ops/systemd/llmcord-llama-server.service").is_file())
        self.assertTrue(Path("ops/systemd/llmcord.service").is_file())


if __name__ == "__main__":
    unittest.main()
