from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from production_runtime import (
    ProductionContractError,
    backend_argv,
    load_config,
    validate_config,
    verify_model_artifact,
)


class ProductionRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("config.yaml")

    def test_repository_config_is_one_pinned_production_contract(self) -> None:
        production = validate_config(self.config)
        self.assertEqual("llama.cpp", production["backend"]["name"])
        self.assertEqual("v0.3.0", production["backend"]["version"])
        self.assertEqual(
            "c1d0e7a004015f23bc0233470b747b596f29b264",
            production["backend"]["commit"],
        )
        self.assertEqual("llamacpp/ornith-1.5-9b-q6_k", production["model_key"])
        self.assertEqual(
            "b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a",
            production["model"]["sha256"],
        )
        self.assertEqual("wsl2-native", production["deployment"]["platform"])
        self.assertEqual("systemd-user", production["deployment"]["supervisor"])
        self.assertEqual(0, self.config["max_images"])

    def test_backend_command_is_loopback_single_slot_and_pinned_context(self) -> None:
        argv = backend_argv(self.config)
        self.assertEqual("127.0.0.1", argv[argv.index("--host") + 1])
        self.assertEqual("8080", argv[argv.index("--port") + 1])
        self.assertEqual("32768", argv[argv.index("--ctx-size") + 1])
        self.assertEqual("1", argv[argv.index("--parallel") + 1])
        self.assertEqual("all", argv[argv.index("--n-gpu-layers") + 1])
        self.assertTrue(argv[argv.index("--model") + 1].endswith("Ornith-1.5-9B-Q6_K.gguf"))

    def test_public_or_unpinned_network_and_artifacts_fail_loudly(self) -> None:
        for mutator in (
            lambda c: c["production"]["deployment"].__setitem__("api_host", "0.0.0.0"),
            lambda c: c["production"]["backend"].__setitem__("version", "latest"),
            lambda c: c["production"]["model"].__setitem__("revision", "main"),
            lambda c: c.__setitem__("max_images", 1),
        ):
            candidate = copy.deepcopy(self.config)
            mutator(candidate)
            with self.assertRaises(ProductionContractError):
                validate_config(candidate)

    def test_runtime_provider_model_and_watchdog_must_match_production_authority(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["providers"]["ollama"] = {"base_url": "http://127.0.0.1:11434/v1"}
        with self.assertRaises(ProductionContractError):
            validate_config(candidate)

        candidate = copy.deepcopy(self.config)
        candidate["health_control"]["restart_command"] = ["pkill", "llama-server"]
        with self.assertRaises(ProductionContractError):
            validate_config(candidate)

        candidate = copy.deepcopy(self.config)
        candidate["models"]["llamacpp/ornith-1.5-9b-q6_k"]["context_management"]["context_window_tokens"] = 65536
        with self.assertRaises(ProductionContractError):
            validate_config(candidate)

    def test_model_hash_verification_uses_exact_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "Ornith-1.5-9B-Q6_K.gguf"
            artifact.write_bytes(b"canonical model bytes")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

            candidate = copy.deepcopy(self.config)
            candidate["production"]["model"]["path"] = str(artifact)
            candidate["production"]["model"]["sha256"] = digest
            verify_model_artifact(candidate)

            candidate["production"]["model"]["sha256"] = "0" * 64
            with self.assertRaises(ProductionContractError):
                verify_model_artifact(candidate)

    def test_backend_binary_and_model_paths_are_not_container_specific(self) -> None:
        production = validate_config(self.config)
        binary = production["backend"]["binary"]
        model_path = production["model"]["path"]
        self.assertTrue(binary.startswith("~/"))
        self.assertTrue(model_path.startswith("~/"))
        self.assertNotIn("/app/", binary)
        self.assertNotIn("/app/", model_path)


if __name__ == "__main__":
    unittest.main()
