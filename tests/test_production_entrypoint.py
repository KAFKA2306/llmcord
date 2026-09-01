from __future__ import annotations

import os
import unittest

from health_control import GPU_PROCESS_PATTERN_ENV
from production_contract import ProductionContract
from production_entrypoint import configure_runtime_environment


class ProductionEntrypointTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(GPU_PROCESS_PATTERN_ENV, None)

    def contract(self, *, gpu_required=True, pattern=r"llama-server(?:\.exe)?$"):
        return ProductionContract(
            backend="llamacpp",
            backend_release="v0.3.0",
            backend_version_or_commit="c1d0e7a004015f23bc0233470b747b596f29b264",
            backend_executable="/opt/llama/bin/llama-server",
            model="llamacpp/local-model",
            model_upstream="example/model",
            artifact_repo="example/model-GGUF",
            model_artifact="model-Q6_K.gguf",
            model_revision="0123456",
            artifact_url="https://huggingface.co/example/model-GGUF/resolve/0123456/model-Q6_K.gguf",
            model_sha256="a" * 64,
            quantization_or_dtype="Q6_K",
            context_window_tokens=32768,
            network_mode="wsl",
            network_endpoint="http://127.0.0.1:8080/v1",
            supervisor_kind="systemd",
            restart_command=("systemctl", "--user", "restart", "llmcord-llama-server.service"),
            gpu_required=gpu_required,
            gpu_device_index=0 if gpu_required else None,
            min_vram_used_mib=6000 if gpu_required else None,
            gpu_process_name_pattern=pattern if gpu_required else None,
            artifact_path="/models/model-Q6_K.gguf",
        )

    def test_production_gpu_process_pattern_is_projected(self):
        configure_runtime_environment(self.contract())
        self.assertEqual(r"llama-server(?:\.exe)?$", os.environ[GPU_PROCESS_PATTERN_ENV])

    def test_disabled_production_removes_stale_pattern(self):
        os.environ[GPU_PROCESS_PATTERN_ENV] = "stale"
        configure_runtime_environment(None)
        self.assertNotIn(GPU_PROCESS_PATTERN_ENV, os.environ)

    def test_cpu_only_contract_removes_stale_pattern(self):
        os.environ[GPU_PROCESS_PATTERN_ENV] = "stale"
        configure_runtime_environment(self.contract(gpu_required=False, pattern=None))
        self.assertNotIn(GPU_PROCESS_PATTERN_ENV, os.environ)


if __name__ == "__main__":
    unittest.main()
