from __future__ import annotations

import unittest

from production_backend import backend_argv
from production_contract import ProductionContract


class ProductionBackendTests(unittest.TestCase):
    def contract(self) -> ProductionContract:
        return ProductionContract(
            backend="llamacpp",
            backend_release="v0.3.0",
            backend_version_or_commit="c1d0e7a004015f23bc0233470b747b596f29b264",
            backend_executable="/opt/llama/bin/llama-server",
            model="llamacpp/ornith-1.5-9b-q6_k",
            model_upstream="ornith-ai/Ornith-1.5-9B",
            artifact_repo="ornith-ai/Ornith-1.5-9B-GGUF",
            model_artifact="Ornith-1.5-9B-Q6_K.gguf",
            model_revision="2b651f3",
            artifact_url="https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/2b651f3/Ornith-1.5-9B-Q6_K.gguf",
            model_sha256="b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a",
            quantization_or_dtype="Q6_K",
            context_window_tokens=32768,
            network_mode="wsl",
            network_endpoint="http://127.0.0.1:8080/v1",
            supervisor_kind="systemd",
            restart_command=("systemctl", "--user", "restart", "llmcord-llama-server.service"),
            gpu_required=True,
            gpu_device_index=0,
            min_vram_used_mib=1,
            gpu_process_name_pattern=r"llama-server(?:\.exe)?$",
            artifact_path="/models/Ornith-1.5-9B-Q6_K.gguf",
        )

    def test_backend_argv_is_loopback_single_slot_gpu_only(self):
        argv = backend_argv(self.contract())
        self.assertEqual("/opt/llama/bin/llama-server", argv[0])
        self.assertEqual("127.0.0.1", argv[argv.index("--host") + 1])
        self.assertEqual("8080", argv[argv.index("--port") + 1])
        self.assertEqual("32768", argv[argv.index("--ctx-size") + 1])
        self.assertEqual("1", argv[argv.index("--parallel") + 1])
        self.assertEqual("all", argv[argv.index("--n-gpu-layers") + 1])
        self.assertEqual("ornith-1.5-9b-q6_k", argv[argv.index("--alias") + 1])


if __name__ == "__main__":
    unittest.main()
