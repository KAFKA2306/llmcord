from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from production_contract import ProductionContract
from runtime_evidence import (
    PhaseEvidence,
    ProbeSnapshot,
    RuntimeEvidenceError,
    ServiceSnapshot,
    _wait_until_healthy,
    build_probe_config,
    parse_systemd_show,
)


class RuntimeEvidenceTests(unittest.TestCase):
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

    def config(self):
        return {
            "health_control": {
                "probe_timeout_seconds": 30,
                "probe_expected_text": "PONG",
                "probe_max_tokens": 8,
                "probe_prompt": "Reply exactly: PONG",
                "restart_command_timeout_seconds": 30,
                "nvidia_gpu": {
                    "enabled": True,
                    "device_index": 0,
                    "min_vram_used_mib": 1,
                    "timeout_seconds": 5,
                },
            }
        }

    def test_parse_systemd_show(self):
        snapshot = parse_systemd_show(
            "llmcord.service",
            "ActiveState=active\nSubState=running\nMainPID=1234\nFragmentPath=/home/u/.config/systemd/user/llmcord.service\n",
        )
        self.assertTrue(snapshot.running)
        self.assertEqual(1234, snapshot.main_pid)
        self.assertEqual("active", snapshot.active_state)

    def test_parse_systemd_show_rejects_non_numeric_pid(self):
        with self.assertRaisesRegex(RuntimeEvidenceError, "MainPID"):
            parse_systemd_show(
                "llmcord.service",
                "ActiveState=active\nSubState=running\nMainPID=oops\n",
            )

    def test_probe_uses_backend_alias_not_provider_prefixed_model(self):
        probe = build_probe_config(self.config(), self.contract())
        self.assertEqual("ornith-1.5-9b-q6_k", probe.model)
        self.assertEqual("http://127.0.0.1:8080/v1", probe.base_url)
        self.assertEqual("PONG", probe.expected_text)


class RuntimeEvidenceAsyncTests(unittest.IsolatedAsyncioTestCase):
    def contract(self) -> ProductionContract:
        return RuntimeEvidenceTests().contract()

    def config(self):
        return RuntimeEvidenceTests().config()

    async def test_restart_requires_new_backend_pid_and_preserves_bot_pid(self):
        backend = ServiceSnapshot(
            service="llmcord-llama-server.service",
            active_state="active",
            sub_state="running",
            main_pid=222,
            fragment_path="/home/u/.config/systemd/user/llmcord-llama-server.service",
        )
        bot = ServiceSnapshot(
            service="llmcord.service",
            active_state="active",
            sub_state="running",
            main_pid=999,
            fragment_path="/home/u/.config/systemd/user/llmcord.service",
        )
        with (
            patch("runtime_evidence.service_snapshot", side_effect=[backend, bot]),
            patch("runtime_evidence._gpu_check", new=AsyncMock(return_value=True)),
            patch(
                "runtime_evidence._probe",
                new=AsyncMock(return_value=ProbeSnapshot(True, 0.2, None, None)),
            ),
        ):
            evidence = await _wait_until_healthy(
                self.config(),
                self.contract(),
                timeout_seconds=1,
                expected_backend_pid_not=111,
                expected_bot_pid=999,
            )
        self.assertEqual("pass", evidence.status)
        self.assertEqual(222, evidence.backend.main_pid)
        self.assertEqual(999, evidence.bot.main_pid)
        self.assertTrue(evidence.probe.healthy)
        self.assertTrue(evidence.gpu_healthy)

    async def test_backend_only_restart_fails_if_bot_pid_changes(self):
        backend = ServiceSnapshot(
            service="llmcord-llama-server.service",
            active_state="active",
            sub_state="running",
            main_pid=222,
            fragment_path="/home/u/.config/systemd/user/llmcord-llama-server.service",
        )
        bot = ServiceSnapshot(
            service="llmcord.service",
            active_state="active",
            sub_state="running",
            main_pid=1000,
            fragment_path="/home/u/.config/systemd/user/llmcord.service",
        )
        with patch("runtime_evidence.service_snapshot", side_effect=[backend, bot]):
            evidence = await _wait_until_healthy(
                self.config(),
                self.contract(),
                timeout_seconds=1,
                expected_backend_pid_not=111,
                expected_bot_pid=999,
            )
        self.assertEqual("fail", evidence.status)
        self.assertIn("bot PID changed", evidence.reason)


if __name__ == "__main__":
    unittest.main()
