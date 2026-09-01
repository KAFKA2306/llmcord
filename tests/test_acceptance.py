from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from acceptance import (
    AcceptanceError,
    AcceptanceStatus,
    REQUIRED_SCENARIOS,
    evaluate_acceptance,
    evaluate_case,
    evaluate_soak,
    load_json_events,
)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.artifact = Path(self.tempdir.name) / "Model-Q6_K.gguf"
        self.artifact.write_bytes(b"fixed acceptance model artifact")
        self.artifact_sha256 = hashlib.sha256(self.artifact.read_bytes()).hexdigest()

    def tearDown(self):
        self.tempdir.cleanup()

    def config(self):
        revision = "2b651f3"
        artifact_repo = "example/model-GGUF"
        artifact = self.artifact.name
        return {
            "max_images": 0,
            "runtime_control": {
                "max_concurrency": 1,
                "max_queue_size": 4,
            },
            "production": {
                "enabled": True,
                "backend": "llamacpp",
                "backend_release": "v0.3.0",
                "backend_version_or_commit": "c" * 40,
                "backend_executable": "/opt/llama-server",
                "model": "llamacpp/local-model",
                "model_artifact": {
                    "upstream": "example/model",
                    "artifact_repo": artifact_repo,
                    "artifact": artifact,
                    "revision": revision,
                    "artifact_url": f"https://huggingface.co/{artifact_repo}/resolve/{revision}/{artifact}",
                    "sha256": self.artifact_sha256,
                    "quantization_or_dtype": "Q6_K",
                    "context_window_tokens": 32768,
                    "verify_path": str(self.artifact),
                },
                "network": {
                    "mode": "wsl",
                    "endpoint": "http://127.0.0.1:8080/v1",
                },
                "supervisor": {
                    "kind": "systemd",
                    "restart_command": ["systemctl", "--user", "restart", "llama-server.service"],
                },
                "gpu": {
                    "required": True,
                    "device_index": 0,
                    "min_vram_used_mib": 1,
                    "process_name_pattern": r"llama-server(?:\.exe)?$",
                },
            },
            "providers": {
                "llamacpp": {"base_url": "http://127.0.0.1:8080/v1"},
            },
            "models": {
                "llamacpp/local-model": {
                    "context_management": {
                        "context_window_tokens": 32768,
                    }
                }
            },
            "health_control": {
                "enabled": True,
                "model": "llamacpp/local-model",
                "restart_command": ["systemctl", "--user", "restart", "llama-server.service"],
                "nvidia_gpu": {
                    "enabled": True,
                    "device_index": 0,
                    "min_vram_used_mib": 1,
                },
            },
        }

    def event(self, event, second=0, request_id=None):
        payload = {
            "timestamp": f"2026-09-01T00:00:{second:02d}+00:00",
            "event": event,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        return payload

    def passing_soak(self):
        return {
            "duration_hours": 24,
            "total_requests": 100,
            "successes": 100,
            "failures": 0,
            "timeouts": 0,
            "hangs": 0,
            "restarts": 2,
            "manual_interventions": 0,
            "peak_ram_mib": 32000,
            "peak_vram_mib": 15000,
            "queue_peak": 4,
            "queue_overflow_incidents": 0,
            "compaction_count": 12,
            "compaction_failures": 0,
            "summary_regressions": 0,
            "unreaped_orphan_processes": 0,
            "restart_storm_incidents": 0,
            "context_overflow_requests_sent": 0,
            "silent_truncation_or_drop_incidents": 0,
            "gpu_cpu_fallback_misclassifications": 0,
            "secret_or_token_leaks": 0,
        }

    def case(self, scenario, *, request_ids=("req-1",), required_events=()):
        return {
            "scenario": scenario,
            "started_at": "2026-09-01T00:00:00+00:00",
            "ended_at": "2026-09-01T00:00:59+00:00",
            "required_events": list(required_events),
            "manual_interventions": 0,
            "request_ids": list(request_ids),
            "assertion_mode": "automated",
            "assertions": {
                "stimulus_applied": True,
                "expected_behavior_observed": True,
                "automatic_recovery_observed": True,
            },
        }

    def test_case_does_not_trust_assertions_without_required_events(self):
        result = evaluate_case(
            self.case("backend_process_kill"),
            [self.event("request.rejected", 1, "req-1")],
        )
        self.assertEqual(AcceptanceStatus.FAIL, result.status)
        self.assertIn("required event missing: watchdog.restart", result.reasons)
        self.assertIn("required event missing: probe.success", result.reasons)

    def test_case_passes_only_when_required_events_and_discord_request_exist(self):
        events = [
            self.event("generation.failure", 1, "req-1"),
            self.event("watchdog.failure", 2),
            self.event("watchdog.restart", 3),
            self.event("probe.success", 4),
        ]
        result = evaluate_case(self.case("backend_process_kill"), events)
        self.assertEqual(AcceptanceStatus.PASS, result.status)

    def test_manual_assertion_mode_is_not_accepted(self):
        case = self.case("queue_full")
        case["assertion_mode"] = "manual"
        result = evaluate_case(case, [self.event("queue.rejected", 1, "req-1")])
        self.assertEqual(AcceptanceStatus.FAIL, result.status)
        self.assertIn("assertion_mode must be 'automated'", result.reasons)

    def test_concurrent_scenario_requires_two_distinct_discord_requests(self):
        result = evaluate_case(
            self.case("concurrent_discord_requests", request_ids=("req-1",)),
            [self.event("queue.admitted", 1, "req-1")],
        )
        self.assertEqual(AcceptanceStatus.FAIL, result.status)
        self.assertIn("scenario requires at least two distinct Discord request_ids", result.reasons)

    def test_repeated_compaction_requires_multiple_compactions(self):
        result = evaluate_case(
            self.case("repeated_compaction"),
            [self.event("request.received", 1, "req-1"), self.event("context.compacted", 2)],
        )
        self.assertEqual(AcceptanceStatus.FAIL, result.status)
        self.assertIn("repeated_compaction requires at least two context.compacted events", result.reasons)

    def test_missing_required_scenarios_make_acceptance_unverified(self):
        events = [self.event("queue.rejected", 1, "req-1")]
        result = evaluate_acceptance(
            config=self.config(),
            events=events,
            cases=[self.case("queue_full")],
            soak_metrics=self.passing_soak(),
        )
        self.assertEqual(AcceptanceStatus.UNVERIFIED, result.status)
        missing = [case for case in result.cases if case.status is AcceptanceStatus.UNVERIFIED]
        self.assertEqual(len(REQUIRED_SCENARIOS) - 1, len(missing))
        self.assertEqual("v0.3.0", result.runtime["backend_release"])
        self.assertEqual(self.artifact_sha256, result.runtime["model_sha256"])

    def test_soak_requires_24_hours_and_zero_manual_intervention(self):
        metrics = self.passing_soak()
        metrics["duration_hours"] = 23.9
        metrics["manual_interventions"] = 1
        result = evaluate_soak(metrics, queue_capacity=4)
        self.assertEqual(AcceptanceStatus.FAIL, result.status)
        self.assertIn("duration_hours must be at least 24", result.reasons)
        self.assertIn("manual_interventions must be 0", result.reasons)

    def test_soak_queue_peak_cannot_exceed_waiting_queue_limit(self):
        metrics = self.passing_soak()
        metrics["queue_peak"] = 5
        result = evaluate_soak(metrics, queue_capacity=4)
        self.assertEqual(AcceptanceStatus.FAIL, result.status)
        self.assertIn("queue_peak must not exceed bounded waiting queue 4", result.reasons)

    def test_missing_soak_metric_is_unverified_not_pass(self):
        metrics = self.passing_soak()
        del metrics["secret_or_token_leaks"]
        result = evaluate_soak(metrics, queue_capacity=4)
        self.assertEqual(AcceptanceStatus.UNVERIFIED, result.status)

    def test_disabled_production_contract_cannot_be_accepted(self):
        config = self.config()
        config["production"]["enabled"] = False
        with self.assertRaisesRegex(AcceptanceError, "production contract must be enabled"):
            evaluate_acceptance(config=config, events=[self.event("request.received", request_id="req-1")], cases=[], soak_metrics={})

    def test_event_evidence_rejects_sensitive_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(
                '{"timestamp":"2026-09-01T00:00:00+00:00","event":"request.received","prompt":"secret"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AcceptanceError, "forbidden sensitive key"):
                load_json_events(path)


if __name__ == "__main__":
    unittest.main()
