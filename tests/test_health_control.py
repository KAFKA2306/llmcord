import asyncio
import unittest

from health_control import (
    BackendState,
    BackendWatchdog,
    HealthControlError,
    WatchdogPolicy,
    parse_nvidia_compute_processes,
    parse_nvidia_gpu_snapshot,
    require_expected_gpu_process,
)


class HealthControlTests(unittest.IsolatedAsyncioTestCase):
    def policy(self, **overrides):
        values = dict(
            probe_interval_seconds=0.01,
            failure_threshold=2,
            restart_cooldown_seconds=0.01,
            post_restart_grace_seconds=0.001,
            max_restart_attempts=2,
        )
        values.update(overrides)
        return WatchdogPolicy(**values)

    async def test_starting_is_not_accepted_until_real_success(self):
        async def probe():
            return None

        async def idle():
            return True

        watchdog = BackendWatchdog(self.policy(), probe=probe, idle=idle)
        self.assertFalse(await watchdog.is_accepting())
        await watchdog.report_success()
        snapshot = await watchdog.snapshot()
        self.assertEqual(BackendState.HEALTHY, snapshot.state)
        self.assertTrue(snapshot.accepting)

    async def test_startup_probe_failure_never_opens_admission(self):
        async def probe():
            return None

        async def idle():
            return True

        watchdog = BackendWatchdog(self.policy(failure_threshold=3), probe=probe, idle=idle)
        await watchdog.report_failure("startup probe failed")
        snapshot = await watchdog.snapshot()
        self.assertEqual(BackendState.STARTING, snapshot.state)
        self.assertFalse(snapshot.accepting)

    async def test_failure_threshold_stops_accepting(self):
        async def probe():
            return None

        async def idle():
            return True

        watchdog = BackendWatchdog(self.policy(), probe=probe, idle=idle)
        await watchdog.report_success()
        await watchdog.report_failure("first")
        snapshot = await watchdog.snapshot()
        self.assertEqual(BackendState.SUSPECT, snapshot.state)
        self.assertTrue(snapshot.accepting)

        await watchdog.report_failure("second")
        snapshot = await watchdog.snapshot()
        self.assertEqual(BackendState.DEGRADED, snapshot.state)
        self.assertFalse(snapshot.accepting)

    async def test_immediate_health_failure_stops_accepting_without_threshold_delay(self):
        async def probe():
            return None

        async def idle():
            return True

        watchdog = BackendWatchdog(self.policy(failure_threshold=3), probe=probe, idle=idle)
        await watchdog.report_success()
        await watchdog.report_failure("GPU disappeared", immediate=True)
        snapshot = await watchdog.snapshot()
        self.assertEqual(BackendState.DEGRADED, snapshot.state)
        self.assertFalse(snapshot.accepting)
        self.assertEqual(3, snapshot.consecutive_failures)

    async def test_user_generation_success_requires_gpu_health(self):
        gpu_healthy = True
        gpu_checks = 0

        async def probe():
            return None

        async def idle():
            return True

        async def gpu_check():
            nonlocal gpu_checks
            gpu_checks += 1
            if not gpu_healthy:
                raise HealthControlError("expected backend process missing from GPU")

        watchdog = BackendWatchdog(self.policy(), probe=probe, idle=idle, gpu_check=gpu_check)
        await watchdog.report_success()
        self.assertEqual(BackendState.HEALTHY, (await watchdog.snapshot()).state)

        gpu_healthy = False
        await watchdog.report_success()
        snapshot = await watchdog.snapshot()
        self.assertEqual(BackendState.DEGRADED, snapshot.state)
        self.assertFalse(snapshot.accepting)
        self.assertEqual(2, gpu_checks)

    async def test_restart_requires_probe_before_reopening(self):
        calls = []

        async def probe():
            calls.append("probe")

        async def idle():
            return True

        async def restart():
            calls.append("restart")

        async def gpu_check():
            calls.append("gpu")

        watchdog = BackendWatchdog(
            self.policy(),
            probe=probe,
            idle=idle,
            restart=restart,
            gpu_check=gpu_check,
        )
        await watchdog.report_failure("one")
        await watchdog.report_failure("two")
        await watchdog._attempt_recovery()

        snapshot = await watchdog.snapshot()
        self.assertEqual(BackendState.HEALTHY, snapshot.state)
        self.assertTrue(snapshot.accepting)
        self.assertEqual(["restart", "gpu", "probe"], calls)

    async def test_failed_restarts_are_capped(self):
        calls = 0

        async def probe():
            raise RuntimeError("still down")

        async def idle():
            return True

        async def restart():
            nonlocal calls
            calls += 1

        watchdog = BackendWatchdog(
            self.policy(max_restart_attempts=2),
            probe=probe,
            idle=idle,
            restart=restart,
        )
        await watchdog.report_failure("one")
        await watchdog.report_failure("two")
        await watchdog._attempt_recovery()
        await watchdog._attempt_recovery()
        await watchdog._attempt_recovery()

        snapshot = await watchdog.snapshot()
        self.assertEqual(2, calls)
        self.assertEqual(2, snapshot.restart_attempts)
        self.assertEqual(BackendState.DEGRADED, snapshot.state)
        self.assertFalse(snapshot.accepting)

    async def test_spontaneous_success_resets_restart_budget(self):
        async def probe():
            return None

        async def idle():
            return True

        watchdog = BackendWatchdog(self.policy(), probe=probe, idle=idle)
        await watchdog.report_failure("one")
        await watchdog.report_failure("two")
        await watchdog.report_success()
        snapshot = await watchdog.snapshot()
        self.assertEqual(0, snapshot.consecutive_failures)
        self.assertEqual(0, snapshot.restart_attempts)
        self.assertEqual(BackendState.HEALTHY, snapshot.state)

    async def test_healthy_state_does_not_inject_background_generation(self):
        probes = 0

        async def probe():
            nonlocal probes
            probes += 1

        async def idle():
            return True

        watchdog = BackendWatchdog(self.policy(probe_interval_seconds=0.005), probe=probe, idle=idle)
        await watchdog.report_success()
        task = asyncio.create_task(watchdog.run())
        await asyncio.sleep(0.02)
        await watchdog.stop()
        await task
        self.assertEqual(0, probes)

    async def test_healthy_state_periodically_checks_gpu_and_degrades_immediately(self):
        gpu_checks = 0
        probes = 0

        async def probe():
            nonlocal probes
            probes += 1

        async def idle():
            return True

        async def gpu_check():
            nonlocal gpu_checks
            gpu_checks += 1
            if gpu_checks >= 2:
                raise HealthControlError("CPU fallback detected")

        watchdog = BackendWatchdog(
            self.policy(probe_interval_seconds=0.005),
            probe=probe,
            idle=idle,
            gpu_check=gpu_check,
        )
        # First check is the user-success authority; subsequent periodic check fails.
        await watchdog.report_success()
        task = asyncio.create_task(watchdog.run())
        await asyncio.sleep(0.012)
        snapshot = await watchdog.snapshot()
        self.assertEqual(BackendState.DEGRADED, snapshot.state)
        self.assertFalse(snapshot.accepting)
        self.assertGreaterEqual(gpu_checks, 2)
        self.assertEqual(0, probes)
        await watchdog.stop()
        await task

    def test_gpu_snapshot_parser(self):
        snapshot = parse_nvidia_gpu_snapshot("GPU-abc, 14321, 77, 68\n")
        self.assertEqual("GPU-abc", snapshot.uuid)
        self.assertEqual(14321, snapshot.vram_used_mib)
        self.assertEqual(77.0, snapshot.utilization_percent)
        self.assertEqual(68.0, snapshot.temperature_c)

    def test_compute_process_parser_accepts_wddm_na_memory(self):
        processes = parse_nvidia_compute_processes(
            "1234, C:\\tools\\llama-server.exe, [N/A]\n"
            "5678, python.exe, 1024\n"
        )
        self.assertEqual(2, len(processes))
        self.assertIsNone(processes[0].used_gpu_memory_mib)
        self.assertEqual(1024, processes[1].used_gpu_memory_mib)

    def test_expected_backend_process_must_exist_in_selected_gpu_query(self):
        processes = parse_nvidia_compute_processes(
            "1234, /opt/llama/llama-server, 12000\n"
            "5678, /usr/bin/python, 8000\n"
        )
        match = require_expected_gpu_process(
            processes,
            process_name_pattern=r"llama-server(?:\.exe)?$",
        )
        self.assertEqual(1234, match.pid)

        with self.assertRaisesRegex(HealthControlError, "possible CPU fallback"):
            require_expected_gpu_process(
                [processes[1]],
                process_name_pattern=r"llama-server(?:\.exe)?$",
            )

    def test_invalid_gpu_process_pattern_fails_loudly(self):
        with self.assertRaisesRegex(HealthControlError, "invalid GPU process_name_pattern"):
            require_expected_gpu_process([], process_name_pattern="[")

    def test_invalid_policy_fails_loudly(self):
        with self.assertRaises(HealthControlError):
            self.policy(failure_threshold=0)


if __name__ == "__main__":
    unittest.main()
