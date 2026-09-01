import asyncio
import unittest

from health_control import BackendState, BackendWatchdog, HealthControlError, WatchdogPolicy


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

    def test_invalid_policy_fails_loudly(self):
        with self.assertRaises(HealthControlError):
            self.policy(failure_threshold=0)


if __name__ == "__main__":
    unittest.main()
