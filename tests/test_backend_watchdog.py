from __future__ import annotations

import unittest

from backend_probe import BackendProbeResult
from backend_watchdog import BackendWatchdog, WatchdogPolicy, WatchdogState


class SequenceProbe:
    def __init__(self, healthy_values: list[bool]) -> None:
        self.values = list(healthy_values)
        self.calls = 0

    async def __call__(self) -> BackendProbeResult:
        self.calls += 1
        if not self.values:
            raise AssertionError("probe sequence exhausted")
        healthy = self.values.pop(0)
        return BackendProbeResult(
            healthy=healthy,
            elapsed_seconds=0.01,
            error_class=None if healthy else "timeout",
        )


class BackendWatchdogTests(unittest.IsolatedAsyncioTestCase):
    def policy(self, **overrides) -> WatchdogPolicy:
        values = dict(
            probe_interval_seconds=30.0,
            failure_threshold=2,
            restart_cooldown_seconds=0.0,
            restart_settle_seconds=0.0,
            max_restart_attempts=2,
        )
        values.update(overrides)
        return WatchdogPolicy(**values)

    async def test_initialize_blocks_admission_until_probe_succeeds(self) -> None:
        admissions: list[bool] = []
        watchdog = BackendWatchdog(
            self.policy(),
            probe=SequenceProbe([True]),
            restart_backend=self._successful_restart,
            set_accepting=self._recorder(admissions),
        )

        await watchdog.initialize()
        self.assertEqual(admissions, [False])
        self.assertFalse(watchdog.snapshot().accepting)

        snapshot = await watchdog.tick()
        self.assertEqual(snapshot.state, WatchdogState.HEALTHY)
        self.assertTrue(snapshot.accepting)
        self.assertEqual(admissions, [False, True])

    async def test_single_failure_after_healthy_is_suspect_without_restart(self) -> None:
        admissions: list[bool] = []
        restarts = 0

        async def restart() -> bool:
            nonlocal restarts
            restarts += 1
            return True

        watchdog = BackendWatchdog(
            self.policy(),
            probe=SequenceProbe([True, False]),
            restart_backend=restart,
            set_accepting=self._recorder(admissions),
        )

        await watchdog.tick()
        snapshot = await watchdog.tick()

        self.assertEqual(snapshot.state, WatchdogState.SUSPECT)
        self.assertTrue(snapshot.accepting)
        self.assertEqual(restarts, 0)

    async def test_failure_threshold_blocks_admission_restarts_and_requires_recovery_probe(self) -> None:
        admissions: list[bool] = []
        restarts = 0

        async def restart() -> bool:
            nonlocal restarts
            restarts += 1
            return True

        watchdog = BackendWatchdog(
            self.policy(),
            probe=SequenceProbe([True, False, False, True]),
            restart_backend=restart,
            set_accepting=self._recorder(admissions),
        )

        await watchdog.tick()
        await watchdog.tick()
        snapshot = await watchdog.tick()

        self.assertEqual(restarts, 1)
        self.assertEqual(snapshot.state, WatchdogState.HEALTHY)
        self.assertTrue(snapshot.accepting)
        self.assertEqual(snapshot.restart_attempts_since_healthy, 0)
        self.assertEqual(snapshot.total_restart_attempts, 1)
        self.assertEqual(admissions, [True, False, True])

    async def test_failed_restarts_reach_degraded_and_do_not_storm(self) -> None:
        admissions: list[bool] = []
        restarts = 0

        async def restart() -> bool:
            nonlocal restarts
            restarts += 1
            return False

        watchdog = BackendWatchdog(
            self.policy(failure_threshold=1, max_restart_attempts=2),
            probe=SequenceProbe([False, False, False]),
            restart_backend=restart,
            set_accepting=self._recorder(admissions),
        )

        await watchdog.initialize()
        first = await watchdog.tick()
        second = await watchdog.tick()
        third = await watchdog.tick()

        self.assertEqual(first.state, WatchdogState.UNAVAILABLE)
        self.assertEqual(second.state, WatchdogState.DEGRADED)
        self.assertEqual(third.state, WatchdogState.DEGRADED)
        self.assertEqual(restarts, 2)
        self.assertFalse(third.accepting)
        self.assertEqual(third.total_restart_attempts, 2)

    async def test_recovery_probe_failure_keeps_admission_closed(self) -> None:
        admissions: list[bool] = []
        watchdog = BackendWatchdog(
            self.policy(failure_threshold=1),
            probe=SequenceProbe([False, False]),
            restart_backend=self._successful_restart,
            set_accepting=self._recorder(admissions),
        )

        await watchdog.initialize()
        snapshot = await watchdog.tick()

        self.assertEqual(snapshot.state, WatchdogState.UNAVAILABLE)
        self.assertFalse(snapshot.accepting)
        self.assertEqual(snapshot.total_restart_attempts, 1)

    async def _successful_restart(self) -> bool:
        return True

    def _recorder(self, target: list[bool]):
        async def record(value: bool) -> None:
            target.append(value)

        return record


if __name__ == "__main__":
    unittest.main()
