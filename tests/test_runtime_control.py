import asyncio
import unittest

from runtime_control import (
    GenerationProtocolError,
    GenerationTimeout,
    InferenceGate,
    InferenceQueueFull,
    InferenceQueueTimeout,
    InferenceUnavailable,
    RuntimeControlError,
    RuntimePolicy,
    stream_with_timeouts,
)


class FakeStream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.events):
            raise StopAsyncIteration
        delay, value = self.events[self.index]
        self.index += 1
        await asyncio.sleep(delay)
        return value

    async def close(self):
        self.closed = True


class RuntimeControlTests(unittest.IsolatedAsyncioTestCase):
    def policy(self, **overrides):
        values = dict(
            max_concurrency=1,
            max_queue_size=1,
            queue_wait_timeout_seconds=0.2,
            connect_timeout_seconds=0.1,
            first_token_timeout_seconds=0.1,
            total_generation_timeout_seconds=0.3,
        )
        values.update(overrides)
        return RuntimePolicy(**values)

    async def test_gate_rejects_work_beyond_active_plus_queue_capacity(self):
        gate = InferenceGate(self.policy())
        first = await gate.acquire()

        second_task = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)

        with self.assertRaises(InferenceQueueFull):
            await gate.acquire()

        await first.release()
        second = await second_task
        await second.release()
        self.assertEqual(0, (await gate.snapshot())["in_system"])

    async def test_gate_queue_wait_has_finite_timeout(self):
        gate = InferenceGate(self.policy(queue_wait_timeout_seconds=0.02))
        first = await gate.acquire()
        with self.assertRaises(InferenceQueueTimeout):
            await gate.acquire()
        await first.release()
        self.assertEqual(0, (await gate.snapshot())["in_system"])

    async def test_health_control_can_close_and_reopen_admission(self):
        gate = InferenceGate(self.policy())
        await gate.set_accepting(False)

        with self.assertRaises(InferenceUnavailable):
            await gate.acquire()

        self.assertFalse((await gate.snapshot())["accepting"])
        await gate.set_accepting(True)
        lease = await gate.acquire()
        await lease.release()
        self.assertTrue((await gate.snapshot())["accepting"])

    async def test_queued_request_does_not_start_after_admission_closes(self):
        gate = InferenceGate(self.policy(queue_wait_timeout_seconds=0.5))
        first = await gate.acquire()
        queued = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)

        await gate.set_accepting(False)
        await first.release()

        with self.assertRaises(InferenceUnavailable):
            await queued
        self.assertEqual(0, (await gate.snapshot())["in_system"])

    async def test_lease_releases_after_exception(self):
        gate = InferenceGate(self.policy())
        with self.assertRaisesRegex(RuntimeError, "boom"):
            async with await gate.acquire():
                raise RuntimeError("boom")
        self.assertEqual(0, (await gate.snapshot())["in_system"])

    async def test_first_generation_signal_timeout_closes_stream(self):
        stream = FakeStream([(0.05, "empty"), (0.2, "token")])

        async def factory():
            return stream

        with self.assertRaises(GenerationTimeout) as raised:
            async for _ in stream_with_timeouts(
                factory,
                first_signal=lambda item: item == "token",
                first_token_timeout_seconds=0.08,
                total_generation_timeout_seconds=0.3,
            ):
                pass

        self.assertEqual("first_token", raised.exception.phase)
        self.assertTrue(stream.closed)

    async def test_total_generation_timeout_closes_stream(self):
        stream = FakeStream([(0, "token"), (0.2, "later")])

        async def factory():
            return stream

        received = []
        with self.assertRaises(GenerationTimeout) as raised:
            async for item in stream_with_timeouts(
                factory,
                first_signal=lambda item: item == "token",
                first_token_timeout_seconds=0.05,
                total_generation_timeout_seconds=0.08,
            ):
                received.append(item)

        self.assertEqual(["token"], received)
        self.assertEqual("total", raised.exception.phase)
        self.assertTrue(stream.closed)

    async def test_stream_ending_without_signal_fails_loudly(self):
        stream = FakeStream([(0, "empty")])

        async def factory():
            return stream

        with self.assertRaises(GenerationProtocolError):
            async for _ in stream_with_timeouts(
                factory,
                first_signal=lambda item: item == "token",
                first_token_timeout_seconds=0.05,
                total_generation_timeout_seconds=0.1,
            ):
                pass
        self.assertTrue(stream.closed)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(RuntimeControlError):
            self.policy(max_concurrency=0)
        with self.assertRaises(RuntimeControlError):
            self.policy(first_token_timeout_seconds=1, total_generation_timeout_seconds=0.5)


if __name__ == "__main__":
    unittest.main()
