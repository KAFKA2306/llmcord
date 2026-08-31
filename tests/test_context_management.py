import unittest

from context_management import ContextManagementError, ContextPolicy, prepare_context


def text_size(messages):
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += len(content)
        else:
            total += len(str(content))
    return total


class ContextManagementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def count_tokens(messages):
            return (text_size(messages) + 9) // 10

        async def compact(messages, max_output_tokens):
            joined = " | ".join(str(message.get("content", "")) for message in messages)
            return ("summary:" + joined[:100])[:max_output_tokens]

        self.count_tokens = count_tokens
        self.compact = compact

    async def test_context_that_fits_is_unchanged(self):
        policy = ContextPolicy(120, 20, 0, 3, 20)
        messages = [{"role": "user", "content": "hello"}]
        result = await prepare_context(messages, policy, self.count_tokens, self.compact)
        self.assertFalse(result.compacted)
        self.assertEqual(messages, result.messages)

    async def test_old_history_is_compacted_and_current_message_is_verbatim(self):
        policy = ContextPolicy(120, 20, 0, 2, 20)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "A" * 600},
            {"role": "assistant", "content": "B" * 600},
            {"role": "user", "content": "current request"},
        ]
        result = await prepare_context(messages, policy, self.count_tokens, self.compact)
        self.assertTrue(result.compacted)
        self.assertEqual("current request", result.messages[-1]["content"])
        self.assertTrue(any("Compacted conversation state" in str(m.get("content")) for m in result.messages))
        self.assertLessEqual(result.input_tokens_after, policy.hard_input_limit)

    async def test_large_current_input_is_compacted_automatically(self):
        policy = ContextPolicy(120, 20, 0, 2, 20)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "X" * 1200},
        ]
        result = await prepare_context(messages, policy, self.count_tokens, self.compact)
        self.assertTrue(result.compacted_current_input)
        self.assertIn("Current user input was automatically compacted", result.messages[-1]["content"])
        self.assertLessEqual(result.input_tokens_after, policy.hard_input_limit)

    async def test_oversized_non_text_message_fails_loudly(self):
        policy = ContextPolicy(120, 20, 0, 1, 20)
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x" * 1200}}]},
        ]
        with self.assertRaises(ContextManagementError):
            await prepare_context(messages, policy, self.count_tokens, self.compact)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ContextManagementError):
            ContextPolicy(100, 90, 10, 1, 20)


if __name__ == "__main__":
    unittest.main()
