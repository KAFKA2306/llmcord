from __future__ import annotations

import json
import unittest

from observability import build_event, sanitize_fields


class ObservabilityTests(unittest.TestCase):
    def test_event_is_machine_readable_and_traceable(self):
        payload = build_event(
            "generation.finish",
            request_id="123",
            timestamp="2026-09-01T00:00:00.000+00:00",
            provider="llamacpp",
            model="local-model",
            duration_seconds=1.25,
            output_tokens=42,
        )
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        self.assertEqual("generation.finish", decoded["event"])
        self.assertEqual("123", decoded["request_id"])
        self.assertEqual(42, decoded["output_tokens"])

    def test_sensitive_bodies_and_secrets_are_redacted(self):
        fields = sanitize_fields(
            {
                "api_key": "secret-key",
                "authorization": "Bearer secret",
                "bot_token": "discord-secret",
                "prompt": "private prompt",
                "content": "private content",
                "message_content": "private message",
                "attachment_body": "private file",
                "input_tokens": 123,
                "nested": {"response_text": "private response", "safe": 1},
            }
        )
        for key in (
            "api_key",
            "authorization",
            "bot_token",
            "prompt",
            "content",
            "message_content",
            "attachment_body",
        ):
            self.assertEqual("[REDACTED]", fields[key])
        self.assertEqual(123, fields["input_tokens"])
        self.assertEqual("[REDACTED]", fields["nested"]["response_text"])
        self.assertEqual(1, fields["nested"]["safe"])

    def test_invalid_required_fields_fail_in_builder(self):
        with self.assertRaises(ValueError):
            build_event("")
        with self.assertRaises(ValueError):
            build_event("request.accepted", request_id="")


if __name__ == "__main__":
    unittest.main()
