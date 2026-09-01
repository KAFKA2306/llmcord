from __future__ import annotations

import json
import logging
import unittest

from observability import StructuredJsonFormatter, build_event, classify_message, sanitize_fields


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
                "message": "private log body",
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
            "message",
            "message_content",
            "attachment_body",
        ):
            self.assertEqual("[REDACTED]", fields[key])
        self.assertEqual(123, fields["input_tokens"])
        self.assertEqual("[REDACTED]", fields["nested"]["response_text"])
        self.assertEqual(1, fields["nested"]["safe"])

    def test_received_message_body_is_not_emitted(self):
        secret = "THIS MUST NEVER REACH OBSERVABILITY"
        payload = classify_message(
            f"request_id=123 Message received (user ID: 999, attachments: 2, conversation length: 7):\n{secret}",
            level="INFO",
            logger_name="root",
        )
        encoded = json.dumps(payload)
        self.assertEqual("request.received", payload["event"])
        self.assertEqual("123", payload["request_id"])
        self.assertEqual(2, payload["attachments"])
        self.assertEqual(7, payload["conversation_messages"])
        self.assertNotIn(secret, encoded)
        self.assertNotIn("999", encoded)

    def test_unclassified_log_body_is_dropped(self):
        secret = "provider returned private request body"
        payload = classify_message(
            f"request_id=123 unexpected diagnostic {secret}",
            level="ERROR",
            logger_name="root",
        )
        encoded = json.dumps(payload)
        self.assertEqual("log", payload["event"])
        self.assertFalse(payload["classified"])
        self.assertEqual("123", payload["request_id"])
        self.assertNotIn(secret, encoded)
        self.assertNotIn("message", payload)

    def test_runtime_log_classification(self):
        admitted = classify_message(
            "request_id=123 inference admitted queue_wait_seconds=0.125",
            level="INFO",
            logger_name="root",
        )
        self.assertEqual("queue.admitted", admitted["event"])
        self.assertEqual(0.125, admitted["queue_wait_seconds"])

        timeout = classify_message(
            "request_id=123 Generation timeout phase=stream_idle",
            level="ERROR",
            logger_name="root",
        )
        self.assertEqual("generation.timeout", timeout["event"])
        self.assertEqual("stream_idle", timeout["phase"])

        compacted = classify_message(
            "Context compacted (input tokens: 15000 -> 8000, current input compacted: False)",
            level="INFO",
            logger_name="root",
        )
        self.assertEqual("context.compacted", compacted["event"])
        self.assertEqual(15000, compacted["input_tokens_before"])
        self.assertEqual(8000, compacted["input_tokens_after"])

    def test_formatter_adds_static_production_identity(self):
        formatter = StructuredJsonFormatter(static_fields={"provider": "llamacpp", "model": "llamacpp/local"})
        record = logging.LogRecord(
            name="root",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="request_id=123 inference queue full",
            args=(),
            exc_info=None,
        )
        payload = json.loads(formatter.format(record))
        self.assertEqual("queue.rejected", payload["event"])
        self.assertEqual("llamacpp", payload["provider"])
        self.assertEqual("llamacpp/local", payload["model"])

    def test_invalid_required_fields_fail_in_builder(self):
        with self.assertRaises(ValueError):
            build_event("")
        with self.assertRaises(ValueError):
            build_event("request.accepted", request_id="")


if __name__ == "__main__":
    unittest.main()
