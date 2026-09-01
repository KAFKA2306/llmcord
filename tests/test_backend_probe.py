from __future__ import annotations

import json
import unittest

import httpx

from backend_probe import BackendProbeConfig, probe_generation


class BackendProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_probe_is_healthy_only_on_expected_output(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "local-model")
            self.assertFalse(payload["stream"])
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "PONG"}}]},
            )

        result = await probe_generation(
            BackendProbeConfig(base_url="http://backend", model="local-model"),
            transport=httpx.MockTransport(handler),
        )

        self.assertTrue(result.healthy)
        self.assertEqual(result.response_text, "PONG")
        self.assertIsNone(result.error_class)

    async def test_v1_base_url_headers_query_and_prompt_are_preserved(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.url.params["api-version"], "test")
            self.assertEqual(request.headers["X-Test"], "yes")
            payload = json.loads(request.content)
            self.assertEqual(payload["messages"][0]["content"], "Say READY")
            return httpx.Response(200, json={"choices": [{"message": {"content": "READY"}}]})

        result = await probe_generation(
            BackendProbeConfig(
                base_url="http://backend/v1",
                model="local-model",
                prompt="Say READY",
                expected_text="READY",
                headers={"X-Test": "yes"},
                query={"api-version": "test"},
            ),
            transport=httpx.MockTransport(handler),
        )
        self.assertTrue(result.healthy)

    async def test_generation_probe_rejects_shallow_but_invalid_response(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"object": "list", "data": []})

        result = await probe_generation(
            BackendProbeConfig(base_url="http://backend", model="local-model"),
            transport=httpx.MockTransport(handler),
        )

        self.assertFalse(result.healthy)
        self.assertEqual(result.error_class, "protocol_error")

    async def test_generation_probe_rejects_unexpected_model_output(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "PONG!"}}]},
            )

        result = await probe_generation(
            BackendProbeConfig(base_url="http://backend", model="local-model"),
            transport=httpx.MockTransport(handler),
        )

        self.assertFalse(result.healthy)
        self.assertEqual(result.error_class, "unexpected_output")

    async def test_generation_probe_classifies_http_failure(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="busy")

        result = await probe_generation(
            BackendProbeConfig(base_url="http://backend", model="local-model"),
            transport=httpx.MockTransport(handler),
        )

        self.assertFalse(result.healthy)
        self.assertEqual(result.error_class, "http_status")
        self.assertEqual(result.detail, "HTTP 503")

    async def test_generation_probe_classifies_timeout(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("generation stalled", request=request)

        result = await probe_generation(
            BackendProbeConfig(base_url="http://backend", model="local-model"),
            transport=httpx.MockTransport(handler),
        )

        self.assertFalse(result.healthy)
        self.assertEqual(result.error_class, "timeout")

    def test_probe_config_requires_positive_limits(self) -> None:
        with self.assertRaises(ValueError):
            BackendProbeConfig(
                base_url="http://backend",
                model="local-model",
                timeout_seconds=0,
            )

        with self.assertRaises(ValueError):
            BackendProbeConfig(
                base_url="http://backend",
                model="local-model",
                max_tokens=0,
            )

        with self.assertRaises(ValueError):
            BackendProbeConfig(
                base_url="http://backend",
                model="local-model",
                prompt="",
            )


if __name__ == "__main__":
    unittest.main()
