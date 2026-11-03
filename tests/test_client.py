"""Mocked HTTP behavior for the remote client."""

import json

import httpx
import pytest
import respx

from qwenfree.client import QwenFreeClient, RemoteError
from qwenfree.config import QwenFreeConfig


@pytest.mark.asyncio
async def test_health_check_uses_health_endpoint() -> None:
    config = QwenFreeConfig("https://example.test", "secret")
    with respx.mock(assert_all_called=True) as router:
        route = router.get("https://example.test/health").mock(return_value=httpx.Response(200))
        assert await QwenFreeClient(config).health_check()
    assert route.called


@pytest.mark.asyncio
async def test_chat_stream_parses_sse_and_sends_auth_header() -> None:
    config = QwenFreeConfig("https://example.test", "secret", "model-x")
    events = [
        {"choices": [{"delta": {"content": "hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]
    payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    payload += "data: [DONE]\n\n"
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://example.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=payload)
        )
        chunks = [
            chunk
            async for chunk in QwenFreeClient(config).stream_chat(
                [{"role": "user", "content": "hi"}]
            )
        ]

    assert chunks == ["hello", " world"]
    assert route.calls[0].request.headers["Authorization"] == "Bearer secret"
    assert route.calls[0].request.read().decode().find('"stream":true') >= 0


@pytest.mark.asyncio
async def test_server_error_is_sanitized() -> None:
    config = QwenFreeConfig("https://example.test", "secret")
    with respx.mock(assert_all_called=True) as router:
        router.post("https://example.test/v1/chat/completions").mock(
            return_value=httpx.Response(401, text="secret")
        )
        with pytest.raises(RemoteError) as caught:
            _ = [
                chunk
                async for chunk in QwenFreeClient(config, max_attempts=1).stream_chat(
                    [{"role": "user", "content": "hi"}]
                )
            ]
    assert "secret" not in str(caught.value)
