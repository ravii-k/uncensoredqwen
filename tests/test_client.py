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
        route = router.get("https://example.test/health").mock(
            return_value=httpx.Response(200)
        )
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
@pytest.mark.parametrize(
    ("configured_limit", "expected_limit"), [(None, 512), ("768", 768)]
)
async def test_cloudflare_quick_tunnel_uses_completed_json_response(
    configured_limit, expected_limit, monkeypatch, capsys
) -> None:
    config = QwenFreeConfig("https://random.trycloudflare.com", "secret", "model-x")
    if configured_limit is None:
        monkeypatch.delenv("QWENFREE_MAX_TOKENS", raising=False)
    else:
        monkeypatch.setenv("QWENFREE_MAX_TOKENS", configured_limit)
    payload = {"choices": [{"message": {"content": "completed answer"}}]}
    with respx.mock(assert_all_called=True) as router:
        route = router.post(
            "https://random.trycloudflare.com/v1/chat/completions"
        ).mock(return_value=httpx.Response(200, json=payload))
        chunks = [
            chunk
            async for chunk in QwenFreeClient(config).stream_chat(
                [{"role": "user", "content": "hi"}]
            )
        ]

    assert chunks == ["completed answer"]
    request = json.loads(route.calls[0].request.read())
    assert request["stream"] is False
    assert request["max_tokens"] == expected_limit
    assert "non-streaming" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_cloudflare_read_timeout_is_not_retried() -> None:
    config = QwenFreeConfig("https://random.trycloudflare.com", "secret")
    with respx.mock(assert_all_called=True) as router:
        route = router.post(
            "https://random.trycloudflare.com/v1/chat/completions"
        ).mock(side_effect=httpx.ReadTimeout("slow response"))
        with pytest.raises(RemoteError):
            _ = [
                chunk
                async for chunk in QwenFreeClient(config, max_attempts=3).stream_chat(
                    [{"role": "user", "content": "hi"}]
                )
            ]

    assert route.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("connection_error", [httpx.ConnectError, httpx.ConnectTimeout])
async def test_cloudflare_retries_connection_errors_only(
    connection_error, monkeypatch
) -> None:
    config = QwenFreeConfig("https://random.trycloudflare.com", "secret")

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("qwenfree.client.asyncio.sleep", no_sleep)
    with respx.mock(assert_all_called=True) as router:
        route = router.post(
            "https://random.trycloudflare.com/v1/chat/completions"
        ).mock(
            side_effect=[
                connection_error("connection refused"),
                httpx.Response(
                    200, json={"choices": [{"message": {"content": "answer"}}]}
                ),
            ]
        )
        chunks = [
            chunk
            async for chunk in QwenFreeClient(config, max_attempts=2).stream_chat(
                [{"role": "user", "content": "hi"}]
            )
        ]

    assert chunks == ["answer"]
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_cloudflare_server_error_is_not_retried() -> None:
    config = QwenFreeConfig("https://random.trycloudflare.com", "secret")
    with respx.mock(assert_all_called=True) as router:
        route = router.post(
            "https://random.trycloudflare.com/v1/chat/completions"
        ).mock(return_value=httpx.Response(503))
        with pytest.raises(RemoteError, match="HTTP 503"):
            _ = [
                chunk
                async for chunk in QwenFreeClient(config, max_attempts=3).stream_chat(
                    [{"role": "user", "content": "hi"}]
                )
            ]

    assert route.call_count == 1


@pytest.mark.asyncio
async def test_cloudflare_buffered_request_uses_300_second_read_timeout(
    monkeypatch,
) -> None:
    config = QwenFreeConfig("https://random.trycloudflare.com", "secret")
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "answer"}}]},
                request=httpx.Request("POST", "https://random.trycloudflare.com"),
            )

    monkeypatch.setattr("qwenfree.client.httpx.AsyncClient", FakeAsyncClient)
    chunks = [
        chunk
        async for chunk in QwenFreeClient(config).stream_chat(
            [{"role": "user", "content": "hi"}]
        )
    ]

    assert chunks == ["answer"]
    assert isinstance(captured["timeout"], httpx.Timeout)
    assert captured["timeout"].read == 300.0


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
