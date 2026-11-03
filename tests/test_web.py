"""Loopback web proxy tests with every remote call intercepted."""

import json

import httpx
import respx
from fastapi.testclient import TestClient

from qwenfree.config import QwenFreeConfig
from qwenfree.web.app import app


def test_index_is_self_contained_and_has_no_secret_channel() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert len(response.content) < 25 * 1024
    assert "https://" not in response.text
    assert "Authorization" not in response.text
    assert "cdn" not in response.text.lower()


def test_status_returns_only_reachability(monkeypatch) -> None:
    config = QwenFreeConfig("https://server.test", "not-for-browser")
    monkeypatch.setattr("qwenfree.web.app.load_config", lambda: config)
    with respx.mock(assert_all_called=True) as router:
        router.get("https://server.test/health").mock(
            return_value=httpx.Response(200)
        )
        response = TestClient(app).get("/api/status")

    assert response.status_code == 200
    assert response.json() == {"connected": True}
    assert "not-for-browser" not in response.text


def test_chat_stream_proxies_with_key_only_server_side(monkeypatch) -> None:
    config = QwenFreeConfig("https://server.test", "not-for-browser", "model-x")
    monkeypatch.setattr("qwenfree.web.app.load_config", lambda: config)
    event = {"choices": [{"delta": {"content": "hello"}}]}
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://server.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=body)
        )
        response = TestClient(app).post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
            },
        )

    assert response.status_code == 200
    assert '"delta": "hello"' in response.text
    assert "not-for-browser" not in response.text
    assert route.calls[0].request.headers["Authorization"] == "Bearer not-for-browser"


def test_non_local_origin_is_rejected() -> None:
    response = TestClient(app).get(
        "/api/status", headers={"Origin": "https://attacker.example"}
    )

    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers
