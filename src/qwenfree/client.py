"""Small async OpenAI-compatible client with tunnel-aware response handling."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from qwenfree.config import QwenFreeConfig


class RemoteError(RuntimeError):
    """A sanitized remote-server or transport error."""


class QwenFreeClient:
    """Call llama-server using SSE or a buffered response when the tunnel needs it."""

    def __init__(
        self,
        config: QwenFreeConfig,
        *,
        timeout: float = 30.0,
        max_attempts: int = 3,
    ) -> None:
        self.config = config
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)

    @property
    def headers(self) -> dict[str, str]:
        """Return request headers without exposing them in diagnostics."""
        return {"Authorization": f"Bearer {self.config.api_key}"}

    async def health_check(self) -> bool:
        """Return true when the server health endpoint answers successfully."""
        for attempt in range(self.max_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(
                        f"{self.config.base_url}/health", headers=self.headers
                    )
                    if response.is_success:
                        return True
                    if response.status_code < 500 and response.status_code != 408:
                        return False
            except httpx.HTTPError:
                pass
            if attempt + 1 < self.max_attempts:
                await asyncio.sleep(0.5 * (2**attempt))
        return False

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.7,
        model_name: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas from a streaming chat completion."""
        body: dict[str, Any] = {
            "model": model_name or self.config.model_name,
            "messages": list(messages),
            "temperature": temperature,
            "stream": True,
        }
        url = f"{self.config.base_url}/v1/chat/completions"
        hostname = (urlsplit(self.config.base_url).hostname or "").lower()
        if hostname.endswith(".trycloudflare.com"):
            # Cloudflare Quick Tunnels currently do not carry SSE responses.
            body["stream"] = False
            async for content in self._buffered_chat(url, body):
                yield content
            return

        emitted = False
        last_error: RemoteError | None = None

        for attempt in range(self.max_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST", url, headers=self.headers, json=body
                    ) as response:
                        if response.status_code >= 400:
                            if (
                                response.status_code < 500
                                and response.status_code != 408
                            ):
                                raise RemoteError(
                                    f"Server returned HTTP {response.status_code}."
                                )
                            raise httpx.HTTPStatusError(
                                f"HTTP {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                event = json.loads(data)
                                delta = event["choices"][0].get("delta", {})
                                content = delta.get("content")
                            except (
                                KeyError,
                                IndexError,
                                TypeError,
                                json.JSONDecodeError,
                            ):
                                continue
                            if isinstance(content, str) and content:
                                emitted = True
                                yield content
                return
            except RemoteError:
                raise
            except httpx.HTTPError as exc:
                last_error = RemoteError(
                    "Could not reach the QwenFree server. Check the tunnel URL and "
                    "API key; the Kaggle session may have ended."
                )
                if emitted or attempt + 1 >= self.max_attempts:
                    raise last_error from exc
                await asyncio.sleep(0.5 * (2**attempt))

        if last_error is not None:
            raise last_error

    async def _buffered_chat(
        self, url: str, body: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Use one completed JSON response for tunnels that do not carry SSE."""
        last_error: RemoteError | None = None
        for attempt in range(self.max_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=self.headers, json=body)
                    if response.status_code >= 400:
                        if response.status_code < 500 and response.status_code != 408:
                            raise RemoteError(
                                f"Server returned HTTP {response.status_code}."
                            )
                        response.raise_for_status()
                    try:
                        payload = response.json()
                        content = payload["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        raise RemoteError(
                            "Server returned an invalid chat response."
                        ) from exc
                    if isinstance(content, str) and content:
                        yield content
                return
            except RemoteError:
                raise
            except httpx.HTTPError as exc:
                last_error = RemoteError(
                    "Could not reach the QwenFree server. Check the tunnel URL and "
                    "API key; the Kaggle session may have ended."
                )
                if attempt + 1 >= self.max_attempts:
                    raise last_error from exc
                await asyncio.sleep(0.5 * (2**attempt))
        if last_error is not None:
            raise last_error
