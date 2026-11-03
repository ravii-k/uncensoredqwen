"""Loopback-only FastAPI proxy for the browser chat interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from qwenfree.client import QwenFreeClient, RemoteError
from qwenfree.config import ConfigError, load_config

STATIC_FILE = Path(__file__).parent / "static" / "index.html"
app = FastAPI(title="QwenFree local chat", docs_url=None, redoc_url=None)


class ChatMessage(BaseModel):
    """One chat turn supplied by the local browser."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=200_000)


class ChatRequest(BaseModel):
    """Validated browser request for a streamed completion."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    temperature: float = Field(default=0.7, ge=0, le=2)


def _client() -> QwenFreeClient:
    try:
        return QwenFreeClient(load_config())
    except ConfigError as exc:
        raise HTTPException(
            status_code=503,
            detail="No valid QwenFree connection is configured. Run qwenfree connect.",
        ) from exc


@app.middleware("http")
async def reject_non_local_origins(request: Request, call_next):
    """Reject cross-site browser origins while leaving CORS disabled."""
    origin = request.headers.get("origin")
    if origin:
        hostname = (urlsplit(origin).hostname or "").lower()
        if hostname not in {"127.0.0.1", "localhost", "::1"}:
            return JSONResponse(
                {"detail": "Local browser origin required."}, status_code=403
            )
    return await call_next(request)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the self-contained, offline-friendly chat page."""
    return FileResponse(STATIC_FILE, media_type="text/html; charset=utf-8")


@app.get("/api/status")
async def api_status() -> dict[str, bool]:
    """Return remote reachability without disclosing credentials."""
    client = _client()
    return {"connected": await client.health_check()}


@app.post("/api/chat")
async def api_chat(payload: ChatRequest) -> StreamingResponse:
    """Proxy a streaming completion without returning the API key to JavaScript."""
    client = _client()
    messages = [message.model_dump() for message in payload.messages]

    async def events():
        try:
            async for delta in client.stream_chat(
                messages, temperature=payload.temperature
            ):
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except RemoteError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
