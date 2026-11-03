"""Size-capped, append-only local conversation history."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwenfree.paths import default_data_dir

MAX_HISTORY_BYTES = 50 * 1024 * 1024


def default_history_path() -> Path:
    """Return the standard local JSONL history path."""
    return default_data_dir() / "history" / "chat.jsonl"


def history_size(path: Path | None = None) -> int:
    """Return history bytes currently stored, or zero when absent."""
    history_path = path or default_history_path()
    try:
        return history_path.stat().st_size
    except FileNotFoundError:
        return 0


def read_history(path: Path | None = None) -> list[dict[str, Any]]:
    """Read valid JSONL records from history."""
    history_path = path or default_history_path()
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def append_turn(
    user: str,
    assistant: str,
    path: Path | None = None,
    *,
    max_bytes: int = MAX_HISTORY_BYTES,
) -> None:
    """Append one turn and trim oldest lines to stay within the size cap."""
    history_path = path or default_history_path()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "user": user,
        "assistant": assistant,
    }
    new_line = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()
    if len(new_line) > max_bytes:
        raise ValueError(
            "This conversation turn exceeds the configured history size cap."
        )

    try:
        existing = history_path.read_bytes().splitlines(keepends=True)
    except FileNotFoundError:
        existing = []
    retained = [line for line in existing if line.strip()]
    retained_bytes = sum(map(len, retained))
    while retained and retained_bytes + len(new_line) > max_bytes:
        retained_bytes -= len(retained[0])
        retained.pop(0)
    payload = b"".join(retained) + new_line
    _write_private(history_path, payload)


def clear_history(path: Path | None = None) -> None:
    """Remove the history file and its now-empty parent directory."""
    history_path = path or default_history_path()
    history_path.unlink(missing_ok=True)
    try:
        history_path.parent.rmdir()
    except OSError:
        pass


def _write_private(path: Path, payload: bytes) -> None:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=".history-", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        path.chmod(0o600)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise
