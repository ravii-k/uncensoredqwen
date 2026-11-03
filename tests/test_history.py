"""History persistence and size cap tests."""

import json
from pathlib import Path

from qwenfree.history import append_turn, clear_history, history_size, read_history


def test_append_turn_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "history" / "chat.jsonl"
    append_turn("question", "answer", path)

    assert read_history(path)[0]["user"] == "question"
    assert read_history(path)[0]["assistant"] == "answer"
    assert path.stat().st_mode & 0o777 == 0o600


def test_append_trims_oldest_record_to_cap(tmp_path: Path) -> None:
    path = tmp_path / "chat.jsonl"
    append_turn("old", "answer", path, max_bytes=256)
    append_turn("new", "answer", path, max_bytes=140)

    records = read_history(path)
    assert [record["user"] for record in records] == ["new"]
    assert history_size(path) <= 140
    assert json.loads(path.read_text().splitlines()[0])["user"] == "new"


def test_clear_history_removes_file(tmp_path: Path) -> None:
    path = tmp_path / "history" / "chat.jsonl"
    append_turn("q", "a", path)

    clear_history(path)

    assert history_size(path) == 0
