"""Command-line connection tests with all network calls mocked."""

from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from qwenfree.cli import app
from qwenfree.config import load_config

runner = CliRunner()


def test_connect_saves_private_config_after_health_check(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "config.toml"
    monkeypatch.setattr("qwenfree.config.default_config_path", lambda: path)
    with respx.mock(assert_all_called=True) as router:
        router.get("https://server.test/health").mock(return_value=httpx.Response(200))
        result = runner.invoke(
            app, ["connect", "https://server.test", "--key", "secret-value"]
        )

    assert result.exit_code == 0, result.output
    assert path.exists()
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_config(path).api_key == "secret-value"
    assert "secret-value" not in result.output


def test_connect_refuses_http_non_loopback() -> None:
    result = runner.invoke(app, ["connect", "http://server.test", "--key", "x"])

    assert result.exit_code == 2
    assert "HTTPS" in result.output


def test_ask_rejects_missing_question() -> None:
    result = runner.invoke(app, ["ask"], input="")

    assert result.exit_code == 2
    assert "Provide a question" in result.output
