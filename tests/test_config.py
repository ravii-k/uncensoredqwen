"""Configuration validation and file permission tests."""

from pathlib import Path

import pytest

from qwenfree.config import (
    ConfigError,
    QwenFreeConfig,
    load_config,
    save_config,
    validate_server_url,
)


def test_url_validation_accepts_https_and_loopback() -> None:
    assert validate_server_url("https://example.trycloudflare.com/") == (
        "https://example.trycloudflare.com"
    )
    assert validate_server_url("http://localhost:8080/") == "http://localhost:8080"


@pytest.mark.parametrize(
    "url",
    ["http://example.com", "file:///tmp/model", "https://user:pass@example.com"],
)
def test_url_validation_rejects_insecure_or_credential_urls(url: str) -> None:
    with pytest.raises(ConfigError):
        validate_server_url(url)


def test_save_and_load_uses_mode_600(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    expected = QwenFreeConfig("https://example.com", "secret", "test-model")
    save_config(expected, path)

    assert load_config(path) == expected
    assert path.stat().st_mode & 0o777 == 0o600


def test_load_missing_config_has_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="qwenfree connect"):
        load_config(tmp_path / "missing.toml")
