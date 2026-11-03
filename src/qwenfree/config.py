"""Secure local connection configuration."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import tomli_w

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


class ConfigError(ValueError):
    """Raised when connection configuration is invalid or unavailable."""


def default_config_path() -> Path:
    """Return the standard per-user QwenFree configuration path."""
    return Path.home() / ".config" / "qwenfree" / "config.toml"


def validate_server_url(value: str) -> str:
    """Validate and normalize an HTTPS URL or an HTTP loopback URL."""
    candidate = value.strip()
    parts = urlsplit(candidate)
    hostname = (parts.hostname or "").lower()
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    is_loopback = hostname in local_hosts
    if parts.scheme != "https" and not (parts.scheme == "http" and is_loopback):
        raise ConfigError("Server URL must use HTTPS (HTTP is allowed for localhost).")
    if not hostname or parts.username or parts.password:
        raise ConfigError(
            "Server URL must include a host and cannot contain credentials."
        )
    if parts.query or parts.fragment:
        raise ConfigError("Server URL cannot include a query string or fragment.")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


@dataclass(frozen=True, slots=True)
class QwenFreeConfig:
    """Connection settings saved on the local Mac."""

    base_url: str
    api_key: str
    model_name: str = "Qwen3.8-27B"

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", validate_server_url(self.base_url))
        if not self.api_key.strip():
            raise ConfigError("API key cannot be empty.")
        if not self.model_name.strip():
            raise ConfigError("Model name cannot be empty.")


def save_config(
    config: QwenFreeConfig, path: Path | None = None
) -> Path:
    """Atomically save settings and restrict the file to mode 0600."""
    config_path = path or default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.is_symlink():
        raise ConfigError("Refusing to write configuration through a symbolic link.")

    payload = tomli_w.dumps(
        {
            "base_url": config.base_url,
            "api_key": config.api_key,
            "model_name": config.model_name,
        }
    )
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=".config-", suffix=".toml", dir=config_path.parent
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, config_path)
        config_path.chmod(0o600)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise
    return config_path


def load_config(path: Path | None = None) -> QwenFreeConfig:
    """Load and validate saved settings, raising a user-friendly error."""
    config_path = path or default_config_path()
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return QwenFreeConfig(
            base_url=raw["base_url"],
            api_key=raw["api_key"],
            model_name=raw.get("model_name", "Qwen3.8-27B"),
        )
    except FileNotFoundError as exc:
        raise ConfigError(
            "No QwenFree connection configured. Run qwenfree connect."
        ) from exc
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Invalid QwenFree configuration at {config_path}.") from exc
