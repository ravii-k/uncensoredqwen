"""Standard per-user data, model, and cache locations."""

from pathlib import Path


def default_data_dir() -> Path:
    """Return the QwenFree data directory."""
    return Path.home() / ".local" / "share" / "qwenfree"


def default_model_dir() -> Path:
    """Return the optional local model directory."""
    return default_data_dir() / "models"


def default_cache_dir() -> Path:
    """Return the QwenFree cache directory."""
    return Path.home() / ".cache" / "qwenfree"
