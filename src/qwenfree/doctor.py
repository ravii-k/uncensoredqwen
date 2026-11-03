"""Environment and connection diagnostics for the Mac client."""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from qwenfree.client import QwenFreeClient
from qwenfree.config import ConfigError, default_config_path, load_config

GIB = 1024**3
MIB = 1024**2
VENV_TARGET_BYTES = 150 * MIB


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One named diagnostic result."""

    name: str
    status: str
    detail: str


def directory_size(path: Path) -> int:
    """Return the size of regular files below a directory without following links."""
    if path.is_file() and not path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    if not path.is_dir() or path.is_symlink():
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _ram_bytes() -> int | None:
    """Read physical RAM size using macOS sysctl or POSIX sysconf."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            return int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _venv_path() -> Path | None:
    """Return the active virtual environment, if one is active."""
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix)
    return None


def _remote_check(config_path: Path) -> DoctorCheck:
    try:
        config = load_config(config_path)
    except ConfigError:
        return DoctorCheck(
            "Remote server", "WARN", "Not configured; run qwenfree connect."
        )
    healthy = asyncio.run(
        QwenFreeClient(config, timeout=5, max_attempts=1).health_check()
    )
    if healthy:
        return DoctorCheck("Remote server", "PASS", f"Responding at {config.base_url}")
    return DoctorCheck(
        "Remote server",
        "WARN",
        "Unreachable; the Kaggle session may have ended or the tunnel changed.",
    )


def collect_checks(
    config_path: Path | None = None,
    venv_path: Path | None = None,
    *,
    machine: str | None = None,
    system_name: str | None = None,
    macos_version: str | None = None,
    free_bytes: int | None = None,
    ram_bytes: int | None = None,
    python_version: tuple[int, int, int] | None = None,
    brew_path: str | None = None,
    llama_server_path: str | None = None,
) -> list[DoctorCheck]:
    """Collect platform, storage, dependency, permission, and server checks."""
    settings_path = config_path or default_config_path()
    active_venv = venv_path if venv_path is not None else _venv_path()
    machine_name = (machine or platform.machine()).lower()
    operating_system = system_name or platform.system()
    py_version = python_version or sys.version_info[:3]
    memory = ram_bytes if ram_bytes is not None else _ram_bytes()

    checks = [
        DoctorCheck(
            "Architecture",
            "PASS" if machine_name in {"arm64", "aarch64"} else "FAIL",
            f"{machine_name}; Apple Silicon arm64 is required.",
        )
    ]

    if operating_system == "Darwin":
        detected_version = macos_version or platform.mac_ver()[0] or "unknown"
        checks.append(DoctorCheck("macOS", "PASS", detected_version))
    else:
        checks.append(
            DoctorCheck(
                "macOS", "WARN", f"Running on {operating_system}; Mac is required."
            )
        )

    available_disk = (
        free_bytes if free_bytes is not None else shutil.disk_usage(Path.home()).free
    )
    disk_status = "WARN" if available_disk < 20 * GIB else "PASS"
    checks.append(
        DoctorCheck(
            "Free disk",
            disk_status,
            f"{available_disk / GIB:.1f} GiB available; keep at least 20 GiB free.",
        )
    )

    if memory is None:
        checks.append(
            DoctorCheck("RAM", "WARN", "Could not determine physical memory.")
        )
    else:
        ram_status = "PASS" if memory >= 8 * GIB else "WARN"
        checks.append(
            DoctorCheck(
                "RAM",
                ram_status,
                f"{memory / GIB:.1f} GiB total; macOS and a browser typically "
                "use 4–5 GiB.",
            )
        )

    py_ok = py_version >= (3, 11, 0)
    checks.append(
        DoctorCheck(
            "Python",
            "PASS" if py_ok else "FAIL",
            f"{py_version[0]}.{py_version[1]}.{py_version[2]}; Python 3.11+ "
            "is required.",
        )
    )

    detected_brew = brew_path if brew_path is not None else shutil.which("brew")
    checks.append(
        DoctorCheck(
            "Homebrew",
            "PASS" if detected_brew else "WARN",
            detected_brew or "Not found; needed only to install optional local tools.",
        )
    )

    local_server = (
        llama_server_path
        if llama_server_path is not None
        else shutil.which("llama-server")
    )
    checks.append(
        DoctorCheck(
            "Local llama-server",
            "PASS" if local_server else "WARN",
            local_server or "Not found; only needed for the optional local fallback.",
        )
    )

    if not settings_path.exists():
        checks.append(
            DoctorCheck(
                "Config permissions", "WARN", "No config yet; run qwenfree connect."
            )
        )
    elif settings_path.is_symlink():
        checks.append(
            DoctorCheck(
                "Config permissions", "FAIL", "Config file must not be a symlink."
            )
        )
    else:
        permissions = settings_path.stat().st_mode & 0o777
        mode_ok = permissions == 0o600
        checks.append(
            DoctorCheck(
                "Config permissions",
                "PASS" if mode_ok else "FAIL",
                f"{permissions:04o}; required mode is 0600.",
            )
        )

    if active_venv is None:
        checks.append(
            DoctorCheck(
                "Virtualenv size", "WARN", "No active virtual environment detected."
            )
        )
    else:
        size = directory_size(active_venv)
        checks.append(
            DoctorCheck(
                "Virtualenv size",
                "PASS" if size <= VENV_TARGET_BYTES else "WARN",
                f"{size / MIB:.1f} MiB at {active_venv}; target is below 150 MiB.",
            )
        )

    checks.append(_remote_check(settings_path))
    return checks
