"""Regression checks for Kaggle's CUDA driver-library discovery."""

from pathlib import Path


def test_bootstrap_adds_kaggle_driver_path_for_cmake() -> None:
    script = Path(__file__).resolve().parents[1] / "server" / "bootstrap.sh"
    source = script.read_text(encoding="utf-8")

    assert '[[ -d /usr/local/nvidia/lib64 ]]' in source
    assert 'export CMAKE_LIBRARY_PATH="/usr/local/nvidia/lib64' in source

