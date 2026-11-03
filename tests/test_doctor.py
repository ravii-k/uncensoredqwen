"""Doctor, disk, and cleanup behavior with remote traffic mocked."""

from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from qwenfree.cli import app
from qwenfree.config import QwenFreeConfig, save_config
from qwenfree.doctor import GIB, DoctorCheck, collect_checks, directory_size

runner = CliRunner()


def test_doctor_checks_config_permissions_and_remote_health(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    save_config(QwenFreeConfig("https://server.test", "private"), config_path)
    venv_path = tmp_path / "venv"
    venv_path.mkdir()
    (venv_path / "sample.py").write_text("x = 1\n")

    with respx.mock(assert_all_called=True) as router:
        router.get("https://server.test/health").mock(
            return_value=httpx.Response(200)
        )
        checks = collect_checks(
            config_path=config_path,
            venv_path=venv_path,
            machine="arm64",
            system_name="Darwin",
            macos_version="15.0",
            free_bytes=25 * GIB,
            ram_bytes=8 * GIB,
            python_version=(3, 13, 0),
            brew_path="/opt/homebrew/bin/brew",
            llama_server_path="/opt/homebrew/bin/llama-server",
        )

    statuses = {check.name: check.status for check in checks}
    assert statuses["Architecture"] == "PASS"
    assert statuses["Config permissions"] == "PASS"
    assert statuses["Remote server"] == "PASS"
    assert statuses["Virtualenv size"] == "PASS"
    assert directory_size(venv_path) > 0


def test_doctor_warns_for_low_disk_and_fails_incompatible_platform(
    tmp_path: Path,
) -> None:
    checks = collect_checks(
        config_path=tmp_path / "missing.toml",
        venv_path=tmp_path,
        machine="x86_64",
        system_name="Linux",
        free_bytes=10 * GIB,
        ram_bytes=4 * GIB,
        python_version=(3, 9, 0),
    )
    statuses = {check.name: check.status for check in checks}

    assert statuses["Architecture"] == "FAIL"
    assert statuses["Free disk"] == "WARN"
    assert statuses["Python"] == "FAIL"
    assert statuses["Remote server"] == "WARN"


def test_doctor_command_renders_status_table(monkeypatch) -> None:
    monkeypatch.setattr(
        "qwenfree.cli.collect_checks",
        lambda: [DoctorCheck("Mock", "PASS", "ok")],
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "QwenFree doctor" in result.output
    assert "PASS" in result.output


def test_clean_removes_only_history_and_cache(
    tmp_path: Path, monkeypatch
) -> None:
    history_file = tmp_path / "history" / "chat.jsonl"
    history_file.parent.mkdir()
    history_file.write_text("{}\n")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "tmp.bin").write_bytes(b"cache")
    monkeypatch.setattr("qwenfree.cli.default_history_path", lambda: history_file)
    monkeypatch.setattr("qwenfree.cli.default_cache_dir", lambda: cache_dir)

    result = runner.invoke(app, ["clean", "--yes"])

    assert result.exit_code == 0
    assert not history_file.parent.exists()
    assert not cache_dir.exists()
