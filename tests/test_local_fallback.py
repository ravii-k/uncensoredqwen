"""Local fallback checks with all Hugging Face HTTP requests mocked."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from qwenfree.local_fallback import (
    GIB,
    LocalFallbackError,
    LocalModelSelection,
    download_local_model,
    start_local_server,
    verify_local_model,
)

INFO_URL = "https://huggingface.co/api/models/example/small-q4?blobs=true"
MODEL_FILE = "Tiny-Q4_K_M.gguf"
runner = CliRunner()


def metadata(*, size: int = 1024, parameters: int = 3_500_000_000) -> dict:
    return {
        "id": "example/small-q4",
        "siblings": [{"rfilename": MODEL_FILE, "size": size}],
        "safetensors": {"parameters": {"F16": parameters}},
    }


def test_verify_local_model_checks_hub_repo_file_quant_size_and_parameters() -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.get(INFO_URL).mock(
            return_value=httpx.Response(200, json=metadata())
        )
        selection = asyncio.run(
            verify_local_model("example/small-q4", MODEL_FILE, token="hf-placeholder")
        )

    assert selection.filename == MODEL_FILE
    assert selection.size_bytes == 1024
    assert selection.parameters_billion == 3.5
    assert selection.parameters_source == "Hub safetensors metadata"
    assert route.calls[0].request.headers["Authorization"] == "Bearer hf-placeholder"


def test_verify_local_model_requires_explicit_q4_file_and_rejects_large_file() -> None:
    with pytest.raises(LocalFallbackError, match="Q4 quant"):
        asyncio.run(verify_local_model("example/small-q4", "model-IQ4_XS.gguf"))

    with respx.mock(assert_all_called=True) as router:
        router.get(INFO_URL).mock(
            return_value=httpx.Response(200, json=metadata(size=3 * GIB + 1))
        )
        with pytest.raises(LocalFallbackError, match="3 GiB"):
            asyncio.run(verify_local_model("example/small-q4", MODEL_FILE))


def test_verify_local_model_rejects_over_four_billion_parameters() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(INFO_URL).mock(
            return_value=httpx.Response(200, json=metadata(parameters=4_100_000_000))
        )
        with pytest.raises(LocalFallbackError, match="at most 4 billion"):
            asyncio.run(verify_local_model("example/small-q4", MODEL_FILE))


def test_verify_local_model_uses_user_parameter_value_when_hub_omits_it() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(INFO_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "example/small-q4",
                    "siblings": [{"rfilename": MODEL_FILE, "size": 1024}],
                },
            )
        )
        selection = asyncio.run(
            verify_local_model("example/small-q4", MODEL_FILE, parameters_billion=2.7)
        )

    assert selection.parameters_billion == 2.7
    assert selection.parameters_source == "user-supplied model card value"


def test_download_streams_to_private_file_and_keeps_token_server_side(
    tmp_path: Path,
) -> None:
    contents = b"verified GGUF bytes"
    selection = LocalModelSelection(
        "example/small-q4", MODEL_FILE, len(contents), 3.5, "Hub metadata"
    )
    url = f"https://huggingface.co/example/small-q4/resolve/main/{MODEL_FILE}?download=true"
    target = tmp_path / MODEL_FILE
    progress: list[int] = []
    with respx.mock(assert_all_called=True) as router:
        route = router.get(url).mock(return_value=httpx.Response(200, content=contents))
        result = asyncio.run(
            download_local_model(
                selection,
                target,
                token="hf-placeholder",
                on_progress=progress.append,
            )
        )

    assert result == target
    assert target.read_bytes() == contents
    assert target.stat().st_mode & 0o777 == 0o600
    assert sum(progress) == len(contents)
    assert route.calls[0].request.headers["Authorization"] == "Bearer hf-placeholder"


def test_local_pull_requires_confirmation_before_download(
    tmp_path: Path, monkeypatch
) -> None:
    import qwenfree.cli as cli

    monkeypatch.setattr(cli, "require_apple_silicon", lambda: None)
    monkeypatch.setattr(cli, "default_model_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cli.shutil,
        "disk_usage",
        lambda _path: cli.shutil._ntuple_diskusage(20 * GIB, 0, 20 * GIB),
    )
    target = tmp_path / MODEL_FILE
    monkeypatch.setattr(cli, "model_target_path", lambda *_args: target)
    with respx.mock(assert_all_called=True) as router:
        router.get(INFO_URL).mock(return_value=httpx.Response(200, json=metadata()))
        result = runner.invoke(
            cli.app,
            [
                "local",
                "pull",
                "--repo",
                "example/small-q4",
                "--file",
                MODEL_FILE,
            ],
            input="n\n",
        )

    assert result.exit_code == 0, result.output
    assert "Canceled; no model was downloaded." in result.output
    assert not target.exists()


def test_local_server_uses_metal_context_one_slot_and_keeps_mlock_off(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return FakeProcess()

    monkeypatch.setattr("qwenfree.local_fallback.require_apple_silicon", lambda: None)
    monkeypatch.setattr(
        "qwenfree.local_fallback.load_model_manifest",
        lambda: {"model_path": str(tmp_path / "model.gguf")},
    )
    monkeypatch.setattr(
        "qwenfree.local_fallback._server_help",
        lambda _path: "-ngl N -c N --host --port --parallel N --mlock",
    )
    monkeypatch.setattr("qwenfree.local_fallback._runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "qwenfree.local_fallback._pid_file", lambda: tmp_path / "server.json"
    )
    monkeypatch.setattr("qwenfree.local_fallback.subprocess.Popen", fake_popen)

    async def healthy(_port=None):
        return True

    monkeypatch.setattr("qwenfree.local_fallback.local_health_check", healthy)

    pid = start_local_server("/opt/homebrew/bin/llama-server", port=8787)
    args = captured["args"]
    assert pid == 4321
    assert args[args.index("-ngl") + 1] == "99"
    assert args[args.index("-c") + 1] == "4096"
    assert args[args.index("--parallel") + 1] == "1"
    assert "--mlock" not in args
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert json.loads((tmp_path / "server.json").read_text())["port"] == 8787
