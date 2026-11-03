"""Explicitly managed, optional small-model fallback for Apple Silicon."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

from qwenfree.paths import default_cache_dir, default_model_dir

GIB = 1024**3
MAX_LOCAL_MODEL_BYTES = 3 * GIB
MAX_LOCAL_PARAMETERS = 4.0
MIN_FREE_BYTES = 15 * GIB
DEFAULT_LOCAL_CONTEXT = 4096
DEFAULT_LOCAL_PORT = 8081
HF_API_ROOT = "https://huggingface.co/api/models"
HF_FILE_ROOT = "https://huggingface.co"


class LocalFallbackError(RuntimeError):
    """A safe, user-facing local fallback error."""


@dataclass(frozen=True, slots=True)
class LocalModelSelection:
    """A Hub-verified small Q4 GGUF file."""

    repo_id: str
    filename: str
    size_bytes: int
    parameters_billion: float
    parameters_source: str


def validate_repo_id(repo_id: str) -> str:
    """Require a standard Hugging Face owner/repository identifier."""
    value = repo_id.strip()
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise LocalFallbackError(
            "Repository must be a Hugging Face owner/repository id."
        )
    return value


def _safe_filename(filename: str) -> bool:
    path = PurePosixPath(filename)
    return (
        bool(filename)
        and not path.is_absolute()
        and "\\" not in filename
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _is_q4_gguf(filename: str) -> bool:
    if not filename.lower().endswith(".gguf"):
        return False
    return (
        re.search(r"(?<![a-z0-9])q4(?:[_-][a-z0-9]+)+(?![a-z0-9])", filename.lower())
        is not None
    )


def _parameters_from_metadata(info: dict[str, object]) -> tuple[float | None, str]:
    """Read an explicit parameter count from Hub metadata, when available."""
    configs = [info.get("config"), info.get("cardData")]
    for config in configs:
        if isinstance(config, dict):
            for key in ("num_parameters", "n_parameters", "parameter_count"):
                value = config.get(key)
                count = _parse_parameter_count(value)
                if count:
                    return count / 1_000_000_000, f"Hub {key} metadata"

    safetensors = info.get("safetensors")
    if isinstance(safetensors, dict):
        counts = safetensors.get("parameters")
        if isinstance(counts, dict):
            parsed = [_parse_parameter_count(value) for value in counts.values()]
            total = sum(value for value in parsed if value is not None)
            if total > 0:
                return total / 1_000_000_000, "Hub safetensors metadata"
    return None, ""


def _parse_parameter_count(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kmb])?\s*", value, re.I)
        if not match:
            return None
        multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
        return (
            float(match.group(1))
            * multiplier[match.group(2).lower() if match.group(2) else ""]
        )
    return None


def _file_size(item: object) -> int | None:
    if not isinstance(item, dict):
        return None
    size = item.get("size")
    if size is None and isinstance(item.get("lfs"), dict):
        size = item["lfs"].get("size")
    try:
        parsed = int(size)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


async def _get_json(url: str, token: str | None = None) -> dict[str, object]:
    """GET JSON with bounded retries and errors that never include credentials."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = httpx.Timeout(connect=15, read=60, write=30, pool=15)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(3):
            try:
                response = await client.get(url, headers=headers)
                if response.status_code == 404:
                    raise LocalFallbackError(
                        "Hugging Face repository or metadata was not found."
                    )
                if response.status_code in {401, 403}:
                    raise LocalFallbackError(
                        "Hugging Face denied access. Check whether the repository is "
                        "gated and HF_TOKEN is set."
                    )
                if response.status_code >= 400:
                    if response.status_code < 500 and response.status_code not in {
                        408,
                        429,
                    }:
                        raise LocalFallbackError(
                            f"Hugging Face returned HTTP {response.status_code}."
                        )
                    response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise LocalFallbackError(
                        "Hugging Face returned invalid repository metadata."
                    )
                return payload
            except LocalFallbackError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                if attempt == 2:
                    raise LocalFallbackError(
                        "Hugging Face could not be reached after three attempts."
                    ) from exc
                await asyncio.sleep(0.5 * (2**attempt))
    raise LocalFallbackError("Hugging Face request failed.")


async def verify_local_model(
    repo_id: str,
    filename: str,
    *,
    token: str | None = None,
    parameters_billion: float | None = None,
) -> LocalModelSelection:
    """Verify the repository, exact file, Q4 quant, size, and parameter limit."""
    repo = validate_repo_id(repo_id)
    if not _safe_filename(filename) or not _is_q4_gguf(filename):
        raise LocalFallbackError(
            "Choose an exact, safe .gguf filename with a Q4 quant."
        )

    repo_path = quote(repo, safe="/")
    info = await _get_json(f"{HF_API_ROOT}/{repo_path}?blobs=true", token)
    siblings = info.get("siblings")
    if not isinstance(siblings, list):
        raise LocalFallbackError("Hugging Face did not return a repository file list.")
    match = next(
        (
            item
            for item in siblings
            if isinstance(item, dict) and item.get("rfilename") == filename
        ),
        None,
    )
    if match is None:
        raise LocalFallbackError(
            "That exact file is not present in the verified repository."
        )
    size = _file_size(match)
    if size is None:
        raise LocalFallbackError(
            "Hugging Face did not provide a verified size for this file."
        )
    if size > MAX_LOCAL_MODEL_BYTES:
        raise LocalFallbackError(
            "The Q4 GGUF file is larger than the 3 GiB local limit."
        )

    reported_parameters, source = _parameters_from_metadata(info)
    if reported_parameters is None and parameters_billion is None:
        expanded = await _get_json(
            f"{HF_API_ROOT}/{repo_path}?expand%5B%5D=config&expand%5B%5D=safetensors",
            token,
        )
        reported_parameters, source = _parameters_from_metadata(expanded)
    if reported_parameters is None:
        if parameters_billion is None:
            raise LocalFallbackError(
                "Hub metadata does not report parameter count. Check the model card "
                "and "
                "provide --params-billions (maximum 4)."
            )
        reported_parameters = parameters_billion
        source = "user-supplied model card value"
    if not 0 < reported_parameters <= MAX_LOCAL_PARAMETERS:
        raise LocalFallbackError(
            "Local fallback models must have at most 4 billion parameters."
        )
    return LocalModelSelection(
        repo_id=repo,
        filename=filename,
        size_bytes=size,
        parameters_billion=reported_parameters,
        parameters_source=source,
    )


async def download_local_model(
    selection: LocalModelSelection,
    target: Path,
    *,
    token: str | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Path:
    """Stream the explicitly selected model to a private temporary file."""
    if not _safe_filename(selection.filename):
        raise LocalFallbackError("Refusing an unsafe model file path.")
    target.parent.mkdir(parents=True, exist_ok=True)
    file_path = quote(selection.filename, safe="/")
    repo_path = quote(selection.repo_id, safe="/")
    url = f"{HF_FILE_ROOT}/{repo_path}/resolve/main/{file_path}?download=true"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = httpx.Timeout(connect=20, read=120, write=30, pool=20)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(3):
            temp_path: Path | None = None
            written = 0
            try:
                async with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 404:
                        raise LocalFallbackError(
                            "The selected Hugging Face file was not found."
                        )
                    if response.status_code in {401, 403}:
                        raise LocalFallbackError(
                            "Hugging Face denied access to the selected file."
                        )
                    if response.status_code >= 400:
                        response.raise_for_status()
                    with tempfile.NamedTemporaryFile(
                        prefix=".qwenfree-download-",
                        suffix=".part",
                        dir=target.parent,
                        delete=False,
                    ) as stream:
                        temp_path = Path(stream.name)
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            if not chunk:
                                continue
                            written += len(chunk)
                            if written > selection.size_bytes:
                                raise LocalFallbackError(
                                    "Downloaded file exceeded its verified Hub size."
                                )
                            stream.write(chunk)
                            if on_progress:
                                on_progress(len(chunk))
                    if written != selection.size_bytes:
                        raise httpx.ReadError(
                            "Downloaded size did not match Hub metadata."
                        )
                os.replace(temp_path, target)
                target.chmod(0o600)
                return target
            except LocalFallbackError:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                raise
            except (httpx.HTTPError, OSError) as exc:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                if attempt == 2:
                    raise LocalFallbackError(
                        "Model download failed after three attempts; no partial file "
                        "was kept."
                    ) from exc
                await asyncio.sleep(0.5 * (2**attempt))
            except BaseException:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                raise
    raise LocalFallbackError("Model download failed.")


def model_manifest_path() -> Path:
    """Return the manifest for the currently selected local fallback."""
    return default_model_dir() / "active.json"


def save_model_manifest(selection: LocalModelSelection, model_path: Path) -> Path:
    """Atomically record the active model without storing credentials."""
    manifest = model_manifest_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps({**asdict(selection), "model_path": str(model_path)}, indent=2)
        + "\n"
    )
    descriptor, name = tempfile.mkstemp(prefix=".active-", dir=manifest.parent)
    temp = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, manifest)
        manifest.chmod(0o600)
    except OSError:
        temp.unlink(missing_ok=True)
        raise
    return manifest


def load_model_manifest() -> dict[str, object]:
    """Read active model metadata and ensure its path remains under the model dir."""
    manifest = model_manifest_path()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LocalFallbackError(
            "No local model is configured. Run qwenfree local pull first."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalFallbackError("The local model manifest is invalid.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("model_path"), str):
        raise LocalFallbackError("The local model manifest is invalid.")
    model_path = Path(data["model_path"]).resolve()
    try:
        model_path.relative_to(default_model_dir().resolve())
    except ValueError as exc:
        raise LocalFallbackError(
            "Model path is outside the QwenFree model directory."
        ) from exc
    if not model_path.is_file():
        raise LocalFallbackError("The configured local model file is missing.")
    data["model_path"] = str(model_path)
    return data


def model_target_path(repo_id: str, filename: str) -> Path:
    """Create a collision-resistant model path after validating its components."""
    repo = validate_repo_id(repo_id)
    if not _safe_filename(filename):
        raise LocalFallbackError("Refusing an unsafe model file path.")
    owner, name = repo.split("/", maxsplit=1)
    return (
        default_model_dir() / f"{owner}--{name}" / Path(*PurePosixPath(filename).parts)
    )


def require_apple_silicon() -> None:
    """Reject local inference on unsupported operating systems and architectures."""
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        raise LocalFallbackError(
            "The local fallback is supported only on Apple Silicon Macs."
        )


def find_llama_server() -> str | None:
    """Find the Homebrew llama.cpp server executable, if installed."""
    return shutil.which("llama-server")


def _runtime_dir() -> Path:
    path = default_cache_dir() / "local"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pid_file() -> Path:
    return _runtime_dir() / "llama-server.json"


def _process_command(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    command = result.stdout.strip()
    return command if result.returncode == 0 and command else None


def _read_pid_record() -> dict[str, object] | None:
    try:
        record = json.loads(_pid_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("pid"), int):
        return None
    if not isinstance(record.get("model_path"), str):
        return None
    return record


def local_server_pid() -> int | None:
    """Return the tracked PID only if it still runs the recorded local model."""
    record = _read_pid_record()
    if record is None:
        return None
    pid = int(record["pid"])
    command = _process_command(pid)
    if command and "llama-server" in command and str(record["model_path"]) in command:
        return pid
    _pid_file().unlink(missing_ok=True)
    return None


def _local_port() -> int:
    raw = os.environ.get("QWENFREE_LOCAL_PORT", str(DEFAULT_LOCAL_PORT))
    try:
        port = int(raw)
    except ValueError as exc:
        raise LocalFallbackError("QWENFREE_LOCAL_PORT must be an integer.") from exc
    return _validate_local_port(port)


def _validate_local_port(port: int) -> int:
    if not 1024 <= port <= 65535:
        raise LocalFallbackError("Local server port must be between 1024 and 65535.")
    return port


def _server_help(server_path: str) -> str:
    try:
        result = subprocess.run(
            [server_path, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalFallbackError(
            "Could not inspect llama-server command options."
        ) from exc
    return result.stdout + result.stderr


def _supports(help_text: str, option: str) -> bool:
    return option in help_text


async def local_health_check(port: int | None = None) -> bool:
    """Check the local model server without logging request details."""
    target_port = _validate_local_port(port) if port is not None else _local_port()
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"http://127.0.0.1:{target_port}/health")
            return response.is_success
    except httpx.HTTPError:
        return False


def start_local_server(server_path: str, *, port: int | None = None) -> int:
    """Start one local Metal slot with 4k context and mlock left disabled."""
    require_apple_silicon()
    model = load_model_manifest()
    target_port = port or _local_port()
    existing_pid = local_server_pid()
    if existing_pid is not None:
        if asyncio.run(local_health_check(target_port)):
            return existing_pid
        raise LocalFallbackError(
            f"A QwenFree llama-server process is already running as PID {existing_pid}."
        )

    help_text = _server_help(server_path)
    for option in ("-ngl", "-c", "--host", "--port"):
        if not _supports(help_text, option):
            raise LocalFallbackError(
                f"Installed llama-server does not support {option}; update "
                "llama.cpp with Homebrew."
            )
    if _supports(help_text, "--parallel"):
        parallel_args = ["--parallel", "1"]
    elif _supports(help_text, "-np"):
        parallel_args = ["-np", "1"]
    else:
        raise LocalFallbackError(
            "Installed llama-server has no recognized one-slot parallel option."
        )

    model_path = str(model["model_path"])
    args = [
        server_path,
        "-m",
        model_path,
        "-ngl",
        "99",
        "-c",
        str(DEFAULT_LOCAL_CONTEXT),
        *parallel_args,
        "--host",
        "127.0.0.1",
        "--port",
        str(target_port),
    ]
    runtime = _runtime_dir()
    log_path = runtime / "llama-server.log"
    try:
        with log_path.open("ab") as log_stream:
            os.chmod(log_path, 0o600)
            process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        raise LocalFallbackError("Could not start llama-server.") from exc

    ready = False
    for _ in range(30):
        if process.poll() is not None:
            break
        if asyncio.run(local_health_check(target_port)):
            ready = True
            break
        time.sleep(1)
    if not ready:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            tail = ""
        detail = f"llama-server did not become healthy. See {log_path}."
        if tail:
            detail += f"\nRecent server output:\n{tail}"
        raise LocalFallbackError(detail)

    record = {"pid": process.pid, "model_path": model_path, "port": target_port}
    _write_private_json(_pid_file(), record)
    return process.pid


def stop_local_server() -> bool:
    """Stop only the tracked llama-server process for the recorded model."""
    pid = local_server_pid()
    if pid is None:
        _pid_file().unlink(missing_ok=True)
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _pid_file().unlink(missing_ok=True)
        return False
    except OSError as exc:
        raise LocalFallbackError(
            "Could not stop the local llama-server process."
        ) from exc
    for _ in range(100):
        if _process_command(pid) is None:
            _pid_file().unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    command = _process_command(pid)
    record = _read_pid_record()
    if record and command and str(record["model_path"]) in command:
        os.kill(pid, signal.SIGKILL)
    _pid_file().unlink(missing_ok=True)
    return True


def _write_private_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".qwenfree-", dir=path.parent)
    temp = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        path.chmod(0o600)
    except OSError:
        temp.unlink(missing_ok=True)
        raise
