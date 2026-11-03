"""Terminal commands for connecting to and using QwenFree."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn
from rich.table import Table

from qwenfree.client import QwenFreeClient, RemoteError
from qwenfree.config import ConfigError, QwenFreeConfig, load_config, save_config
from qwenfree.doctor import collect_checks, directory_size
from qwenfree.history import append_turn, clear_history, default_history_path
from qwenfree.local_fallback import (
    GIB,
    MIN_FREE_BYTES,
    LocalFallbackError,
    download_local_model,
    find_llama_server,
    load_model_manifest,
    local_health_check,
    local_server_pid,
    model_target_path,
    require_apple_silicon,
    save_model_manifest,
    start_local_server,
    stop_local_server,
    verify_local_model,
)
from qwenfree.paths import default_cache_dir, default_model_dir

app = typer.Typer(help="Use a temporary remote QwenFree server from your Mac.")
local_app = typer.Typer(
    help="Manage the optional small local fallback.", no_args_is_help=True
)
app.add_typer(local_app, name="local")
console = Console()


async def _stream_to_console(
    client: QwenFreeClient,
    messages: list[dict[str, str]],
    temperature: float,
    answer_parts: list[str],
) -> None:
    """Print response deltas and collect the complete assistant turn."""
    async for chunk in client.stream_chat(messages, temperature=temperature):
        answer_parts.append(chunk)
        console.print(chunk, end="", markup=False, highlight=False)


@app.callback()
def main() -> None:
    """Manage a remote QwenFree connection and optional local fallback."""


@app.command()
def connect(
    url: str = typer.Argument(..., help="HTTPS server URL printed by the notebook."),
    key: str = typer.Option(..., "--key", help="Server API key."),
    model_name: str = typer.Option("Qwen3.8-27B", "--model-name"),
) -> None:
    """Check a server and save its URL and API key with private permissions."""
    try:
        config = QwenFreeConfig(base_url=url, api_key=key, model_name=model_name)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    if not asyncio.run(QwenFreeClient(config).health_check()):
        console.print(
            "[red]Server health check failed. Check the URL, key, and "
            "Kaggle session.[/red]"
        )
        raise typer.Exit(code=1)

    try:
        path = save_config(config)
    except OSError as exc:
        console.print(f"[red]Could not save configuration: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Connected.[/green] Private config saved to {path}")


@app.command()
def status() -> None:
    """Show the saved connection and whether its server is responding."""
    table = Table(title="QwenFree status", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Result")
    try:
        config = load_config()
    except ConfigError as exc:
        table.add_row("Configuration", f"[yellow]{exc}[/yellow]")
        console.print(table)
        raise typer.Exit(code=1) from exc

    table.add_row("Server", config.base_url)
    table.add_row("Model", config.model_name)
    healthy = asyncio.run(QwenFreeClient(config).health_check())
    table.add_row(
        "Remote health",
        "[green]PASS[/green]" if healthy else "[red]UNREACHABLE[/red]",
    )
    console.print(table)
    if not healthy:
        console.print(
            "The Kaggle session may have ended. Run the notebook again, then connect."
        )
        raise typer.Exit(code=1)


@app.command()
def doctor() -> None:
    """Check Mac compatibility, local storage, permissions, and server status."""
    table = Table(title="QwenFree doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")
    checks = collect_checks()
    for check in checks:
        color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[check.status]
        table.add_row(check.name, f"[{color}]{check.status}[/{color}]", check.detail)
    console.print(table)
    if any(check.status == "FAIL" for check in checks):
        raise typer.Exit(code=1)


@app.command()
def disk() -> None:
    """List QwenFree configuration, history, model, and cache storage."""
    config_path = Path.home() / ".config" / "qwenfree" / "config.toml"
    history_dir = default_history_path().parent
    locations = [
        ("Config", config_path),
        ("History", history_dir),
        ("Local model(s)", default_model_dir()),
        ("Cache", default_cache_dir()),
    ]
    table = Table(title="QwenFree disk usage", show_header=True, header_style="bold")
    table.add_column("Data")
    table.add_column("Path")
    table.add_column("Size", justify="right")
    total = 0
    for label, path in locations:
        size = directory_size(path)
        total += size
        table.add_row(label, str(path), _format_size(size))
    table.add_row("Total", "", _format_size(total), style="bold")
    console.print(table)


@app.command()
def clean(
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm without prompting."),
) -> None:
    """Delete chat history and cache after confirmation; preserve config and models."""
    if not yes and not typer.confirm(
        "Delete QwenFree chat history and cache?", default=False
    ):
        console.print("Canceled; nothing was deleted.")
        return

    history_path = default_history_path()
    history_dir = history_path.parent
    cache_dir = default_cache_dir()
    if history_dir.exists() or history_dir.is_symlink():
        if history_dir.is_symlink():
            history_dir.unlink()
        else:
            clear_history(history_path)
            shutil.rmtree(history_dir, ignore_errors=True)
        console.print(f"Removed history: {history_dir}")
    else:
        console.print("No chat history found.")
    if cache_dir.exists() or cache_dir.is_symlink():
        if cache_dir.is_symlink():
            cache_dir.unlink()
        else:
            shutil.rmtree(cache_dir)
        console.print(f"Removed cache: {cache_dir}")
    else:
        console.print("No cache found.")


@app.command()
def ui(
    port: int = typer.Option(8765, min=1024, max=65535, help="Local web port."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a tab."),
) -> None:
    """Run the local chat page and proxy on 127.0.0.1."""
    import uvicorn

    url = f"http://127.0.0.1:{port}"
    console.print(f"QwenFree local UI: {url} (Ctrl+C to stop)")
    if not no_browser:
        threading.Timer(0.8, webbrowser.open_new_tab, args=(url,)).start()
    uvicorn.run(
        "qwenfree.web.app:app",
        host="127.0.0.1",
        port=port,
        access_log=False,
        log_level="warning",
    )


@local_app.command("pull")
def local_pull(
    repo: str | None = typer.Option(None, "--repo", envvar="QWENFREE_LOCAL_REPO"),
    file: str | None = typer.Option(None, "--file", envvar="QWENFREE_LOCAL_FILE"),
    params_billions: float | None = typer.Option(
        None, "--params-billions", envvar="QWENFREE_LOCAL_PARAMS_BILLIONS"
    ),
) -> None:
    """Verify and explicitly download a selected small Q4 GGUF."""
    try:
        require_apple_silicon()
        if not repo or not file:
            raise LocalFallbackError(
                "Choose a small model explicitly with --repo and --file (or set "
                "QWENFREE_LOCAL_REPO and QWENFREE_LOCAL_FILE)."
            )
        token = os.environ.get("HF_TOKEN") or None
        selection = asyncio.run(
            verify_local_model(
                repo,
                file,
                token=token,
                parameters_billion=params_billions,
            )
        )
        model_root = default_model_dir()
        model_root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(model_root).free
        if free_bytes < MIN_FREE_BYTES:
            raise LocalFallbackError(
                f"Only {free_bytes / GIB:.1f} GiB is free; local pull requires "
                "at least 15 GiB."
            )
        target = model_target_path(selection.repo_id, selection.filename)
    except (LocalFallbackError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Verified local model", show_header=True, header_style="bold")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Repository", selection.repo_id)
    table.add_row("File", selection.filename)
    table.add_row("Q4 file size", _format_size(selection.size_bytes))
    table.add_row(
        "Parameters",
        f"{selection.parameters_billion:g}B ({selection.parameters_source})",
    )
    table.add_row("Free disk", f"{free_bytes / GIB:.1f} GiB")
    console.print(table)
    if not typer.confirm("Download this model to the Mac?", default=False):
        console.print("Canceled; no model was downloaded.")
        return

    try:
        if target.is_file() and target.stat().st_size == selection.size_bytes:
            model_path = target
            console.print("That exact model file is already present.")
        else:
            with Progress(
                "{task.description}",
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "Downloading selected GGUF", total=selection.size_bytes
                )
                model_path = asyncio.run(
                    download_local_model(
                        selection,
                        target,
                        token=token,
                        on_progress=lambda amount: progress.update(
                            task, advance=amount
                        ),
                    )
                )
        save_model_manifest(selection, model_path)
    except (LocalFallbackError, OSError) as exc:
        console.print(f"[red]Local model pull failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Local fallback ready:[/green] {model_path}")


@local_app.command("start")
def local_start() -> None:
    """Start the selected Q4 model with Metal, 4k context, and one slot."""
    try:
        require_apple_silicon()
        load_model_manifest()
    except LocalFallbackError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    server_path = find_llama_server()
    if server_path is None:
        brew = shutil.which("brew")
        if brew is None:
            console.print("[red]Install Homebrew first; llama.cpp is optional.[/red]")
            raise typer.Exit(code=1)
        if not typer.confirm("Install llama.cpp with Homebrew now?", default=False):
            console.print("Canceled; llama.cpp was not installed.")
            return
        try:
            subprocess.run([brew, "install", "llama.cpp"], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            console.print(f"[red]Homebrew could not install llama.cpp: {exc}[/red]")
            raise typer.Exit(code=1) from exc
        server_path = find_llama_server()
        if server_path is None:
            console.print(
                "[red]Homebrew finished, but llama-server was not found on PATH.[/red]"
            )
            raise typer.Exit(code=1)

    console.print(
        "[yellow]On an 8 GB Mac, macOS and a browser typically use 4–5 GB of RAM. "
        "This fallback uses Q4, a 4096-token context, one parallel slot, and leaves "
        "mlock disabled.[/yellow]"
    )
    try:
        pid = start_local_server(server_path)
    except LocalFallbackError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    port = int(os.environ.get("QWENFREE_LOCAL_PORT", "8081"))
    console.print(
        f"[green]Local server ready[/green] at http://127.0.0.1:{port} (PID {pid})."
    )


@local_app.command("stop")
def local_stop() -> None:
    """Stop the tracked local llama-server process."""
    try:
        stopped = stop_local_server()
    except LocalFallbackError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        "Local server stopped." if stopped else "Local server is not running."
    )


@local_app.command("status")
def local_status() -> None:
    """Show local model, llama-server, and process status."""
    table = Table(
        title="QwenFree local fallback", show_header=True, header_style="bold"
    )
    table.add_column("Check")
    table.add_column("Result")
    server_path = find_llama_server()
    table.add_row("llama-server", server_path or "Not installed (optional)")
    try:
        model = load_model_manifest()
    except LocalFallbackError as exc:
        table.add_row("Model", str(exc))
    else:
        table.add_row("Model", f"{model['repo_id']} / {model['filename']}")
        table.add_row("Model file", str(model["model_path"]))
    pid = local_server_pid()
    if pid is None:
        table.add_row("Process", "Stopped")
    else:
        record_path = default_cache_dir() / "local" / "llama-server.json"
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            port = int(record.get("port", 8081)) if isinstance(record, dict) else 8081
        except (OSError, ValueError, TypeError):
            port = 8081
        healthy = asyncio.run(local_health_check(port))
        table.add_row(
            "Process",
            f"PID {pid}; {'healthy' if healthy else 'starting or unresponsive'}",
        )
    console.print(table)


@app.command()
def chat(
    no_history: bool = typer.Option(
        False, "--no-history", help="Do not save chat history."
    ),
) -> None:
    """Start an interactive streaming terminal chat."""
    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}]
    temperature = 0.7
    client = QwenFreeClient(config)
    console.print(
        "QwenFree chat · /reset /system <text> /save /temp <value> /ctx /exit"
    )
    try:
        while True:
            try:
                user_text = console.input("[bold green]you> [/bold green]").strip()
            except EOFError:
                break
            if not user_text:
                continue
            if user_text in {"/exit", "/quit"}:
                break
            if user_text == "/reset":
                messages = [{"role": "system", "content": system_prompt}]
                console.print("Conversation reset.")
                continue
            if user_text.startswith("/system"):
                system_prompt = user_text[len("/system") :].strip()
                messages = [{"role": "system", "content": system_prompt}]
                console.print("System prompt updated; conversation reset.")
                continue
            if user_text.startswith("/temp"):
                try:
                    value = float(user_text[len("/temp") :].strip())
                    if not 0 <= value <= 2:
                        raise ValueError
                except ValueError:
                    console.print(
                        "Use /temp with a number from 0 to 2, such as /temp 0.7."
                    )
                    continue
                temperature = value
                console.print(f"Temperature set to {temperature:g}.")
                continue
            if user_text == "/ctx":
                approximate = sum(len(message["content"]) for message in messages) // 4
                console.print(f"Approximate conversation size: {approximate:,} tokens")
                continue
            if user_text == "/save":
                if no_history:
                    console.print("History is disabled for this session.")
                else:
                    console.print(
                        f"Completed turns are saved to {default_history_path()}"
                    )
                continue
            if user_text.startswith("/"):
                console.print(
                    "Unknown command. Use /reset, /system, /save, /temp, "
                    "/ctx, or /exit."
                )
                continue

            messages.append({"role": "user", "content": user_text})
            answer_parts: list[str] = []
            console.print("[bold cyan]assistant>[/bold cyan] ", end="")
            try:
                asyncio.run(
                    _stream_to_console(client, messages, temperature, answer_parts)
                )
                console.print()
            except KeyboardInterrupt:
                console.print("\nInterrupted.")
                messages.pop()
                break
            except RemoteError:
                console.print(
                    "\n[red]Server unreachable. The Kaggle session may have ended; "
                    "run the notebook again.[/red]"
                )
                messages.pop()
                continue

            answer = "".join(answer_parts)
            messages.append({"role": "assistant", "content": answer})
            if not no_history:
                append_turn(user_text, answer)
    except KeyboardInterrupt:
        console.print("\nChat closed.")
    console.print("Chat ended.")


@app.command()
def ask(
    question: str | None = typer.Argument(
        None, help="Question, or read it from stdin."
    ),
    system: str | None = typer.Option(None, "--system", help="Optional system prompt."),
    file: Path | None = typer.Option(  # noqa: B008
        None, "--file", exists=True, dir_okay=False
    ),
    no_history: bool = typer.Option(
        False, "--no-history", help="Do not save this turn."
    ),
) -> None:
    """Ask one question; accepts piped input and optional file context."""
    prompt = question
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if file is not None:
        try:
            file_text = file.read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]Could not read {file}: {exc}[/red]")
            raise typer.Exit(code=2) from exc
        if not prompt:
            prompt = "Please help with the attached file."
        prompt = f"{prompt}\n\nFile: {file.name}\n```\n{file_text}\n```"
    if not prompt:
        console.print("Provide a question, pipe text to qwenfree ask, or use --file.")
        raise typer.Exit(code=2)

    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    parts: list[str] = []

    async def stream_response() -> None:
        async for chunk in QwenFreeClient(config).stream_chat(messages):
            parts.append(chunk)

    try:
        asyncio.run(stream_response())
    except RemoteError:
        console.print(
            "[red]Server unreachable. The Kaggle session may have ended; "
            "run the notebook again.[/red]"
        )
        raise typer.Exit(code=1) from None
    answer = "".join(parts)
    sys.stdout.write(answer + "\n")
    if not no_history:
        append_turn(prompt, answer)


def _format_size(size: int) -> str:
    """Format a byte count using a compact binary unit."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


if __name__ == "__main__":
    app()
