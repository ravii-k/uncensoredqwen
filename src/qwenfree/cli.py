"""Terminal commands for connecting to and using QwenFree."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from qwenfree.client import QwenFreeClient, RemoteError
from qwenfree.config import ConfigError, QwenFreeConfig, load_config, save_config
from qwenfree.history import append_turn, default_history_path

app = typer.Typer(help="Use a temporary remote QwenFree server from your Mac.")
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


if __name__ == "__main__":
    app()
