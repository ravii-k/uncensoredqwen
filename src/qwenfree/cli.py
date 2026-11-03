"""Command-line entry point, expanded as features are added."""

import typer

app = typer.Typer(help="Use a temporary remote QwenFree server from your Mac.")


@app.callback()
def main() -> None:
    """Manage a remote QwenFree connection and optional local fallback."""


if __name__ == "__main__":
    app()
