#!/usr/bin/env python3
"""Generate the Kaggle notebook by embedding the maintained server scripts."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks" / "kaggle_server.ipynb"


def _cell(cell_type: str, source: str, **extra: Any) -> dict[str, Any]:
    """Build one nbformat v4 cell without depending on nbformat at build time."""
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    cell.update(extra)
    return cell


def generate_notebook(output: Path = DEFAULT_OUTPUT) -> Path:
    """Write a runnable Kaggle notebook with both server files embedded."""
    embedded: dict[str, str] = {}
    for filename in ("bootstrap.sh", "find_model.py"):
        source_path = ROOT / "server" / filename
        embedded[filename] = base64.b64encode(source_path.read_bytes()).decode("ascii")

    setup_source = (
        "from pathlib import Path\n"
        "import base64\n\n"
        "server_dir = Path('/kaggle/working/qwenfree_server')\n"
        "server_dir.mkdir(parents=True, exist_ok=True)\n"
        "embedded_files = "
        + json.dumps(embedded, indent=4)
        + "\nfor filename, content in embedded_files.items():\n"
        "    file_path = server_dir / filename\n"
        "    file_path.write_bytes(base64.b64decode(content))\n"
        "    if filename.endswith('.sh'):\n"
        "        file_path.chmod(0o700)\n"
        'print(f"Prepared server scripts in {server_dir}")\n'
    )

    secrets_source = """import os
from kaggle_secrets import UserSecretsClient

def optional_secret(name):
    try:
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

for name in ('HF_TOKEN', 'QWENFREE_API_KEY', 'NGROK_AUTHTOKEN'):
    value = optional_secret(name)
    if value:
        os.environ[name] = value

# Change this value to choose another verified Hugging Face GGUF repository.
os.environ.setdefault('MODEL_REPO', 'JonathanColetti/Qwen3.8-27B-Uncensored-GGUF')
os.environ.setdefault('CTX_SIZE', '8192')
os.environ.setdefault('MAX_MODEL_GIB', '24')
print('Secret values were loaded without displaying them.')
print('Selected model repository:', os.environ['MODEL_REPO'])
"""

    notebook = {
        "cells": [
            _cell(
                "markdown",
                """# QwenFree Kaggle server

1. In Notebook options, select **Accelerator → GPU T4 x2** and turn **Internet** on.
2. Add and attach the recommended `NGROK_AUTHTOKEN` Secret for the primary
   tunnel. `HF_TOKEN` and `QWENFREE_API_KEY` are optional.
3. Review the repository and context values in the next cell, then run all cells.
4. Copy the printed `qwenfree connect` command into Terminal on your Mac.

The notebook prefers ngrok when the secret provides a URL and falls back to
Cloudflare Quick Tunnel. Cloudflare Quick Tunnels do not support Server-Sent
Events, so QwenFree uses a completed response for `trycloudflare.com` URLs.

The final cell runs the authenticated server until you interrupt it. It reports
health every five minutes and does not try to bypass Kaggle session or quota limits.
""",
            ),
            _cell("code", secrets_source),
            _cell("code", setup_source),
            _cell(
                "code",
                "%cd /kaggle/working/qwenfree_server\n!bash bootstrap.sh\n",
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return output


def main() -> None:
    """Generate the notebook at the default path or a provided path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(generate_notebook(args.output))


if __name__ == "__main__":
    main()
