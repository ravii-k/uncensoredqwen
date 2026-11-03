#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

confirm() {
  local prompt="$1"
  local answer
  read -r -p "${prompt} [y/N] " answer
  [[ "${answer}" == "y" || "${answer}" == "Y" || "${answer}" == "yes" || "${answer}" == "YES" ]]
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script requires macOS." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Apple Silicon arm64 is required; found $(uname -m)." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  if ! confirm "Homebrew is missing. Install it from the official Homebrew installer?"; then
    echo "Homebrew is required to finish setup." >&2
    exit 1
  fi
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
fi

find_supported_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c \
      'import platform, sys; raise SystemExit(sys.version_info < (3, 11) or platform.machine() != "arm64")' 2>/dev/null; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN=""
if ! PYTHON_BIN="$(find_supported_python)"; then
  FORMULA="python@3.13"
  if ! brew list --versions "${FORMULA}" >/dev/null 2>&1; then
    if ! confirm "Python 3.11+ is missing. Install ${FORMULA} with Homebrew?"; then
      echo "Python 3.11 or newer is required." >&2
      exit 1
    fi
    brew install "${FORMULA}"
  fi
  PYTHON_BIN="$(brew --prefix "${FORMULA}")/bin/python3.13"
fi

if [[ -x "${VENV_PATH}/bin/python" ]] && ! "${VENV_PATH}/bin/python" -c \
  'import platform, sys; raise SystemExit(sys.version_info < (3, 11) or platform.machine() != "arm64")' 2>/dev/null; then
  if ! confirm "The existing .venv is incompatible. Remove and recreate it?"; then
    echo "Setup stopped; the existing .venv was preserved." >&2
    exit 1
  fi
  rm -rf -- "${VENV_PATH}"
fi

cd "${PROJECT_ROOT}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install --editable '.[dev]'
"${VENV_PATH}/bin/qwenfree" doctor
