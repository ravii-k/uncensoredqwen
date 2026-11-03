#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-/kaggle/working}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${WORK_DIR}/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-${WORK_DIR}/model}"
BIN_DIR="${WORK_DIR}/qwenfree-bin"
LOG_DIR="${WORK_DIR}/qwenfree-logs"
SERVER_BIN="${LLAMA_CPP_DIR}/build/bin/llama-server"
CTX_SIZE="${CTX_SIZE:-8192}"
MAX_MODEL_GIB="${MAX_MODEL_GIB:-24}"
MODEL_REPO="${MODEL_REPO:-JonathanColetti/Qwen3.8-27B-Uncensored-GGUF}"
export MODEL_REPO MAX_MODEL_GIB

SERVER_PID=""
CLOUDFLARED_PID=""
NGROK_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  echo "Stopping QwenFree server and tunnel processes..."
  for process_id in "${CLOUDFLARED_PID}" "${NGROK_PID}" "${SERVER_PID}"; do
    if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill "${process_id}" 2>/dev/null || true
    fi
  done
  for process_id in "${CLOUDFLARED_PID}" "${NGROK_PID}" "${SERVER_PID}"; do
    if [[ -n "${process_id}" ]]; then
      wait "${process_id}" 2>/dev/null || true
    fi
  done
  exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for dependency in git cmake curl python3; do
  if ! command -v "${dependency}" >/dev/null 2>&1; then
    echo "Required command not found: ${dependency}. Use a Kaggle GPU notebook." >&2
    exit 1
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing; enable Accelerator -> GPU T4 x2 in Kaggle settings." >&2
  exit 1
fi
GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
if [[ "${GPU_COUNT}" != "2" ]]; then
  echo "Expected exactly 2 GPUs, found ${GPU_COUNT}. Select GPU T4 x2 and rerun." >&2
  nvidia-smi -L || true
  exit 1
fi
echo "Detected two GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! "${CTX_SIZE}" =~ ^[0-9]+$ ]] || (( CTX_SIZE < 1024 || CTX_SIZE > 32768 )); then
  echo "CTX_SIZE must be an integer from 1024 to 32768; native 262k context is not used." >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${BIN_DIR}" "${LOG_DIR}" "${MODEL_DIR}"
if ! python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
  echo "Installing the small server-side Hugging Face client..."
  python3 -m pip install --quiet huggingface_hub
fi

if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
  echo "Updating llama.cpp from its configured default branch..."
  git -C "${LLAMA_CPP_DIR}" pull --ff-only
else
  echo "Cloning the latest default branch of ggml-org/llama.cpp..."
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR}"
fi

echo "Building llama-server with CUDA for NVIDIA T4 (compute capability 7.5)..."
cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${LLAMA_CPP_DIR}/build" --config Release \
  --target llama-server -j 2

if [[ ! -x "${SERVER_BIN}" ]]; then
  echo "Build completed, but llama-server was not found at ${SERVER_BIN}." >&2
  exit 1
fi

SELECTION_FILE="${WORK_DIR}/qwenfree-model-selection.json"
echo "Checking the selected Hugging Face repository and GGUF sizes..."
python3 "${SCRIPT_DIR}/find_model.py" --json --max-model-gib "${MAX_MODEL_GIB}" \
  > "${SELECTION_FILE}"

python3 - "${SELECTION_FILE}" "${MODEL_DIR}" <<'PY'
import json
import os
import sys
from huggingface_hub import snapshot_download

selection_path, model_dir = sys.argv[1:]
with open(selection_path, encoding="utf-8") as source:
    selection = json.load(source)
print(
    "Downloading {} {} ({:.2f} GiB of weights) from {}".format(
        selection["quant"],
        selection["primary_file"],
        selection["total_size_bytes"] / 1024**3,
        selection["repo_id"],
    )
)
snapshot_download(
    repo_id=selection["repo_id"],
    local_dir=model_dir,
    allow_patterns=selection["files"],
    token=os.environ.get("HF_TOKEN") or None,
)
for filename in selection["files"]:
    if not os.path.isfile(os.path.join(model_dir, filename)):
        raise SystemExit("Model download is incomplete: " + filename)
PY

PRIMARY_FILE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["primary_file"])' "${SELECTION_FILE}")"
MODEL_PATH="${MODEL_DIR}/${PRIMARY_FILE}"
if [[ ! -s "${MODEL_PATH}" ]]; then
  echo "Selected model file is missing or empty: ${MODEL_PATH}" >&2
  exit 1
fi

SERVER_HELP="$("${SERVER_BIN}" --help 2>&1 || true)"
require_server_option() {
  local option="$1"
  if ! grep -Fq -- "${option}" <<<"${SERVER_HELP}"; then
    echo "The built llama-server does not support required option ${option}." >&2
    echo "Rebuild from the latest llama.cpp default branch and try again." >&2
    exit 1
  fi
}
for option in --tensor-split --flash-attn --ctx-size --host --port --api-key --jinja; do
  require_server_option "${option}"
done

SERVER_ARGS=(
  -m "${MODEL_PATH}"
  -ngl 99
  --tensor-split 1,1
  -c "${CTX_SIZE}"
  --host 127.0.0.1
  --port 8080
  --jinja
)
if grep -Fq -- '--flash-attn [on|off|auto]' <<<"${SERVER_HELP}" || \
  grep -Fq -- '--flash-attn <on|off|auto>' <<<"${SERVER_HELP}"; then
  SERVER_ARGS+=(--flash-attn on)
else
  SERVER_ARGS+=(--flash-attn)
fi
if grep -Fq -- '--cache-type-k' <<<"${SERVER_HELP}" && \
  grep -Fq -- '--cache-type-v' <<<"${SERVER_HELP}" && \
  grep -Fq 'q8_0' <<<"${SERVER_HELP}"; then
  SERVER_ARGS+=(--cache-type-k q8_0 --cache-type-v q8_0)
  echo "Using q8_0 KV cache because this build advertises both cache flags."
else
  echo "This llama-server build does not advertise q8_0 KV cache; using its default KV cache."
fi
if grep -Fq -- '--parallel' <<<"${SERVER_HELP}"; then
  SERVER_ARGS+=(--parallel 1)
fi

if [[ -n "${QWENFREE_API_KEY:-}" ]]; then
  API_KEY="${QWENFREE_API_KEY}"
else
  API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
fi

echo "Starting llama-server on loopback..."
"${SERVER_BIN}" "${SERVER_ARGS[@]}" --api-key "${API_KEY}" \
  > "${LOG_DIR}/llama-server.log" 2>&1 &
SERVER_PID=$!

SERVER_READY="false"
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    break
  fi
  if curl --silent --show-error --fail --max-time 5 \
    http://127.0.0.1:8080/health >/dev/null 2>&1; then
    SERVER_READY="true"
    break
  fi
  sleep 5
done
if [[ "${SERVER_READY}" != "true" ]]; then
  if grep -Eqi 'unknown (model )?architecture' "${LOG_DIR}/llama-server.log"; then
    echo "Unknown model architecture: rebuild llama.cpp from its latest master/default branch." >&2
  else
    echo "llama-server did not become healthy. Recent server log follows:" >&2
  fi
  tail -n 60 "${LOG_DIR}/llama-server.log" >&2 || true
  exit 1
fi

CLOUDFLARED_BIN="${BIN_DIR}/cloudflared"
if [[ ! -x "${CLOUDFLARED_BIN}" ]]; then
  echo "Downloading cloudflared linux-amd64 from Cloudflare's official GitHub release..."
  curl --fail --location --silent --show-error \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    --output "${CLOUDFLARED_BIN}"
  chmod 700 "${CLOUDFLARED_BIN}"
fi
"${CLOUDFLARED_BIN}" --version
"${CLOUDFLARED_BIN}" tunnel --url http://127.0.0.1:8080 \
  > "${LOG_DIR}/cloudflared.log" 2>&1 &
CLOUDFLARED_PID=$!

PUBLIC_URL=""
for _ in $(seq 1 30); do
  PUBLIC_URL="$(grep -Eo 'https://[[:alnum:]-]+\.trycloudflare\.com' \
    "${LOG_DIR}/cloudflared.log" | head -n 1 || true)"
  if [[ -n "${PUBLIC_URL}" ]]; then
    break
  fi
  if ! kill -0 "${CLOUDFLARED_PID}" 2>/dev/null; then
    break
  fi
  sleep 2
done

if [[ -z "${PUBLIC_URL}" && -n "${NGROK_AUTHTOKEN:-}" ]]; then
  echo "Cloudflare quick tunnel did not provide a URL; trying the optional ngrok fallback..."
  NGROK_DIR="${BIN_DIR}/ngrok"
  mkdir -p "${NGROK_DIR}"
  if [[ ! -x "${NGROK_DIR}/ngrok" ]]; then
    curl --fail --location --silent --show-error \
      https://bin.ngrok.com/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz \
      --output "${NGROK_DIR}/ngrok.tgz"
    tar -xzf "${NGROK_DIR}/ngrok.tgz" -C "${NGROK_DIR}"
    chmod 700 "${NGROK_DIR}/ngrok"
  fi
  NGROK_CONFIG="${WORK_DIR}/qwenfree-ngrok.yml"
  "${NGROK_DIR}/ngrok" config add-authtoken "${NGROK_AUTHTOKEN}" \
    --config "${NGROK_CONFIG}" >/dev/null
  chmod 600 "${NGROK_CONFIG}"
  "${NGROK_DIR}/ngrok" http 8080 --config "${NGROK_CONFIG}" \
    --log stdout --log-format json > "${LOG_DIR}/ngrok.log" 2>&1 &
  NGROK_PID=$!
  for _ in $(seq 1 30); do
    PUBLIC_URL="$(python3 - <<'PY'
import json
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1) as response:
        tunnels = json.load(response).get("tunnels", [])
    print(next((item["public_url"] for item in tunnels if item.get("proto") == "https"), ""))
except (OSError, urllib.error.URLError, json.JSONDecodeError):
    print("")
PY
    )"
    if [[ -n "${PUBLIC_URL}" ]]; then
      break
    fi
    sleep 2
  done
fi

if [[ -z "${PUBLIC_URL}" ]]; then
  echo "No public tunnel URL was obtained. Recent tunnel log follows:" >&2
  tail -n 40 "${LOG_DIR}/cloudflared.log" >&2 || true
  if [[ -z "${NGROK_AUTHTOKEN:-}" ]]; then
    echo "Set the optional NGROK_AUTHTOKEN Kaggle Secret to enable ngrok fallback." >&2
  fi
  exit 1
fi

cat <<EOF

============================================================
QwenFree server is ready. Keep this Kaggle cell running.
QWENFREE_URL=${PUBLIC_URL}
QWENFREE_KEY=${API_KEY}

On your Mac, run:
qwenfree connect ${PUBLIC_URL} --key ${API_KEY}
============================================================
EOF

while kill -0 "${SERVER_PID}" 2>/dev/null; do
  sleep 300
  if curl --silent --show-error --fail --max-time 10 \
    http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') QwenFree server health: OK"
  else
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') QwenFree server health: unavailable"
  fi
done

echo "llama-server stopped; check ${LOG_DIR}/llama-server.log."
