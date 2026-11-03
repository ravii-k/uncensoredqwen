# Uncensored Qwen Kaggle GPU server

`bootstrap.sh` builds llama.cpp from its current default branch for CUDA compute
capability 7.5, checks for two visible GPUs, verifies the selected GGUF through
the Hugging Face API, downloads only its required file or complete shard set,
and serves it through an authenticated temporary tunnel.

The notebook generator embeds both server scripts in
`notebooks/kaggle_server.ipynb`. The regular way to launch the server is to
open that notebook in Kaggle, enable **GPU T4 x2** and Internet, add any desired
Kaggle Secrets, and run every cell. Keep the final cell running while chatting.

The default is the community uncensored Qwen3.8-27B GGUF at
`JonathanColetti/Qwen3.8-27B-Uncensored-GGUF`; it is not an official Qwen
release. `find_model.py` asks Hugging Face for the current file list and sizes
each time. Set `MODEL_REPO` in the notebook to choose a different repository.
If the default is missing, the script lists similar uncensored GGUF repos and
stops for an explicit choice. It prefers Q4_K_M, then IQ4_XS, then Q3_K_M, and
limits model weights to 24 GiB to leave room in the 28 GiB disk budget for the
KV cache and runtime overhead.

`CTX_SIZE` defaults to 8192 and is limited to 32768; native 262k context is not
attempted. The script checks the built server's `--help` output before enabling
q8_0 KV cache flags. If a model reports an unknown architecture, rebuild from
the latest llama.cpp default branch before trying again.

Ngrok is the recommended primary tunnel: add and attach the Kaggle
`NGROK_AUTHTOKEN` Secret, and the bootstrap uses ngrok if it provides a URL.
Cloudflare Quick Tunnel is the fallback and needs no account. Cloudflare
Quick Tunnels are intended for testing, have no uptime guarantee, and do not
support Server-Sent Events (SSE). The client detects a `trycloudflare.com` URL
and asks llama-server for a completed JSON response, so output arrives when the
generation finishes. Check current ngrok plan limits and terms. Both tunnel
processes stop when the notebook cell is interrupted.

Kaggle Internet must be enabled in the active session before llama.cpp can be
cloned and Hugging Face files can be queried. When the Settings menu says
**Turn off internet**, Internet is on; if it says **Turn on internet**, enable
it. The bootstrap reports this hint if a GitHub clone or update fails.

Kaggle GPU availability, weekly quotas, and session duration change over time.
Check Kaggle's current limits and terms before use. This script reports health
but does not try to keep a session active or bypass platform limits.
