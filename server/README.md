# Kaggle GPU server

`bootstrap.sh` builds llama.cpp from its current default branch for CUDA compute
capability 7.5, checks for two visible GPUs, verifies the selected GGUF through
the Hugging Face API, downloads only its required file or complete shard set,
and serves it through an authenticated temporary tunnel.

The notebook generator embeds both server scripts in
`notebooks/kaggle_server.ipynb`. The regular way to launch the server is to
open that notebook in Kaggle, enable **GPU T4 x2** and Internet, add any desired
Kaggle Secrets, and run every cell. Keep the final cell running while chatting.

The selected model repository defaults to
`JonathanColetti/Qwen3.8-27B-Uncensored-GGUF`, but `find_model.py` always asks
the Hugging Face API for the live file list and file sizes. Set `MODEL_REPO` in
the notebook when choosing another repository. If the default repo is missing,
the script lists other matching uncensored GGUF repositories and stops so you
can choose one explicitly. It prefers Q4_K_M, then IQ4_XS, then Q3_K_M, and
limits model weights to 24 GiB to reserve about 4 GiB of the 28 GiB budget for
KV cache and runtime overhead.

`CTX_SIZE` defaults to 8192 and is limited to 32768; native 262k context is not
attempted. The script checks the built server's `--help` output before enabling
q8_0 KV cache flags. If a model reports an unknown architecture, rebuild from
the latest llama.cpp default branch before trying again.

Cloudflare Quick Tunnels use a random public URL for each session and require
no account. They are intended for testing, have no uptime guarantee, and do not
support Server-Sent Events (SSE). The client detects a `trycloudflare.com` URL
and asks llama-server for a completed JSON response, so output arrives when the
generation finishes. If a Kaggle `NGROK_AUTHTOKEN` Secret is configured, the
bootstrap tries ngrok and uses it when it returns a URL; Cloudflare remains the
fallback. This preserves token streaming when ngrok is available. Check
current ngrok plan limits and terms. Both tunnel processes stop when the
notebook cell is interrupted.

Kaggle GPU availability, weekly quotas, and session duration change over time.
Check Kaggle's current limits and terms before use. This script reports health
but does not try to keep a session active or bypass platform limits.
