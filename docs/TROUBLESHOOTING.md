# Troubleshooting

## Unknown model architecture

The Kaggle bootstrap clones or updates the latest default branch of
`ggml-org/llama.cpp`. If the server log reports an unknown architecture,
interrupt the run, restart the notebook session, and rerun the cells so the
build starts from the latest upstream code. Confirm that the selected model
repository actually contains the intended architecture and compatible GGUF.
The bootstrap prints recent server logs when startup fails. See the upstream
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Out of memory while loading or generating

1. Set `CTX_SIZE=4096` in the notebook environment cell and rerun the server.
   The default is 8192; QwenFree does not attempt native 262k context.
2. If the model still does not fit, check live file sizes and lower
   `MAX_MODEL_GIB` so the selector can skip oversized Q4_K_M or IQ4_XS files
   and try a complete Q3_K_M shard set. The selector must still find a full
   model within that budget.
3. Confirm the notebook has both GPUs enabled and inspect the llama-server log.
   A file-size budget cannot guarantee runtime memory fit because the KV cache
   and server buffers also consume memory.

For a local Mac fallback, close memory-heavy apps, stop and restart the local
server, or select a smaller Q4 model. The fallback's 3 GiB file cap is not a
guarantee that every allowed file runs comfortably on an 8 GB Mac.

## Tunnel URL not printed

- Check Kaggle notebook Internet is on and wait for the server health check to
  pass before tunnel startup.
- Inspect the final cell output for Cloudflare or ngrok errors. The bootstrap
  logs are under `/kaggle/working/qwenfree-logs/` for the current session.
- If Cloudflare does not provide a URL, add and attach the optional
  `NGROK_AUTHTOKEN` Kaggle Secret. Check ngrok's current [free-plan
  limits](https://ngrok.com/docs/pricing-limits) and account status.
- Cloudflare Quick Tunnels do not support SSE. Without ngrok, QwenFree uses a
  completed response, so token-by-token output is not available even when the
  URL works. This is a documented Cloudflare limitation, not a Kaggle model
  error. See [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

## HTTP 401 or unauthorized

Copy the URL and key exactly from the current notebook run. If you set
`QWENFREE_API_KEY` in Kaggle Secrets, confirm the secret is attached to this
notebook and that the session was restarted after changing it. Run the printed
`qwenfree connect ... --key ...` command again. Do not paste the key into an
issue or public notebook. See [security guidance](SECURITY.md) for key rotation.

## SSL or certificate errors

Use the exact HTTPS tunnel URL printed by the notebook. Do not disable TLS
certificate verification. Check the Mac's date and time and update Python's
certificate bundle if the error persists. For the local fallback only, HTTP
is accepted for `127.0.0.1` or `localhost`; remote HTTP URLs are rejected.

## Kaggle GPU quota exhausted or accelerator unavailable

Check the current usage and accelerator options in Kaggle's notebook/profile
interface, along with the [GPU usage guide](https://www.kaggle.com/docs/efficient-gpu-usage)
and [Terms of Use](https://www.kaggle.com/terms). Wait for the account's quota
to reset or use the optional local fallback. Availability, session length,
queues, and quotas can change. QwenFree does not automate activity or bypass
platform limits.

## Slow startup or generation

The first run builds llama.cpp and downloads model shards. Later runs may still
need to rebuild or fetch missing files in a fresh Kaggle session. Generation
speed depends on the current T4 allocation, model quant, context, server build,
prompt length, and queue/host conditions. No fixed tokens-per-second figure is
promised. Try a smaller context or a supported smaller quant if memory is the
limiting factor. With a Cloudflare Quick Tunnel, output arrives only after the
model completes the response; use configured ngrok if token streaming is
important and available on your plan.

## Local server will not start

Run `qwenfree local status`. Confirm that a model was pulled, that `llama-server`
is installed, and that the Mac is Apple Silicon. If QwenFree asks to install
llama.cpp, decline to leave it uninstalled or confirm to use Homebrew. Check
`~/.cache/qwenfree/local/llama-server.log` for the local server output and
ensure port `8081` (or `QWENFREE_LOCAL_PORT`) is available.
