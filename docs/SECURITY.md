# Security and privacy

## Public tunnel and API key

The Kaggle server binds to `127.0.0.1`; Cloudflare or ngrok creates a public
HTTPS endpoint to that loopback service. Anyone who obtains the URL can reach
the endpoint, and Cloudflare documents that Quick Tunnel hostnames are public
and temporary. The llama.cpp server requires the API key for model requests,
but a public URL is not an access-control boundary. Quick Tunnels are intended
for testing and development and have no uptime guarantee. See [Cloudflare's
Quick Tunnel documentation](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

Treat the printed `QWENFREE_URL` and `QWENFREE_KEY` as credentials. Do not put
them in public notebook output, source control, screenshots, support tickets,
or chat messages. Do not send sensitive personal or confidential data through
this temporary server. Kaggle notebooks and sessions are temporary compute,
not private infrastructure.

The URL changes when the tunnel or session restarts. If a key may have leaked,
stop the notebook session and start it again with a new key. If you set
`QWENFREE_API_KEY` as a Kaggle Secret, replace that secret with a new random
value before restarting. If the notebook generated its own key, a new run
generates a new value. Re-run `qwenfree connect` on the Mac with the new URL and
key. Check Kaggle's and the tunnel provider's current terms and quotas before
use.

## Where data is stored

- The connection URL, model name, and API key are stored in
  `~/.config/qwenfree/config.toml` with mode `0600`.
- Chat history is local JSONL under `~/.local/share/qwenfree/history/`, capped
  at 50 MiB. Use `--no-history` to opt out for a command.
- An explicitly pulled local GGUF and its model manifest are stored under
  `~/.local/share/qwenfree/models/`. QwenFree never pulls a model implicitly.
- Local server runtime files and logs are under `~/.cache/qwenfree/local/`.
  `qwenfree clean` removes cache and history after confirmation, but keeps the
  config and local model.
- The web UI binds to `127.0.0.1`; the API key remains in the Python proxy and
  is not sent to browser JavaScript. The UI does not enable CORS.

## Hugging Face and tunnel credentials

`HF_TOKEN` is read from the environment (or the optional Kaggle Secret) for
Hugging Face requests. Use a least-privilege token if one is required; the
default public model does not require that you put a token in source code.
The token is not printed by the notebook.

If `NGROK_AUTHTOKEN` is configured, the notebook writes ngrok's config in the
temporary Kaggle working directory with mode `0600` and removes the tunnel
process on shutdown. Do not publish that token. Review ngrok's current
[limits](https://ngrok.com/docs/pricing-limits) and [Terms of
Service](https://ngrok.com/tos).

`.env.example` contains empty placeholders only. Do not copy live secrets into
it or commit a populated `.env` file.
