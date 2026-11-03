# Run uncensored Qwen3.8-27B on Kaggle

The notebook builds llama.cpp, downloads a verified GGUF into temporary Kaggle
storage, starts an authenticated loopback server, and prints
a tunnel URL and Mac command. Keep the last notebook cell running while you
chat. This is a temporary personal server, not private or production hosting.

Before starting, review [Kaggle Notebook documentation](https://www.kaggle.com/docs/notebooks),
the [GPU usage guide](https://www.kaggle.com/docs/efficient-gpu-usage), and
[Kaggle Terms of Use](https://www.kaggle.com/terms). Account eligibility,
available accelerators, quota, queues, and session length can change. Do not
try to bypass them.

## Steps

1. Sign in or create a Kaggle account. Complete phone verification if Kaggle
   requests it before enabling internet or an accelerator.
2. In this repository, download `notebooks/kaggle_server.ipynb` to your Mac.
   In Kaggle, click **Create → New Notebook**, then in the editor choose
   **File → Import Notebook** and select that downloaded file. The menu may
   move as Kaggle updates its interface.
3. In the notebook's **Settings / Session options** pane, choose
   **Accelerator → GPU T4 x2** if it is currently offered to your account, and
   switch **Internet** on. The bootstrap requires exactly two visible GPUs.
   Kaggle may restart the session after an accelerator change; wait for it to
   come back before running cells. The Internet menu shows the action it will
   take: **Turn off internet** means Internet is already on. If the menu says
   **Turn on internet**, click it. Check again in the active session before
   starting the build.
4. Add the recommended ngrok secret from **Add-ons → Secrets → Add a new
   secret**, then attach it to this notebook. Add the other secrets if needed:

   | Secret | When to set it |
   | --- | --- |
   | `NGROK_AUTHTOKEN` | Recommended primary tunnel. If ngrok cannot provide a URL, the notebook falls back to Cloudflare Quick Tunnel. |
   | `HF_TOKEN` | Only if the selected Hugging Face repository requires access. |
   | `QWENFREE_API_KEY` | Optional fixed key. Leave unset to let the notebook generate a random one for this run. |

   Never paste secret values into a notebook cell. A configured ngrok account
   has its own terms and quotas; check the [current ngrok limits](https://ngrok.com/docs/pricing-limits).
5. Review the repository, context, and model-size values in the notebook's
   second cell. The default model repository is verified live by the server
   script. Do not change the model repository unless you have checked that it
   contains a compatible GGUF. `CTX_SIZE` defaults to `8192`. You can set it to
   a larger value before starting the server, up to `32768`, but larger
   contexts use more GPU memory and may prevent the model from loading. Set it
   to a smaller value such as `4096` if you need to reduce memory use.
6. Choose **Run → Run All** (or **Run All** in the toolbar) and wait. The first
   run builds the latest default branch of llama.cpp for T4 and downloads the
   selected model. Both steps can take a while; CUDA compilation may sit at the
   same percentage as a large kernel builds. Keep the session connected. If
   Kaggle reports that the requested accelerator is unavailable or your quota
   is exhausted, stop and check the account's current usage panel.
7. When the final cell prints `QWENFREE_URL`, `QWENFREE_KEY`, and a
   `qwenfree connect ...` line, copy that exact command into Terminal on your
   Mac. Then run `qwenfree chat` or `qwenfree ui`.
8. To end the session, interrupt the final cell or stop the notebook session.
   The server and tunnel shut down; the next run has a new URL (and a new key
   unless you supplied `QWENFREE_API_KEY`).

## What the notebook does

- Verifies two GPUs, queries Hugging Face for actual repository files and file
  sizes, and chooses a complete supported quant within its weight budget.
- Builds llama.cpp for T4 CUDA and starts `llama-server` on Kaggle loopback
  only, protected by an API key.
- Prefers ngrok when `NGROK_AUTHTOKEN` is configured and it can provide a URL;
  otherwise it falls back to a Cloudflare Quick Tunnel.
- Prints a health line every five minutes. It does not keep the session alive
  or bypass Kaggle quotas or idle limits.

Cloudflare documents that Quick Tunnels do not support Server-Sent Events
(SSE). If the fallback tunnel is used, QwenFree detects `trycloudflare.com`,
asks for a completed JSON response, and prints a notice that long answers may
take time to appear. This path defaults to 512 output tokens; set
`QWENFREE_MAX_TOKENS` in the Mac environment to change it. See [Cloudflare Quick
Tunnel limitations](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

The local browser chat has an optional **Answer tokens** field. Leave it blank
to keep the existing default; enter a value to limit the generated answer.
This does not change the server's input context. If a prompt exceeds the
context, start a new chat or shorten/split the prompt. To raise the context,
edit `CTX_SIZE` in the notebook and restart the Kaggle server. The project caps
this setting at `32768`, so larger documents still need to be split.

The model defaults to `/kaggle/temp/model`; the build and logs live under
`/kaggle/working`. These are temporary Kaggle paths, so a new session may need
to rebuild and download the model again. Keep the notebook private unless you
have a reason to share it, and never include the printed key in public notebook
output or screenshots.
