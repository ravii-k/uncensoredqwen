# MacBook Air M2 with 8 GB unified memory

QwenFree's remote path keeps the 27B model on Kaggle. The optional Mac fallback
is a separate small model, limited to at most 4B parameters and a Q4 GGUF file
no larger than 3 GiB. It is an opt-in convenience, not a way to fit the 27B
model into 8 GB.

## Memory budget

The Mac's memory is shared by macOS, applications, model weights, llama.cpp,
the context cache, and Metal. A typical macOS desktop and browser can already
use around 4–5 GB, leaving only about 3–4 GB before model runtime overhead.
The actual available amount changes with open applications and workload.

| Verified Q4 GGUF file size | Practical reading on an 8 GB Mac |
| --- | --- |
| Under 1 GiB | Most room of the supported choices, but context and runtime still use additional memory. |
| 1–2 GiB | Moderate memory pressure is possible; close other applications and watch Activity Monitor. |
| 2–3 GiB | Near the fallback file-size ceiling; startup or generation may cause memory pressure or swap. Not guaranteed to fit comfortably. |

These are file-size bands, not promises of runtime RAM use. GGUF size and
parameter count alone do not capture KV cache, compute buffers, macOS, or
other applications. The default fallback uses a 4096-token context and one
parallel slot to keep overhead lower. If memory pressure appears, stop the
server, close other applications, and choose a smaller model. A smaller
context can help, but QwenFree intentionally keeps the default at 4096.

## Use the local fallback

Select a repository and exact Q4 GGUF filename explicitly. QwenFree checks the
live Hugging Face file list and size, prefers parameter counts from Hub
metadata, shows the result, checks for at least 15 GiB free, and asks before
downloading. If the Hub does not expose parameter count, inspect the model
card and supply it with `--params-billions`.

```bash
qwenfree local pull --repo OWNER/REPO --file MODEL-Q4_K_M.gguf
qwenfree local start
qwenfree connect http://127.0.0.1:8081 --key local
qwenfree chat
```

`llama.cpp` is optional and is installed through Homebrew only after a separate
confirmation. The server binds to `127.0.0.1`, uses Metal with `-ngl 99`,
leaves `--mlock` disabled, and runs one slot. Stop it with
`qwenfree local stop`.

## Keep storage under control

- The Mac client never downloads the remote 27B model. The Kaggle notebook
  stores its GGUF in Kaggle's temporary working directory.
- Keep 20 GiB or more free for general operation. The local pull's 15 GiB
  minimum is a safety gate, not a recommendation to fill the SSD.
- Model files are stored under `~/.local/share/qwenfree/models/`. Pull only
  models you intend to keep. Remove an unused local model from this directory
  after stopping its server.
- `qwenfree disk` reports QwenFree config, history, model, and cache sizes.
  `qwenfree clean` removes history and cache after confirmation; it preserves
  the connection config and model files. Stop the local server before cleaning
  its cache.
- History is capped at 50 MiB. Use `qwenfree chat --no-history` or
  `qwenfree ask --no-history` to skip saving new turns.

## Battery and heat

Model generation can keep the GPU and memory active. Use a hard, unobstructed
surface, close demanding apps, and keep the Mac connected to power for longer
sessions if convenient. Expect the case to warm and performance to vary with
temperature and background memory use. Stop the server when you are done.
