# mlx-chat

`mlx-chat` is a local terminal chat app for Apple Silicon Macs. It starts an
MLX-LM server for you, streams Markdown answers, and keeps conversations on
your Mac.

```text
install → mlx-chat medium → type or speak → /stop when finished
```

## What you need

- An Apple Silicon Mac (M1 or newer) running macOS
- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) recommended; `setup.sh` also supports a normal virtual environment
- Enough disk space and unified memory for the model you choose

Inference is local. The first use of a model downloads its weights from Hugging Face.

## Install

Download or clone this repository, then open a terminal in its directory.
The project requires Apple Silicon; Intel Macs, Linux, and Windows are not
supported for inference.

From the project directory:

```bash
chmod +x setup.sh mlx-chat
./setup.sh
```

The installer adds `mlx-chat` to `~/.local/bin`. If it is not immediately
available in a new terminal, restart that terminal or add `~/.local/bin` to
your `PATH`.

For text-only use without audio dependencies:

```bash
uv sync --locked --python 3.11
uv run --no-sync mlx-chat small
```

For development (including optional audio tests):

```bash
uv sync --locked --python 3.11 --extra dev --extra voice
```

## Start chatting

For a 48 GB M4 Max, start with the medium preset:

```bash
mlx-chat medium
```

The default system prompt tailors answers to an Apple Silicon MacBook, macOS,
and zsh. Use `/system TEXT` to replace it or `/system clear` to restore the default.

Type a question and press Enter. The first launch can take a while because it
downloads and loads the model. Later launches reuse the local model server when
possible.

When you are done and want to release model memory:

```text
/stop
```

`/exit` leaves the server running; `/stop` shuts it down and releases memory.

## Choose a model

Three size presets make model selection easy. The `tiny` alias selects the
default 4B model:

| Alias | Model | Best for | Notes |
| --- | --- | --- | --- |
| `small` | `mlx-community/Qwen3-8B-4bit` | Fast everyday questions and light coding | Best responsiveness. |
| `medium` | `mlx-community/Qwen3-30B-A3B-4bit` | Default choice for a 48 GB M4 Max | MoE: strong quality with only ~3B active parameters per token. |
| `large` | `mlx-community/Llama-3.3-70B-Instruct-4bit` | Harder writing and reasoning | About 40 GB of weights; close other heavy apps and keep context modest. |

Launch one directly:

```bash
mlx-chat small
mlx-chat medium
mlx-chat large
```

Or switch during a conversation:

```text
/model medium
```

You can also pass any MLX-compatible Hugging Face model or a local model path:

```bash
mlx-chat mlx-community/Mistral-Small-24B-Instruct-2501-4bit
mlx-chat /path/to/local-mlx-model
```

The built-in default remains `mlx-community/Qwen3-4B-4bit` when no model is specified.

## Speak instead of typing

Voice input is fully local and is installed by `./setup.sh`. If you set up the
environment yourself, install its dependencies once:

```bash
uv sync --extra dev --extra voice
brew install ffmpeg
```

Then, in chat:

1. Press `F1` to start voice input and speak normally.
2. Press `F2` to stop and transcribe immediately (`Ctrl-C` also works).
3. `/voice` remains available as an alternative to the shortcut.

On MacBooks where function keys control hardware, hold `Fn` when pressing `F1`–`F4`.

Voice uses `mlx-community/whisper-large-v3-turbo`, which loads when chat starts
so the first voice prompt is ready. Its loading is shown cleanly in the startup
flow and its ready state appears in the chat panel. macOS may ask for microphone
permission. Audio and transcription stay on the Mac.

## Hear model answers

Use `/speak` or `F3` to read the latest answer in English with local
`mlx-community/Kokoro-82M-bf16` and its `af_heart` voice. The model loads on
first use and is reused afterward. Press `F4` (or `Ctrl-C`) to stop playback.
Emoji, model tokens, Markdown formatting, URLs, and fenced code are omitted.
Only one English speech model is used. FFmpeg is needed for microphone
transcription, not speech output. On first use, Kokoro may also download its
English phonemizer data; synthesis stays local afterward.

```text
/speak
/speak reasoning
/speak all
```

## Everyday controls

| What you want | Command or key |
| --- | --- |
| A fast, brief answer | `/concise on` |
| More deliberate reasoning | `/think on` |
| Change model | `/model small`, `/model medium`, or `/model large` |
| Start voice input | `F1` or `/voice` |
| Stop voice input | `F2` or `Ctrl-C` |
| Read the latest answer aloud | `F3` or `/speak` |
| Stop speaking | `F4` or `Ctrl-C` |
| Retry the last answer | `/retry` |
| Remove the latest turn | `/undo` |
| Start fresh | `/new` or `/reset` |
| Save / load a conversation | `/save NAME` / `/load NAME` |
| Show server and model details | `/status` |
| Shut down and free memory | `/stop` |

Useful keys: `Tab` completes commands, `Esc`+`Enter` inserts a newline,
`Ctrl-L` clears the display, and `Ctrl-D` exits while leaving the server running.

Every completed response includes elapsed time, time-to-first-token, throughput,
and the active server's CPU and RAM use. Markdown and links render while the
response streams.

## Commands

Type `/help` in chat for the same guide.

| Command | Purpose |
| --- | --- |
| `/web QUERY` | Search the web explicitly, then answer with linked citations. |
| `/think [on\|off\|toggle]` | Control reasoning mode. |
| `/concise [on\|off\|toggle]` | Prefer short, direct answers and skip thinking. |
| `/voice [SECONDS]` | Record a local voice prompt; default maximum is 30 seconds. |
| `/speak [answer\|reasoning\|all]` | Read the latest result in English with Kokoro. |
| `/new [NAME]`, `/reset`, `/clear` | Start fresh, clear context, or clear only the screen. |
| `/system [TEXT\|@FILE\|clear]` | Show, set, load, or clear a system prompt. |
| `/retry`, `/undo` | Regenerate or remove the latest turn. |
| `/settings`, `/stats`, `/status` | Inspect settings, the latest generation, or server state. |
| `/model [MODEL]` | Show or switch the active model. |
| `/save NAME`, `/load NAME`, `/sessions` | Manage saved conversations. |
| `/history [N]`, `/copy`, `/export` | Inspect, copy, or export your chat. |
| `/logs [N]`, `/restart` | Inspect logs or restart the managed server. |
| `/stop`, `/exit` | Stop the server and exit, or just exit. |

To send text that begins with `/`, prefix it with another slash: `//example`.

## One-off prompts and server mode

Generate once without entering chat:

```bash
mlx-chat run "Explain unified memory in three sentences."
mlx-chat run "Write a release checklist." --json
```

Run an OpenAI-compatible local server without the chat UI:

```bash
mlx-chat serve medium
# endpoint: http://127.0.0.1:8080/v1
```

Useful server commands:

```bash
mlx-chat status
mlx-chat logs --lines 100
mlx-chat restart medium
mlx-chat stop
```

Only servers started and tracked by `mlx-chat` are stopped. External MLX-LM
servers can be reused but are never taken over or shut down by this app.

## Configuration

View persistent settings:

```bash
mlx-chat config show
```

Set defaults for future launches:

```bash
mlx-chat config set model mlx-community/Qwen3-30B-A3B-4bit
mlx-chat config set think on
mlx-chat config set max-tokens 8192
```

The command-line model wins over `MLX_CHAT_MODEL`, which wins over saved
configuration. `mlx-chat config path` shows where settings are stored, and
`mlx-chat config reset` restores defaults. Use `mlx-chat COMMAND --help` for
the complete command-line reference.

## Where data lives

By default, settings, sessions, input history, exports, and server logs live in:

```text
~/Library/Application Support/mlx-chat/
```

Set `MLX_CHAT_HOME` to move this directory. Model files remain in the Hugging
Face cache (normally `~/.cache/huggingface`).

## Privacy and safety

- The server listens on `127.0.0.1` by default; prompts and answers stay local.
- Voice recording and transcription are local.
- Speech output is generated and played locally with Kokoro.
- `/web` is the exception: its search query goes to a search provider. Full pages are not fetched.
- First use of a model downloads weights from Hugging Face; inference is never sent to a cloud model.
- Saved chats and logs are ordinary local files, not encrypted. Treat them as sensitive.
- Binding the server to a network address requires `--allow-network`. Do this only on a trusted network.

## Troubleshooting

Start here:

```bash
mlx-chat doctor
mlx-chat status
mlx-chat logs --lines 100
```

| Problem | What to do |
| --- | --- |
| First launch is slow | Model weights may be downloading or loading. Check `/logs 100`; use `--startup-timeout 1800` if needed. |
| `mlx_lm.server was not found` | Run `./setup.sh` again. |
| `/voice` says FFmpeg is missing | Run `brew install ffmpeg`. |
| No audio is captured | Allow microphone access for the terminal in macOS Privacy & Security settings. |
| `/speak` says speech output is not installed | Run `uv sync --extra voice`, then restart chat. |
| `large` is unstable or slow | Close memory-heavy apps, shorten the conversation, or use `medium`. |

## Development and license

See [CONTRIBUTING.md](CONTRIBUTING.md), [the source guide](src/README.md), and
[the test guide](tests/README.md). GitHub Actions runs lint, formatting, tests,
and package builds for text-only and audio installations on Apple Silicon macOS.
Tests never download inference models or use real microphones/playback.

The application is [MIT licensed](LICENSE). Downloaded models and third-party
dependencies have their own licenses; review them before redistributing weights.
Model caches, virtual environments, recordings, and local chat data are not
part of this repository. Never commit conversations, credentials, or private logs.
