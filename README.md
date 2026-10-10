# mlx-chat

Terminal chat with local MLX models on Apple Silicon Macs. `mlx-chat` starts an
MLX-LM server, streams answers, and saves conversations when you ask it to.

The default installation supports text chat. Voice input, speech output, and
web search are optional. File context and prompt profiles need no extra packages.

## Requirements

- An Apple Silicon Mac running macOS 14 or newer
- Python 3.11 or newer; the installer uses Python 3.11 when `uv` is available
- [uv](https://docs.astral.sh/uv/getting-started/installation/) recommended
- Disk space and unified memory for your chosen model

Intel Macs, Linux, and Windows are not supported for inference. Model size,
conversation length, and other running apps all affect memory use.

## Install

```sh
git clone https://github.com/DevZoneX/infer-on-mac.git
cd infer-on-mac
./setup.sh
```

The installer creates `.venv` and links `mlx-chat` into `~/.local/bin`. If that
directory is not on your `PATH`, use `./mlx-chat` from the checkout or add:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

Optional features:

```sh
./setup.sh --voice
./setup.sh --web
./setup.sh --voice --web
```

Pass every extra you want to keep when rerunning the installer. `uv sync` removes
optional packages that were not selected. For voice transcription, also install
[FFmpeg](https://ffmpeg.org/); with Homebrew, use `brew install ffmpeg`.

See the [installation reference](docs/REFERENCE.md#installation) for direct `uv`
and `pip` commands, custom installation paths, and updating the checkout.

## Start a conversation

```sh
mlx-chat small
```

The first launch downloads the model from Hugging Face. Later launches use the
cache and reuse a compatible running server when available.

Type a message and press Enter. Use `/help` for commands. The normal interface
supports scrolling and streamed Markdown. `--plain` displays completed responses
without the full-screen view.

```text
/save notes
/exit
```

`/save notes` creates a named session; named sessions save after completed turns.
`/exit` leaves the model server running. Use `/stop` instead to shut down the
managed server, or run `mlx-chat stop` from another terminal.

An unnamed conversation is not saved automatically. Load a saved session with
`mlx-chat --session notes` or `/load notes`.

## Model shortcuts

| Shortcut | Model repository |
| --- | --- |
| `tiny` (default) | `mlx-community/Qwen3-4B-4bit` |
| `small` | `mlx-community/Qwen3-8B-4bit` |
| `medium` | `mlx-community/Qwen3-30B-A3B-4bit` |
| `large` | `mlx-community/Llama-3.3-70B-Instruct-4bit` |

These are shortcuts, not automatic memory-fit checks. You can also supply an
MLX-compatible Hugging Face model ID or a local MLX model directory.

```sh
mlx-chat medium
mlx-chat /path/to/mlx-model
```

Use `/model small` to switch during chat. The model cards describe each model's
requirements and license. The app's license does not cover downloaded weights.

## Files and profiles

Attach a text file or a directory to the next message:

```text
/attach src/mlx_chat/client.py
Explain how streaming works.
/profile coding
/attach src/mlx_chat
Suggest a focused refactor of the attached code.
```

Attachments are read-only excerpts. The app lists included files, marks clipped
files, and keeps their snapshots in the conversation. It does not execute code
or index the whole project. `/attach` lists pending files; `/detach` clears them.

Profiles set instructions and generation settings without switching the model:

```text
/profile writing
/system Help edit technical documentation.
/profile save docs
/profile docs
```

The built-in profiles are `default`, `coding`, and `writing`. See
[file context](docs/REFERENCE.md#file-and-project-context) for selection limits
and [profiles](docs/REFERENCE.md#prompt-profiles) for custom presets.

## One-off prompts

```sh
mlx-chat run "Explain unified memory briefly." --concise
mlx-chat run "Review this file." --profile coding --file src/mlx_chat/client.py
printf '%s\n' 'Summarize this text.' | mlx-chat run - --json
```

`--json` returns a result envelope containing the answer, reasoning, usage, and
metrics. It does not enforce a JSON schema on the answer. Add `--ephemeral` to
stop a server started by this invocation when it exits.

## Voice and web search

With the `voice` extra installed, F1 starts recording and F2 reads the latest
answer aloud. F3 stops recording or playback. Hold Fn for function keys if your
Mac uses them for hardware controls. The models load on first use; speech output
uses one English Kokoro voice.

Stopping speech fades the output to silence. Its popup closes automatically one
second after speech finishes or F3 is pressed; Enter or Esc can dismiss it sooner.

With the `web` extra installed, `/web QUERY` sends an explicit search query to a
search provider and gives the model the returned snippets. Normal chat does not
perform web searches.

[Voice details](docs/REFERENCE.md#voice-input-and-speech-output) include microphone
permissions, silence detection, playback controls, and startup preloading.

## Local server

```sh
mlx-chat serve small
mlx-chat status
mlx-chat logs --lines 100
mlx-chat stop
```

The default endpoint is `http://127.0.0.1:8080/v1`. `--attach` requires an existing
healthy server and prevents the app from starting one. External servers are
reused without taking over their lifecycle.

See [server lifecycle](docs/REFERENCE.md#server-lifecycle) for model switching,
ownership checks, ports, timeouts, and network binding.

## Data and privacy

Settings, profiles, saved sessions, history, exports, and logs normally live in:

```text
~/Library/Application Support/mlx-chat/
```

`MLX_CHAT_HOME` changes that location. Keep it outside the source checkout.
Model downloads use the Hugging Face cache.

Inference and audio processing use the local server by default. Downloads and
explicit `/web` searches require network access. Saved files are not encrypted;
sessions with attachments include the selected file contents. A server bound to
a network interface exposes the API to that network without authentication.

Read the [privacy reference](docs/REFERENCE.md#storage-and-network-access) and
[security policy](SECURITY.md) before sharing sessions or exposing an endpoint.

## Documentation and development

- [Command and configuration reference](docs/REFERENCE.md)
- [Troubleshooting](docs/REFERENCE.md#troubleshooting)
- [Contributing](CONTRIBUTING.md)
- [Source layout](src/README.md)
- [Test suite](tests/README.md)

The tests cover application behavior, local HTTP streaming, process ownership,
CLI output, installation, and package imports. They do not run model inference
or use audio hardware. CI checks text-only and optional-feature installations.

The [recorded demo](assets/demo-muted.mp4) shows an earlier interface. Demo media
stays in the repository and is excluded from package archives.

## License

[MIT](LICENSE). Models and dependencies retain their own licenses.
