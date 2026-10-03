# Source code guide

This directory contains the `mlx_chat` Python package behind the `mlx-chat`
command-line program.

## How a chat request flows

```text
cli.py → server.py → client.py → chat.py → ui.py
                 │                 │
                 └── starts MLX ───┴── streams the model response
```

- `mlx_chat/cli.py` parses commands, loads settings, and starts/reuses the MLX
  server.
- `mlx_chat/server.py` safely launches, records, checks, and stops managed
  `mlx_lm.server` processes.
- `mlx_chat/client.py` calls the OpenAI-compatible local API and turns its
  streaming events into a final result.
- `mlx_chat/chat.py` owns the conversation, slash commands, sessions, model
  switching, concise mode, and voice entry.
- `mlx_chat/ui.py` renders the terminal interface, live Markdown, metrics, and
  keyboard shortcuts.

## Supporting modules

- `config.py` defines validated settings and manages config, history, sessions,
  and exports below the application data directory.
- `web.py` performs explicit `/web` searches and constructs bounded, untrusted
  source context for the local model.
- `voice.py` preloads MLX Whisper at chat startup and records microphone input.
- `tts.py` removes markup/emoji, lazily loads one English Kokoro voice, and lets
  Kokoro handle speech playback. Ctrl-C stops speech. No language routing or
  custom audio processing is involved.
- `resources.py` reads CPU and RAM use for managed MLX server processes so the
  UI can append them to each completed response.
- `errors.py` contains user-facing error types.
- `__init__.py` defines the default model and the `small`, `medium`, and
  `large` preset mappings.
- `__main__.py` supports `python -m mlx_chat`.

## Conventions

- Keep UI presentation in `ui.py`; conversation behavior belongs in `chat.py`.
- Preserve the separation between model reasoning and final assistant content.
- Treat server lifecycle changes as safety-sensitive: never signal a process
  without the ownership checks in `server.py`.
- Add or update a focused test in `../tests/` whenever behavior changes.

From the project root, run the checks with:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

Install developer tools first with `uv sync --locked --extra dev --extra voice`.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the complete workflow.
