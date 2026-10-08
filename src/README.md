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
  Chat modules load only when starting chat; utility commands do not import the
  interactive UI or optional backends.
- `mlx_chat/server.py` safely launches, records, checks, and stops managed
  `mlx_lm.server` processes.
- `mlx_chat/client.py` calls the OpenAI-compatible local API and turns its
  streaming events into a final result. `transport.py` reads streams through a
  bounded queue and interrupts stalled connections when cancelled.
- `mlx_chat/chat.py` owns the conversation, slash commands, sessions, model
  switching, concise mode, and voice entry.
- `mlx_chat/ui.py` builds the terminal interface and keyboard shortcuts.
  `rendering.py` presents live Markdown, response panels, and generation metrics.
  `conversation.py` owns scrolling, simple operation modes, and static command
  previews. Plain chat renders completed responses without a second live engine.

## Supporting modules

- `config.py` defines validated settings and manages config, history, sessions,
  and exports below the application data directory.
- `context.py` selects recent complete turns within a conservative request budget
  without changing the transcript.
- `sessions.py` validates saved state and reconstructs the latest generation.
  The UI renders saved turns directly with historical metrics.
- `terminal.py` groups text sanitization and bounded macOS clipboard helpers.
  `commands.py` shares command names across components; `startup.py` provides
  the startup card without importing the interactive UI.
- `web.py` performs explicit `/web` searches and constructs bounded, untrusted
  source context for the local model. Its optional provider loads only on use.
- `voice.py` records microphone input and loads MLX Whisper on first use, or
  preloads it when explicitly configured.
- `tts.py` removes markup/emoji, lazily loads one English Kokoro voice, and lets
  Kokoro handle speech playback. Ctrl-C stops speech. No language routing or
  custom audio processing is involved.
- `resources.py` reads CPU and RAM use for the active managed MLX server so the
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

Install developer tools with `uv sync --locked --extra dev`. Add `--extra voice`
and `--extra web` to exercise optional features.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the complete workflow.
