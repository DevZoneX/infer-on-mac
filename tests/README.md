# Test suite guide

Tests use `pytest` and are intentionally split by module. Run them from the
project root:

```bash
uv sync --locked --extra dev --extra voice
uv run --no-sync pytest
```

Run one file or test while working on a feature:

```bash
uv run pytest tests/test_chat.py
uv run pytest tests/test_cli.py -k preset
```

## Test map

| File | Covers |
| --- | --- |
| `test_cli.py` | Argument parsing, model aliases, settings resolution, and CLI behavior. |
| `test_config.py` | Validation, saved configuration, sessions, and filesystem-safe names. |
| `test_client.py` | API payloads, streaming events, reasoning splitting, and result metrics. |
| `test_chat.py` | Slash-command behavior and conversation state changes. |
| `test_ui.py` | Terminal sanitization, live rendering bounds, and presentation helpers. |
| `test_server.py` | Managed-server records, health checks, process identity, and shutdown safety. |
| `test_lifecycle.py` | Start, reuse, switching, restoration, and cleanup behavior. |
| `test_web.py` | Search input validation and safe bounded web context. |
| `test_tts.py` | Text cleanup, English Kokoro speech, errors, and playback cancellation. |
| `test_resources.py` | CPU/RAM snapshots and unavailable process metrics. |

`conftest.py` points imports at this source checkout. Tests that write application
state isolate it through `MLX_CHAT_HOME`, so they do not overwrite real sessions
or configuration. Text-only installations can omit `--extra voice`; tests that
need optional audio libraries skip. GitHub checks exercise both installations.

## Adding a test

1. Put the test in the file for the module whose public behavior changes.
2. Mock process, network, microphone, or server interactions; tests must not
   download models, open a microphone, or start a real MLX server.
3. Cover success and an important failure or boundary case.
4. Run formatting, linting, and the full suite before finishing:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```
