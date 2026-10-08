# Test suite guide

Tests use `pytest` and are intentionally split by module. Run them from the
project root:

```bash
uv sync --locked --extra dev
uv build
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
| `test_cli.py` | Argument parsing, model aliases, settings resolution, config keys, and one-off concise prompts. |
| `test_config.py` | Validation, saved configuration, sessions, and filesystem-safe names. |
| `test_client.py` | API payloads, malformed responses, streaming events, reasoning splitting, and result metrics. |
| `test_chat.py` | Slash commands, transactional settings/session changes, and endpoint health. |
| `test_ui.py` | Terminal sanitization, plain output, input editing, and presentation helpers. |
| `test_conversation.py` | Streaming, scrolling, static previews, selection, and keyboard controls. |
| `test_server.py` | Managed-server records, health checks, process identity, and shutdown safety. |
| `test_lifecycle.py` | Start, reuse, switching, restoration, interrupted initialization, and ephemeral cleanup. |
| `test_web.py` | Search input validation and safe bounded web context. |
| `test_tts.py` | Text cleanup, English Kokoro speech, errors, and playback cancellation. |
| `test_resources.py` | CPU/RAM snapshots and unavailable process metrics. |
| `test_context.py` | Request budgeting, complete-turn retention, and oversized prompts. |
| `test_sessions.py` | Saved-state validation, historical metrics, and result restoration. |
| `test_integration.py` | Real localhost SSE, cancellation, timeouts, CLI processes, safe dummy-child lifecycle, and wheel imports. |
| `test_setup.py` | Text-only installer defaults and explicit voice/web extras. |

`conftest.py` points imports at this source checkout. Tests that write application
state isolate it through `MLX_CHAT_HOME`, so they do not overwrite real sessions
or configuration. Text-only installations can omit `--extra voice`; tests that
need optional audio libraries skip. Add `--extra voice --extra web` for a full
installation. GitHub checks exercise both installations.

Build the wheel before running the full suite (`uv build`). The packaging smoke
test extracts that wheel into a temporary installation and imports it outside
the checkout. It skips if no wheel is available. CI builds before running tests.
Local integration tests require permission to bind a loopback port, inspect
process identity, and signal the exact dummy child created by the test. They do
not start an inference server or access external network services.

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
