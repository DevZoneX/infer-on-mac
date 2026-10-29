# Tests

From the repository root:

```sh
uv sync --locked --python 3.11 --extra dev
uv build
uv run --no-sync pytest
```

For optional dependencies, include `--extra voice --extra web` in the sync.
Select tests with normal pytest arguments:

```sh
uv run --no-sync pytest tests/test_chat.py
uv run --no-sync pytest tests/test_server.py -k identity
```

## Coverage

| Module | Tests |
| --- | --- |
| CLI and installation | `test_cli.py`, `test_setup.py` |
| Settings/storage | `test_config.py` |
| Chat commands/state | `test_chat.py` |
| API events/reasoning/metrics | `test_client.py` |
| Server identity/shutdown/recovery | `test_server.py`, `test_lifecycle.py` |
| Input and full-screen behavior | `test_ui.py`, `test_conversation.py` |
| Context budgets and saved results | `test_context.py`, `test_sessions.py` |
| File snapshots and generation profiles | `test_attachments.py`, `test_profiles.py` |
| Agent loop, tool-call streaming, saved tool steps | `test_agent.py` |
| Workspace tools, approvals, memory sizing | `test_tools.py` |
| Search snippets | `test_web.py` |
| Voice and speech cleanup/stop controls | `test_voice.py`, `test_tts.py` |
| Process resource samples | `test_resources.py` |
| Local HTTP, CLI processes, real dummy-child ownership, wheel imports | `test_integration.py` |

Tests import the source checkout through `conftest.py`. Tests that write app data
set `MLX_CHAT_HOME` to a temporary directory. Keep this isolation when adding tests.
Audio/network/model interfaces are mocked except for the local fixtures below.

## Integration fixtures

A loopback HTTP server exercises SSE, headers, stalled/partial events, timeouts,
disconnects, request payloads, and CLI stdout/exit codes. It contacts no external
service. A dummy Python child verifies process identity and shutdown; it runs no
inference server. Tests use direct and symlinked interpreters because OS process
listings can show a resolved executable name.

These tests require permission to bind a loopback port, inspect process details,
and signal the exact child created by the fixture. A restricted sandbox needs
those permissions for the full suite.

The wheel smoke test extracts the newest built wheel into a temporary installation
and imports it outside the checkout. It also checks optional-dependency metadata.
Build first so an old wheel is not tested. Without a wheel the smoke test skips;
CI builds before running pytest.

## Limits and conventions

Tests do not download weights, perform inference, record a microphone, or play
sound. Device behavior, model quality, and inference speed need separate manual
checks. Optional audio tests can skip without their dependencies.

Test public behavior and a useful failure/boundary case. Use a local fixture when
process or transport behavior matters; mock external services and devices. Avoid
sleep-only assertions where a readiness event or the app's render loop can be
observed. Keep fixture cleanup in `finally` blocks so failures leave no children.

```sh
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv build
uv run --no-sync pytest
```
