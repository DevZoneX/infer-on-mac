# Contributing

Development requires an Apple Silicon Mac, Python 3.11+, and uv.

## Environment and checks

```sh
uv sync --locked --python 3.11 --extra dev
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv build
uv run --no-sync pytest
```

For audio/web dependencies, include `--extra voice --extra web` in the sync.
Build before running the full suite so package-import tests use a fresh wheel.
To format changed Python files, run `uv run --no-sync ruff format src tests`.
Update `uv.lock` with `uv lock` when dependency declarations change.

Tests must not download model weights, run inference, record microphones, or
play audio. Local HTTP fixtures and dummy child processes are used for transport
and lifecycle integration tests. Optional audio dependency tests skip when the
extra is absent. See [the test guide](tests/README.md).

CI runs on macOS arm64 with Python 3.11. It checks text-only and full optional
installations, lint, formatting, builds, and tests. Actions are pinned to reviewed
commits; version comments identify their releases. A failure in one matrix job
does not cancel the other job.

## Changes

Keep a change focused on a user-visible problem. Describe the trigger, the new
behavior, and how it was tested. Include a regression test for bugs and meaningful
boundary cases for new behavior. Update the [reference](docs/REFERENCE.md) if
commands, defaults, storage, or installation behavior change.

Keep comments for constraints and non-obvious behavior. Avoid comments that
repeat the next line, unsupported performance claims, or documentation copied
from a temporary implementation plan.

[Source layout](src/README.md) describes module boundaries. In particular:

- Keep input/rendering behavior in the UI modules and chat state in `chat.py`.
- Validate candidate settings and sessions before replacing active state.
- Preserve separate reasoning and final-answer channels.
- Recheck process identity before signals; preserve endpoint-lock and exact-child
  cleanup behavior. Do not replace these checks with broad process-name matching.
- Keep optional imports lazy and avoid background work in utility commands.
- Keep private app data, weights, recordings, and caches outside the checkout.

## Manual checks

Automated tests do not establish model or audio performance. For changes to those
paths, separately check startup, an ordinary request, cancellation, server reuse,
and shutdown with a cached model. Audio changes also need recording, silence
handling, transcription, playback, and stop-key checks on real devices. Record
which model/chip/platform was used; do not turn one measurement into a general
performance promise.

## Releases

Bump `version` in `pyproject.toml` and `__version__` in `src/mlx_chat/__init__.py`,
run `uv lock`, add a section to [CHANGELOG.md](CHANGELOG.md), and tag `vX.Y.Z`.
To re-record the README demo, run `vhs assets/demo.tape` in a project with a
failing test and a cached model.

## Issues and security

For bugs, include reproduction steps, expected/actual behavior, the commit,
platform/chip, Python, extras, and model. Redact private text and logs.
For security concerns, follow [SECURITY.md](SECURITY.md) rather than publishing
credentials or exploit details in an ordinary issue.

Contributions are covered by the repository's [MIT license](LICENSE). Do not add
model weights or assets whose redistribution rights are unclear.
