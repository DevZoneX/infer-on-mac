# Contributing

Develop on an Apple Silicon Mac with Python 3.11+ and uv.

```bash
uv sync --locked --python 3.11 --extra dev
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv build
uv run --no-sync pytest
```

Add `--extra voice --extra web` for optional features. Audio tests that require
optional dependencies otherwise skip. Automated tests must not download models,
record microphones, play sound, or start inference servers.
Local HTTP fixtures and harmless dummy child processes are permitted for
integration tests. Build the wheel before testing to exercise packaged imports.

See [the source guide](src/README.md) for module responsibilities and
[the test guide](tests/README.md) for coverage and conventions. Keep changes
focused, add regression tests, and update user-facing documentation when
behavior changes. Update `uv.lock` with `uv lock` when dependencies change.

For bug reports, include the macOS version, chip, Python version, model name,
and steps to reproduce. Redact prompts, transcripts, local paths, and secrets
from logs. Do not post sensitive security reports in public issues; use GitHub
private vulnerability reporting if the repository owner has enabled it.
