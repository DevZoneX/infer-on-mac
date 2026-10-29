# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-29

### Added

- Agent mode, on by default in interactive chat. The tools are `list_files`,
  `find_files`, `read_file`, `search`, `outline`, `edit_file`, `write_file`, and
  `run_command`, all confined to the workspace.
- Approval cards with the diff or command, and the `ask`, `edits`, and `all` modes
  (`/approve`, `--approve`).
- `/plan` for read-only exploration and a proposed plan, plus a live checklist
  through `update_plan`.
- Project memory from `AGENTS.md`, `CLAUDE.md`, or `.mlx-chat/instructions.md`
  (`/memory`).
- `fetch_url` and `web_search` agent tools. Both need approval; `web_search` needs
  the `web` extra.
- Automatic context compaction at `compact_at` (75%) and `/compact`.
- `mlx-chat run --agent` for non-interactive agent runs.
- A model recommendation by unified memory in `mlx-chat doctor`.
- `/tools` and `/agent` commands, and the `agent_max_steps` and `tool_output_chars`
  settings.

### Changed

- The interface now has a one-line header with a context meter, an animated status
  line showing phase, time, and speed, and a new color palette.
- Tool steps show as one line each and expand on click.
- Managed servers cap the prompt cache at an eighth of unified memory (1–8 GB).
- Agent requests keep earlier messages byte-identical so the prompt cache is reused.

## [0.1.0]

### Added

- Full-screen terminal chat with streamed Markdown, separate reasoning traces, and
  metrics.
- Managed MLX-LM server: start, reuse, attach, logs, and safe shutdown.
- Model shortcuts, saved sessions, prompt profiles, and file or folder attachments.
- `mlx-chat run` for one-off prompts with JSON output.
- Optional voice input (Whisper), speech output (Kokoro), and `/web` search.

[0.2.0]: https://github.com/DevZoneX/infer-on-mac/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DevZoneX/infer-on-mac/releases/tag/v0.1.0
