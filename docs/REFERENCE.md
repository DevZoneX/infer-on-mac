# Command and configuration reference

This guide describes the source checkout. Use `mlx-chat --help` and
`mlx-chat COMMAND --help` for the options accepted by the installed version.

## Installation

From the repository root:

```sh
./setup.sh
./setup.sh --voice --web
```

The installer requires macOS on arm64. With `uv`, it uses Python 3.11 and the
committed lockfile. Without `uv`, it uses `PYTHON_BIN` (default `python3`), checks
for Python 3.11+, creates `.venv`, and installs with `pip`. The pip path resolves
from `pyproject.toml`, not `uv.lock`.

The command symlink defaults to `~/.local/bin/mlx-chat`. Set `MLX_CHAT_BIN_DIR`
to change its directory. Existing files and symlinks to other targets are not
replaced. The checkout must remain at its installed location because the link
points into its environment. After moving it, rerun setup and resolve an old
symlink conflict. `./mlx-chat` also works directly from the checkout.

```sh
MLX_CHAT_BIN_DIR="$HOME/bin" ./setup.sh
PYTHON_BIN=python3.12 ./setup.sh
```

`PYTHON_BIN` affects only installation without uv.

### Direct uv installation

```sh
uv sync --locked --python 3.11
uv run --no-sync mlx-chat small
uv sync --locked --extra voice --extra web
```

Pass every extra you want to retain. A later `uv sync` with fewer extras removes
packages that are no longer selected. Include `--extra dev` to keep developer
tools. Voice transcription also requires FFmpeg on `PATH`.

### Direct pip installation

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/mlx-chat small
```

For extras, use `'.[voice,web]'` or `'.[dev,voice,web]'` as the install target.
Quote targets with brackets in zsh.

### Updating and uninstalling

Stop managed servers before replacing their environment, then update the source
and rerun setup with the extras to retain:

```sh
git pull --ff-only
./setup.sh --voice --web
```

Removing the command symlink and checkout does not remove saved conversations or
Hugging Face caches. Back up data before deleting it separately. There is no
automatic model-cache deletion command.

## CLI commands

| Command | Behavior |
| --- | --- |
| `mlx-chat [MODEL]` | Short form of `chat`; start/reuse a server and open chat. |
| `chat [MODEL]` | Explicit interactive mode. |
| `run PROMPT [MODEL]` | Generate once; `-` reads the prompt from stdin. |
| `serve [MODEL]`, `start [MODEL]` | Start/reuse a background server and print the endpoint. |
| `restart [MODEL]` | Restart a managed endpoint; attempt to restore its previous configuration if replacement fails. |
| `stop` | Stop tracked servers; `--port N` limits the selection. |
| `status` | List tracked servers and probe their health. |
| `logs` | Show the latest managed log; `--follow` follows new output. |
| `models` | List shortcuts and cached Hugging Face model repositories. |
| `doctor` | Check the platform, Python, mlx-lm/server command, available memory/disk, and tracked servers. |
| `config show` | Show saved settings with defaults filled in. |
| `config path` | Print the configuration filename. |
| `config set KEY VALUE` | Save one validated setting. |
| `config reset` | Remove saved configuration; sessions and profiles remain. |

`models` lists all cached model repositories, including audio and non-MLX models;
it is not a compatibility test. `doctor` does not download models, run inference,
or test audio devices.

### Runtime options

These options are accepted by `chat`, `run`, `serve`/`start`, and `restart`.
Generation settings affect requests; server mode does not run a full chat.

| Option | Purpose |
| --- | --- |
| `-m MODEL`, `--model-name MODEL` | Explicit model; overrides a positional model. |
| `--host HOST`, `--port N` | Bind/connect endpoint; defaults `127.0.0.1:8080`. |
| `--profile NAME` | Built-in or saved generation profile. |
| `--think`, `--no-think` | Reasoning mode; support depends on the model/template. |
| `--concise`, `--no-concise` | Brief direct answers; concise mode skips thinking. |
| `--temperature N` | Sampling temperature. |
| `--top-p N`, `--top-k N`, `--min-p N` | Sampling filters. |
| `--max-tokens N` | Output token limit. |
| `--context-tokens N` | Request context budget. |
| `--system TEXT` | Literal system prompt. |
| `--startup-timeout N` | Timeout used for health/loading and warm-up. |
| `--connect-timeout N` | Streaming socket connection timeout. |
| `--request-timeout N` | Inactivity timeout during socket reads. |
| `--mlx-arg ARG` | Additional server argument; repeatable. Use `--mlx-arg=--flag` for a flag. |
| `--allow-network` | Permit a non-loopback host. |
| `--no-warmup` | Skip first-generation validation before reporting readiness. |

Extra server arguments cannot override the managed model, host, or port options,
including their recognized abbreviations. Consult `mlx_lm.server --help` for the
arguments supported by your installed version.

Additional `chat` options:

| Option | Purpose |
| --- | --- |
| `--session NAME` | Load a named session or create one for autosaving. |
| `--file PATH` | Queue file/project excerpts for the next new prompt; repeatable. |
| `--attach` | Require an existing healthy endpoint; do not start a server. |
| `--ephemeral` | Stop a newly started managed instance on exit. |
| `--plain` | Print completed responses without the full-screen view. |
| `--preload-voice`, `--no-preload-voice` | Override the voice-preload preference. |

Additional `run` options: `--file PATH` (repeatable), `--attach`, `--ephemeral`,
`--json`, and `--show-reasoning`.

`serve`, `stop`, `status`, `models`, and `doctor` also accept `--json`.
`stop` accepts `--all` (the default), `--port N`, and `--grace SECONDS`.
`logs` accepts `-n N`/`--lines N` (1–10000, default 80) and `-f`/`--follow`.
`--version` prints the package version.

### Precedence

Model selection: explicit `--model-name`, then positional model, then
`MLX_CHAT_MODEL`, then saved configuration, then the built-in default. Model
shortcuts work in saved configuration too.

Other settings start from saved defaults. A selected profile replaces generation
fields; explicit CLI options override the profile. A loaded session restores its
saved generation settings and instructions. At startup, an explicit `--profile`
or generation option overrides the loaded session. During chat, `/load` restores
saved settings; use `/profile` or other slash commands afterward to change them.

A session's saved model name is metadata. Loading a conversation does not
switch the server model automatically.

```sh
mlx-chat small --session notes --profile coding
mlx-chat run "Review this file." --file src/mlx_chat/client.py --profile coding
mlx-chat run "Summarize this input." medium --max-tokens 512 --ephemeral
printf '%s\n' 'A prompt from stdin' | mlx-chat run - --json
mlx-chat serve medium --port 8081
mlx-chat chat medium --port 8081 --attach
mlx-chat stop --port 8081
```

### JSON and exit codes

`run --json` prints one object with `model`, `answer`, `reasoning`, `usage`,
`finish_reason`, `elapsed`, `first_token_seconds`, and `tokens_per_second`.
Startup status and errors go to stderr. This output envelope does not constrain
the model's answer to JSON or validate it against a schema.

| Code | Meaning |
| --- | --- |
| `0` | Completed. A token-limited answer may report `finish_reason: length`. |
| `2` | CLI syntax, conversion, or filesystem/value error. |
| `3` | One or more `doctor` checks failed. |
| `4` | Configuration/lifecycle/application error, unhealthy status, or failed shutdown. |
| `5` | Inference/API error, or token limit reached during reasoning before a final answer. |
| `130` | Interrupted; a cancelled one-off request can still emit its partial result. |

Unexpected network EOF is an inference error, not a completed answer. Ctrl-C
retains text already received and marks the result as cancelled.

## Interactive commands

Arguments use shell-style quoting. `/attach "notes and ideas.md"` selects one
path. `//example` sends the literal message `/example` to the model.

| Command | Behavior |
| --- | --- |
| `/help` | Show commands. Typing previews usage; Enter runs the command. |
| `/agent [on\|off]` | Turn workspace tools on or off; no argument toggles. |
| `/approve [ask\|edits\|all]` | Show or set when tool calls need approval. `ask` resets "always" choices. |
| `/tools` | List agent tools, the workspace, and which tools need approval. |
| `/plan [TASK]` | Plan with read-only tools, then run (Return), revise (type changes), or keep (`n`). No argument shows the current plan. |
| `/memory [add TEXT\|reload]` | Show the loaded project memory, append a note to `AGENTS.md`, or reread the files. |
| `/compact` | Summarize all but the latest turn; only the summary is sent afterwards. |
| `/attach [PATH ...]` | Select file/project excerpts; no arguments lists pending files. |
| `/detach` | Clear pending attachments. |
| `/profile [NAME]` | List profiles or apply one. |
| `/profile save NAME`, `/profile delete NAME` | Save/delete a custom profile. |
| `/web QUERY` | Search and generate from returned snippets; requires `web`. |
| `/think [on\|off\|toggle]` | Change reasoning mode; no argument toggles it. |
| `/concise [on\|off\|toggle]` | Change concise mode; no argument toggles it. |
| `/voice [SECONDS]` | Record/transcribe; default 30, range 1–180 seconds. |
| `/speak [answer\|reasoning\|all]` | Speak the latest result; default `answer`. |
| `/new [NAME]` | Start empty, optionally named. Reject an existing session name. |
| `/reset` | Clear context and pending files; autosave empty context for a named session. |
| `/clear` | Clear the display without changing context. |
| `/system` | Display the effective instructions. |
| `/system TEXT`, `/system @FILE` | Set text or read a UTF-8 file. CLI `--system` treats its value literally. |
| `/system clear` | Restore default Mac instructions. |
| `/retry` | Replace the latest answer using its original prompt and file snapshots; restore the old answer if retry fails. |
| `/undo` | Remove the latest user/assistant turn and restore the preceding result. |
| `/settings`, `/stats`, `/status` | Show settings, result metrics, or endpoint/session/ownership details. |
| `/temperature [N]` | Show/set temperature (0–5). |
| `/top-p [N]`, `/top-k [N]` | Show/set nucleus threshold (0–1) or non-negative top-k. |
| `/max-tokens [N]` | Show/set positive output limit within the context budget. |
| `/model [MODEL]` | Show the model/shortcuts or switch models. |
| `/save [NAME]` | Save and activate a name; no argument reuses the current name. Explicit save can replace an existing file. |
| `/load NAME` | Autosave the current named session, then load another one. |
| `/sessions` | List saved sessions. |
| `/history [N]` | Show the latest N messages, not turns; default 6. |
| `/copy [answer\|reasoning\|all]` | Copy the latest assistant response with macOS `pbcopy`. |
| `/export [md\|json] [PATH]` | Export; default Markdown under the app's exports directory. |
| `/logs [N]` | Show current server logs; default 60, range 1–1000. |
| `/restart` | Restart the active managed configuration. |
| `/stop` | Autosave, stop the managed server, and exit. Refuse to stop an external endpoint. |
| `/exit` | Autosave and exit, leaving the server running. |

Aliases: `/quit`, `/q` → `/exit`; `/shutdown` → `/stop`; `/temp` → `/temperature`;
`/max` → `/max-tokens`; `/top_p` → `/top-p`; `/top_k` → `/top-k`;
`/clear-context` → `/reset`.

### Keyboard and mouse

| Input | Effect |
| --- | --- |
| Enter | Send a message or execute a command. |
| Esc, then Enter | Insert a newline. |
| Esc at the input prompt | Run `/stop`; during generation, request cancellation. |
| Ctrl-C | Clear idle input, cancel generation, or stop audio. With selected input text, copy it. |
| Ctrl-D at the prompt | Exit, leaving the server running. |
| Ctrl-L | Clear/redraw the screen. |
| Tab | Complete slash commands. |
| Ctrl-A | Select all input. |
| Ctrl-X / Ctrl-V | Cut / paste through the macOS clipboard. |
| Ctrl-U or Esc+Backspace | Clear input. |
| Page Up / Page Down | Scroll conversation or an open popup. |
| Ctrl-End | Follow the newest text again. |
| F1 / F2 / F3 | Record / speak / stop audio. |
| Mouse wheel | Scroll. |
| Drag conversation text | Select and copy it. |
| Click a thinking label | Expand/collapse the reasoning trace. |
| Click a tool line | Expand/collapse that tool's output or diff. |
| Return / `a` / `n` at an approval card | Allow once / always allow that tool / decline. Other text declines and tells the model what to do instead. |

Terminals do not report Fn separately for Escape. Hold Fn for F1–F3 if those
keys control hardware. Popups close with Enter or Esc.

The normal interface allows typing during generation. Enter with a new message
cancels the current response and submits that input next; unfinished drafts stay
in the composer. `--plain`, `NO_COLOR`, and `TERM=dumb` select completed-response
output instead of the full-screen interface.

## File and project context

```text
/attach src/mlx_chat/client.py README.md
Explain these files.
/attach src/mlx_chat
What should change in the selected excerpts?
/attach
/detach
```

`--file PATH` selects the same context at startup or for `run`. Repeat it for
multiple paths. Snapshots are read immediately; later edits to source files do
not change a stored turn. Reattach to refresh a pending file. Retry reuses the
original snapshot.

Limits and selection:

- UTF-8 text only. Directory scans skip binary/unreadable files; explicit
  selection of a non-text/unreadable file reports an error.
- At most 20 files and 16 KiB of file text in a pending selection.
- At most 8 KiB per file; larger files are clipped and marked `truncated`.
- Directory traversal examines at most 500 entries. It is a partial selection,
  not a whole-project index. The app lists included files.
- Hidden entries, symlinks within folders, common lockfiles, and directories
  `node_modules`, `venv`, `build`, `dist`, `target`, `assets`, and `__pycache__`
  are skipped. An explicitly named hidden file can be attached.
- `.gitignore` is not interpreted. Check the included list before sending,
  especially if private files have ordinary filenames.

Instructions, filenames, formatting, and output reserves count toward the
context budget too. A selection below file-size limits can still exceed it.
Use fewer files, `/detach`, or a suitable context/output limit.

Files belong to the next new user turn and remain in that turn's history until
cleared or omitted from a request by context budgeting. They are not reread or
appended anew to every later message. Failed generation retains pending files.
`/detach` clears pending selection only. `/new`, `/reset`, and `/load` also clear
pending files. The app does not execute attached code or edit files automatically.

Sessions and JSON exports include snapshots in `api_content` and filenames in
turn metadata. Markdown export contains displayed conversation text rather than
private API context. Treat saved sessions/JSON exports as containing source text.

## Prompt profiles

| Built-in | Instructions and temperature |
| --- | --- |
| `default` | Default Mac instructions; 0.7. |
| `coding` | Coding/debugging/review instructions; 0.2. |
| `writing` | Writing/editing instructions; 0.7. |

Built-ins reset generation fields to defaults before applying instructions.
Custom profiles store a complete generation snapshot. Profile fields are
`system_prompt`, `think`, `concise`, `temperature`, `top_p`, `top_k`, `min_p`,
`max_tokens`, `context_tokens`, and `repetition_penalty`. Model, endpoint,
timeouts, voice preference, and extra server arguments are unchanged.

```text
/profile coding
/temperature 0.35
/system Give short code reviews with concrete examples.
/profile save review
/profile writing
/profile review
/profile delete review
```

Built-ins cannot be overwritten/deleted. Saving a custom name replaces its
previous file. Files are plain JSON in the app's `profiles` directory. Manual
edits are validated before application; unknown keys and invalid values leave
active settings unchanged. There is no active-profile label after individual
settings change; `/settings` shows effective values.

```sh
mlx-chat small --profile coding
mlx-chat run "Edit this paragraph." --profile writing --temperature 0.5
```

## Voice input and speech output

Install `voice` and FFmpeg for microphone transcription. macOS may request
microphone permission for the terminal. Use System Settings → Privacy & Security
→ Microphone if access was denied. Input/output devices are also required.

Whisper uses `mlx-community/whisper-large-v3-turbo` and loads on first
transcription. `--preload-voice` warms it at startup; `config set preload-voice on`
saves that preference, and `--no-preload-voice` overrides it.

Recording is mono, 16 kHz, and capped by its duration. It stops after about 1.2
seconds of silence following detected speech. Initial silence waits for speech,
the limit, or F3/Ctrl-C. Finishing capture retains audio for transcription.
Native model loading/transcription finishes when Whisper returns; the audio stop
key does not interrupt that native call.

Speech output uses `mlx-community/Kokoro-82M-bf16`, English voice `af_heart`.
The model loads on first playback and stays cached in this process. `/speak`
also works for a restored session's latest answer. F3/Ctrl-C stops playback.
Playback ends with a 20 ms fade and a short silent tail. The speech popup closes
one second after normal completion or F3, including when synthesis is still
winding down. Other command popups keep
their normal Enter/Esc controls.
Code fences (including incomplete ones), URLs, formatting, model tokens, and
emoji are removed before synthesis.

First use may download model and English phonemizer data. Temporary WAV files
are deleted after transcription. Processing is local; saved data is not encrypted.

## Web search

Install `web` with `./setup.sh --web` or `uv sync --extra web`.

```text
/web MLX release notes
```

Only explicit `/web` requests search. Queries are capped at 500 characters.
`web_results` selects up to 10 usable sources (default 5). Titles/snippets are
bounded, duplicate URLs and non-HTTP(S) results are discarded. The app supplies
snippets as untrusted reference data and asks the model for exact source links.
It does not fetch complete articles or verify every generated citation.

## Configuration

```sh
mlx-chat config show
mlx-chat config set model small
mlx-chat config set max-tokens 8192
mlx-chat config set server-args '["--adapter-path", "/path/to/adapter"]'
```

Hyphens/underscores are equivalent in keys. Booleans accept `on/off`,
`true/false`, `yes/no`, or `1/0`. `server_args` takes a JSON array of strings,
not a shell command.

| Setting | Default | Accepted values |
| --- | --- | --- |
| `model` | `mlx-community/Qwen3-4B-4bit` | Non-empty ID, shortcut, or local path. |
| `host` | `127.0.0.1` | Non-empty address; non-loopback needs `--allow-network`. |
| `port` | `8080` | Integer 1–65535. |
| `think` | `false` | Boolean. |
| `concise` | `false` | Boolean; skips thinking during generation. |
| `preload_voice` | `false` | Boolean. |
| `temperature` | `0.7` | Finite number 0–5. |
| `top_p` | `0.8` | Finite number 0–1. |
| `top_k` | `20` | Non-negative integer. |
| `min_p` | `0.0` | Finite number 0–1. |
| `max_tokens` | `4096` | Positive integer. |
| `repetition_penalty` | `1.0` | Finite non-negative number. |
| `system_prompt` | Empty | Text; empty selects default Mac instructions. |
| `startup_timeout` | `900` | Positive integer, seconds. |
| `request_timeout` | `3600` | Positive integer, seconds. |
| `connect_timeout` | `10` | Positive integer, seconds. |
| `context_tokens` | `32768` | Integer greater than `max_tokens + 256`. |
| `web_results` | `5` | Integer 1–10. |
| `web_timeout` | `10` | Integer 1–60, seconds. |
| `web_region` | `us-en` | Non-empty provider region string. |
| `web_safesearch` | `moderate` | `on`, `moderate`, or `off`. |
| `agent` | `true` | Boolean; interactive chat only. `run` needs `--agent`. |
| `approval` | `ask` | `ask`, `edits`, or `all`. |
| `agent_max_steps` | `30` | Integer 1–200 model calls per turn; then the model must answer. |
| `tool_output_chars` | `12000` | Integer 1000–200000 characters kept from each tool result. |
| `compact_at` | `75` | Integer 0–100: summarize older turns when the next request would use this percentage of the budget; 0 turns it off. |
| `server_args` | `[]` | Strings excluding managed lifecycle options. |

Chat settings affect the current conversation and named-session snapshots, not
global `config.json`. Use `config set` for future launch defaults. Resetting
configuration does not remove profiles, sessions, exports, or logs.

## Agent tools

`--workspace PATH` (default: the current folder) is the only folder tools may
touch. Paths are resolved, including symlinks, and anything outside is refused.
`list_files` and `search` skip `.git`, virtualenvs, `node_modules`, caches, and
build folders; `search` uses ripgrep when installed. `read_file` returns up to 400
numbered lines per call. Writes are atomic and keep file permissions.
`run_command` uses `/bin/zsh -c` in the workspace with stdin closed, `TERM=dumb`,
pagers disabled, a 120-second default timeout (600 maximum), and 256 KiB of
captured output; cancelling or timing out stops its whole process group.

`find_files` matches names (`*.swift`) or relative paths (`src/**/*.py`).
`outline` parses Python exactly and uses patterns for JavaScript/TypeScript,
Swift, Go, Rust, Ruby, C-family languages, shell, and Markdown headings.
`fetch_url` downloads up to 2 MiB over http(s) and strips scripts, styles, and
markup. Web content is untrusted, so both web tools ask for approval unless
approval is `all`.

Project memory files (`AGENTS.md`, `CLAUDE.md`, `.mlx-chat/instructions.md`, up to
12,000 characters together) are read when agent mode starts and on `/memory
reload`. They stay fixed between messages so the prompt cache keeps working.

Planning sends the same tool list as normal work, so cached context carries over
when the plan runs; edits and commands requested during planning are refused.

Sessions save tool calls, results, diffs, and approval decisions with the turn.
`/undo` and `/retry` remove a whole agent turn. Tool results are sent back to the
model as untrusted data; review diffs before approving, since a prompt-injected
file could ask the model to do something you did not intend.

## Context budget and metrics

Requests retain system instructions and a contiguous suffix of recent complete
turns. The budget reserves the output limit and a template allowance. Older
messages are omitted from requests, not deleted from saved transcripts. The UI
reports how many messages were omitted.

The counter estimates UTF-8 bytes and per-message overhead rather than loading
a tokenizer. It may retain fewer turns than an exact counter. The fixed template
allowance may differ from a model's template. Set `context_tokens` at or below
the model's supported context size. Oversized required prompts are rejected
instead of silently shortened.

The default prompt describes an Apple Silicon Mac with macOS/zsh. Thinking mode
changes the default Qwen sampling pair from `0.7/0.8` (temperature/top-p) to
`0.6/0.95`; a customized pair is retained. Model/template support varies.
Reasoning and final content are stored separately when the API supplies separate
channels or `<think>` tags.

Agent turns count three bytes per token, which still overestimates code and
prose for the Qwen and Llama tokenizers, so long tool output fits.

Metrics include elapsed request time, first content/reasoning-token delay,
completion tokens when supplied, and completion tokens divided by elapsed time.
That average includes prompt processing/first-token wait, not decode time alone.
CPU and "GB in use" (resident memory) come from `ps` for the active tracked server
when available; this is not total device memory usage. Historical responses retain
saved generation metrics without attaching current CPU/RSS readings.

## Server lifecycle

A healthy tracked endpoint is reused when its model and extra arguments match.
Chat/startup can replace a managed configuration at the selected endpoint. A
failed switch attempts to restore the previous model/arguments; chat history is
preserved.

Processes started by the app use separate groups. The registry records the PID,
group, OS start stamp, launch options, and instance token. Shutdown rechecks
identity and arguments before signaling. A stale/recycled PID is left alone.
Interrupted startup cleans up the exact child created by its `Popen` handle.

An untracked healthy endpoint can be reused; `/restart` and `/stop` refuse to
control it. `mlx-chat stop` targets tracked servers, not every Python/MLX process.
`status` reports tracked servers, so an external endpoint may be healthy even
when that list is empty.

`--ephemeral` stops only a newly started instance from this invocation. After a
model switch, it targets the newly started replacement. Reused/external servers
are left running. Normal chat exit leaves the server running.

Startup checks health and sends a one-token warm-up. Health alone does not prove
generation works. `--no-warmup` skips that request. `startup_timeout` applies to
stages rather than one end-to-end deadline. Streaming `connect_timeout` covers
socket connection; OS hostname resolution runs in the HTTP worker.
`request_timeout` limits inactivity, not total generation duration. Cancellation
can return while an OS connect is finishing; the worker closes its socket when
that attempt returns.

Non-loopback binding requires `--allow-network`, including a host saved in
config. This app configures no API authentication. Do not expose the endpoint
directly to the public Internet.

## Storage and network access

Default directory: `~/Library/Application Support/mlx-chat/`.

| File/directory | Contents |
| --- | --- |
| `config.json` | Global defaults. |
| `profiles/*.json` | Custom generation presets. |
| `sessions/*.json` | Named transcripts, instructions, settings, and metrics. |
| `prompt_history` | Typed prompts and commands. |
| `exports/` | Default Markdown/JSON exports. |
| `logs/` | Managed server output. |
| `servers.json`, `servers.lock`, `locks/` | Process registry and lifecycle locks. |

`MLX_CHAT_HOME` moves app data. `MLX_CHAT_MODEL` overrides the saved model.
`MLX_LM_SERVER_BIN` selects a server executable. `MLX_CHAT_BIN_DIR` and
`PYTHON_BIN` affect installation. `NO_COLOR`/`TERM=dumb` select plain chat.
Hugging Face cache configuration is honored; the usual location is
`~/.cache/huggingface`, and `HF_HOME` can change it.

Local defaults keep requests on this Mac. Model/phonemizer downloads and `/web`
queries access external services. A remote endpoint receives requests sent to
it. Saved data is not encrypted. Input history and logs can contain private
text; sessions/JSON exports can contain attached source. Inspect exports before
sharing. Keep app data and model caches outside the repository.

## Troubleshooting

| Symptom | Checks and next steps |
| --- | --- |
| Command not found | Use `./mlx-chat`, add the installer directory to `PATH`, or resolve an existing command conflict. |
| Python/dependency error | Rerun setup with desired extras; inspect `mlx-chat doctor`. |
| Slow first launch | Check download/loading logs; increase `--startup-timeout` if needed. |
| Occupied port | Use another port, stop its owner, or attach to a compatible healthy server. |
| Failed model switch | Inspect logs; restoration is attempted. Use `/restart` if restoration also failed. |
| Memory pressure | Choose a smaller model, close other apps, shorten context/output, and stop unused tracked servers. |
| Context-budget error | Shorten instructions/prompt, select fewer files, or adjust limits for the model. |
| Missing/empty attachment | Check the path, encoding, exclusions, and included list. Use a smaller folder. |
| Invalid/missing profile | List profiles and check JSON/allowed fields; built-ins are read-only. |
| Missing voice support | Install `voice` and keep it selected on later syncs. |
| Missing FFmpeg | Install it and check terminal `PATH`. |
| No microphone audio | Check permission/input device; speak closer or reduce background noise. |
| Playback error | Check voice dependencies/output device; downloads may occur on first use. |
| Missing/failed search | Install `web`, check connectivity, and try a narrower query. |
| Delayed Ctrl-C | Generation transport is cancellable; native audio loading/transcription and restart tasks finish through their underlying operation. |
| External server refuses stop | Stop it through the program that started it; this app requires a tracked identity. |
| Terminal rendering trouble | Try `--plain`, resize, or restart chat. Check function-key mode for audio shortcuts. |

Bug reports should include the commit/version, macOS version, chip, Python,
installation extras, model ID, and reproduction steps. Redact private prompts,
source text, credentials, and local paths from logs.
