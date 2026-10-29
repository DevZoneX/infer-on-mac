# Source layout

`mlx_chat` is the package used by the `mlx-chat` entry point and
`python -m mlx_chat`.

```text
CLI settings → server start/attach → chat controller → API stream → terminal view
```

| Module | Responsibility |
| --- | --- |
| `cli.py` | Argument parsing, configuration commands, one-off output, and startup coordination. |
| `server.py` | Endpoint/registry locks, process identity, spawn/reuse, recovery, and shutdown. |
| `client.py` | Request payloads, SSE parsing, reasoning and tool-call separation, and generation results. |
| `agent.py` | The tool-calling loop, approvals, plan mode, project memory, and step limit. |
| `tools.py` | Workspace-confined file, search, outline, edit, command, plan, and web tools with their JSON schemas. |
| `hardware.py` | Unified-memory model recommendation and the server's prompt-cache limit. |
| `transport.py` | Bounded HTTP worker queue and cancellable reads. |
| `chat.py` | Conversation state, slash commands, attachments, profiles, sessions, and model switches. |
| `ui.py` | Prompt setup, key bindings, messages, help, and transcript restoration. |
| `conversation.py` | Full-screen view, scrolling, popups, selection, cached lines, and UI/worker coordination. |
| `rendering.py` | Stream previews, reasoning traces, answer panels, and metrics. |
| `startup.py`, `theme.py` | Startup progress and colors. |
| `config.py` | Validated settings, app paths, atomic JSON writes, and session filenames. |
| `context.py` | System prompts and bounded selection of complete recent turns. |
| `sessions.py` | Saved-state validation and result reconstruction. |
| `attachments.py` | Bounded UTF-8 snapshots and project traversal. |
| `profiles.py` | Built-in and saved generation presets. |
| `terminal.py`, `commands.py` | Text sanitization, clipboard helpers, and command names. |
| `voice.py`, `tts.py`, `keyboard.py` | Whisper recording/transcription, Kokoro synthesis, faded output-stream shutdown, and audio stop keys. |
| `web.py` | Explicit searches and bounded snippet context; provider import is lazy. |
| `resources.py` | CPU/resident-memory samples for the active tracked process. |
| `errors.py` | Application, endpoint, and API errors. |
| `__init__.py`, `__main__.py` | Version/model shortcuts and module entry point. |

Utility CLI commands do not import the interactive UI or audio/search backends.
Profiles and attachments use the standard library; they start no service and
load no additional inference model.

## Constraints

A settings object is shared by the client and UI. Validate a candidate before
changing its fields. Loading a session validates messages, settings, and metrics
before replacing current state.

A server record is not sufficient by itself to send a signal. Match its process
group, start stamp, executable, and managed options. Startup failure cleanup uses
the exact `Popen` child; registry cleanup compares instance tokens. Endpoint locks
span health checks and generation warm-up.

HTTP reads run in a worker so the UI can cancel before a token arrives. The queue
is bounded; shutdown wakes connected socket reads. Native Whisper work stays on
the caller thread while the UI runs separately. Do not move it into the generic
HTTP/task worker without checking MLX thread behavior.

The transcript keeps full history, including private API context for attachments
and searches. Request budgeting selects recent complete turns without deleting
saved history. Display/export paths must not accidentally expose private API
context when they are meant to show ordinary conversation text.

Run the checks in [CONTRIBUTING.md](../CONTRIBUTING.md). The
[test guide](../tests/README.md) maps behavior to test modules.
