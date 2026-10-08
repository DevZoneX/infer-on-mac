"""Interactive conversation controller and slash commands."""

from __future__ import annotations

import difflib
import json
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import MODEL_PRESETS, resolve_model_preset
from .client import ChatClient, GenerationResult
from .config import (
    Settings,
    app_home,
    list_sessions,
    load_session,
    normalize_session_name,
    save_session,
)
from .errors import MlxChatError
from .server import (
    ServerManager,
    StartResult,
    stop_record,
    tail_log,
)
from .tts import speak
from .ui import READ_ONLY_COMMANDS, SLASH_COMMANDS, TerminalUI, sanitize_terminal_text, short_model
from .voice import (
    DEFAULT_TRANSCRIPTION_MODEL,
    preload_transcription_model,
    record_microphone,
    transcribe_audio,
)
from .web import WebSearcher

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant running locally on an Apple Silicon MacBook with macOS. "
    "Give clear, direct answers. Tailor commands, paths, shortcuts, and setup instructions "
    "to macOS and its default zsh shell unless the user asks for another platform. "
    "Use fenced Markdown code blocks with a language label for code and commands. "
    "Be honest about uncertainty and never claim to have run commands or accessed files "
    "unless you actually have."
)

CONCISE_SYSTEM_PROMPT = (
    "Answer directly and briefly. Prefer the minimum explanation needed to be useful; "
    "do not reveal chain-of-thought."
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_export_path(session_name: str | None, extension: str) -> Path:
    name = session_name or datetime.now().strftime("chat-%Y%m%d-%H%M%S")
    directory = app_home() / "exports"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.{extension}"


class ChatApp:
    def __init__(
        self,
        settings: Settings,
        start_result: StartResult,
        *,
        session_name: str | None = None,
        plain: bool = False,
    ):
        self.settings = settings
        self.start_result = start_result
        self.manager = ServerManager(settings)
        self.client = ChatClient(settings)
        self.ui = TerminalUI(settings, plain=plain)
        self.web = WebSearcher(settings)
        self.transcript: list[dict[str, Any]] = []
        self.session_name = session_name
        self.last_result: GenerationResult | None = None
        self.should_exit = False
        self.voice_ready = False
        self.ui.command_previewer = self._preview_command

        self._preload_voice()

        if session_name:
            try:
                self._load(session_name, announce=False)
            except MlxChatError as exc:
                if "No saved session" not in str(exc):
                    raise

    def _preload_voice(self) -> None:
        """Warm Whisper alongside the chat model without making text chat unavailable."""
        try:
            with self.ui.startup_status("Preparing voice recognition…"):
                preload_transcription_model()
            self.voice_ready = True
        except (MlxChatError, OSError, ValueError) as exc:
            self.ui.warning(f"Voice is unavailable: {exc}")

    def api_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        messages.append(
            {"role": "system", "content": self.settings.system_prompt or DEFAULT_SYSTEM_PROMPT}
        )
        if self.settings.concise:
            messages.append({"role": "system", "content": CONCISE_SYSTEM_PROMPT})
        for item in self.transcript:
            role = item.get("role")
            content = item.get("api_content", item.get("content"))
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        return messages

    def run(self) -> int:
        self.ui.enable_conversation()
        try:
            return self._run()
        finally:
            self.ui.set_context(self.session_name, len(self.transcript))
            self.ui.close_conversation()

    def _run(self) -> int:
        self.ui.banner(
            reused=self.start_result.reused,
            managed=self.start_result.managed,
            voice_ready=self.voice_ready,
        )
        if self.session_name:
            self.ui.info(f"Session {self.session_name!r} is active and will auto-save.")

        while not self.should_exit:
            self.ui.set_context(self.session_name, len(self.transcript))
            try:
                value = self.ui.prompt()
            except EOFError:
                self._autosave()
                self.ui.info("Leaving the server running. Use `mlx-chat stop` when finished.")
                break
            except KeyboardInterrupt:
                self.ui.warning("Input cleared. Use /exit to leave or /stop to shut down.")
                continue

            if not value.strip():
                continue
            if value.startswith("//"):
                value = value[1:]
            elif value.startswith("/"):
                self._handle_command(value)
                continue
            self._generate(value)
        return 0

    def _generate(
        self,
        prompt: str | None = None,
        *,
        api_content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if prompt is not None:
            item: dict[str, Any] = {"role": "user", "content": prompt, "created_at": utc_now()}
            if api_content is not None:
                item["api_content"] = api_content
            if metadata:
                item.update(metadata)
            self.transcript.append(item)
        if not self.transcript or self.transcript[-1].get("role") != "user":
            self.ui.warning("There is no user prompt to generate from.")
            return False

        try:
            result = self.ui.render_generation(
                lambda callback: self.client.collect(
                    self.api_messages(),
                    think=self.settings.think and not self.settings.concise,
                    on_event=callback,
                ),
                think=self.settings.think and not self.settings.concise,
            )
        except KeyboardInterrupt:
            if prompt is not None and self.transcript[-1].get("role") == "user":
                self.transcript.pop()
            self.ui.warning("Generation cancelled; the server is still running.")
            return False
        except MlxChatError as exc:
            if prompt is not None and self.transcript[-1].get("role") == "user":
                self.transcript.pop()
            self.ui.error(str(exc))
            return False

        self.last_result = result
        self.transcript.append(
            {
                "role": "assistant",
                "content": result.answer,
                "reasoning": result.reasoning,
                "usage": result.usage,
                "metrics": {
                    "elapsed": result.elapsed,
                    "first_token_seconds": result.first_token_seconds,
                    "tokens_per_second": result.tokens_per_second,
                },
                "finish_reason": result.finish_reason,
                "created_at": utc_now(),
            }
        )
        self._autosave()
        return True

    def _preview_command(self, command):
        if command in READ_ONLY_COMMANDS:
            self._dispatch_command(command)
        else:
            self.ui.preview_command(command)

    def _handle_command(self, value: str) -> None:
        command = value.split()[0].lower() if value.split() else ""
        if (
            command in READ_ONLY_COMMANDS
            and getattr(self.ui, "last_preview_command", None) == command
        ):
            self.ui.last_preview_command = None
            return
        popup = getattr(self.ui, "command_popup", None)
        if popup is not None and command not in {
            "/voice",
            "/web",
            "/retry",
            "/stop",
            "/shutdown",
            "/exit",
            "/quit",
            "/q",
        }:
            with popup(command):
                self._dispatch_command(value)
        else:
            self._dispatch_command(value)

    def _dispatch_command(self, value: str) -> None:
        try:
            parts = shlex.split(value)
        except ValueError as exc:
            self.ui.error(f"Could not parse command: {exc}")
            return
        if not parts:
            return
        command = parts[0].lower()
        args = parts[1:]
        aliases = {
            "/quit": "/exit",
            "/q": "/exit",
            "/temp": "/temperature",
            "/max": "/max-tokens",
            "/top_p": "/top-p",
            "/top_k": "/top-k",
            "/clear-context": "/reset",
            "/shutdown": "/stop",
        }
        command = aliases.get(command, command)
        try:
            if command == "/help":
                self.ui.help()
            elif command == "/web":
                self._cmd_web(args)
            elif command == "/think":
                self._cmd_think(args)
            elif command == "/concise":
                self._cmd_concise(args)
            elif command == "/voice":
                self._cmd_voice(args)
            elif command == "/speak":
                self._cmd_speak(args)
            elif command == "/new":
                self._cmd_new(args)
            elif command == "/reset":
                self.transcript.clear()
                self.last_result = None
                self._autosave()
                self.ui.success("Conversation context cleared.")
            elif command == "/clear":
                self._cmd_clear(args)
            elif command == "/system":
                self._cmd_system(args)
            elif command == "/retry":
                self._cmd_retry()
            elif command == "/undo":
                self._cmd_undo()
            elif command == "/settings":
                self._cmd_settings()
            elif command == "/temperature":
                self._set_number("temperature", args, float, 0, 5)
            elif command == "/top-p":
                self._set_number("top_p", args, float, 0, 1)
            elif command == "/top-k":
                self._set_number("top_k", args, int, 0, None)
            elif command == "/max-tokens":
                self._set_number("max_tokens", args, int, 1, None)
            elif command == "/stats":
                self._cmd_stats()
            elif command == "/status":
                self._cmd_status()
            elif command == "/model":
                self._cmd_model(args)
            elif command == "/save":
                self._cmd_save(args)
            elif command == "/load":
                self._cmd_load(args)
            elif command == "/sessions":
                self._cmd_sessions()
            elif command == "/history":
                self._cmd_history(args)
            elif command == "/copy":
                self._cmd_copy(args)
            elif command == "/export":
                self._cmd_export(args)
            elif command == "/logs":
                self._cmd_logs(args)
            elif command == "/restart":
                self._restart(self.settings.model)
            elif command == "/stop":
                self._cmd_stop()
            elif command == "/exit":
                self._autosave()
                self.should_exit = True
                self.ui.info("Leaving the server running. Use `mlx-chat stop` when finished.")
            else:
                names = SLASH_COMMANDS + list(aliases)
                match = difflib.get_close_matches(command, names, n=1)
                hint = f" Did you mean {match[0]}?" if match else " Use /help to list commands."
                self.ui.error(f"Unknown command {command!r}.{hint}")
        except (MlxChatError, OSError, ValueError) as exc:
            self.ui.error(str(exc))

    def _cmd_think(self, args: list[str]) -> None:
        if not args:
            self.settings.think = not self.settings.think
        else:
            mode = args[0].lower()
            if mode in {"on", "true", "1", "yes"}:
                self.settings.think = True
            elif mode in {"off", "false", "0", "no"}:
                self.settings.think = False
            elif mode == "toggle":
                self.settings.think = not self.settings.think
            else:
                raise MlxChatError("Usage: /think [on|off|toggle]")
        mode = (
            "ON — reasoning and answer will be shown separately" if self.settings.think else "OFF"
        )
        self._autosave()
        self.ui.success(f"Thinking is {mode}.")

    def _cmd_concise(self, args: list[str]) -> None:
        if len(args) > 1:
            raise MlxChatError("Usage: /concise [on|off|toggle]")
        if not args or args[0].lower() == "toggle":
            self.settings.concise = not self.settings.concise
        elif args[0].lower() in {"on", "true", "1", "yes"}:
            self.settings.concise = True
        elif args[0].lower() in {"off", "false", "0", "no"}:
            self.settings.concise = False
        else:
            raise MlxChatError("Usage: /concise [on|off|toggle]")
        self._autosave()
        state = (
            "ON — direct mode is active and thinking is skipped" if self.settings.concise else "OFF"
        )
        self.ui.success(f"Concise mode is {state}.")

    def _cmd_voice(self, args: list[str]) -> None:
        if len(args) > 1 or (args and not args[0].isdigit()):
            raise MlxChatError("Usage: /voice [SECONDS]")
        seconds = 30 if not args else int(args[0])
        with self.ui.audio_status(f"/voice Listening… F3 to finish · up to {seconds}s"):
            audio = self.ui.run_audio(
                lambda stop_event: record_microphone(seconds=seconds, stop_event=stop_event)
            )
        with self.ui.audio_status("/voice Transcribing…"):
            prompt = self.ui.run_native(
                lambda: transcribe_audio(audio, model=DEFAULT_TRANSCRIPTION_MODEL)
            )
        voice_message = getattr(self.ui, "voice_message", self.ui.user_message)
        voice_message(prompt)
        self._generate(
            prompt,
            metadata={"input": "voice", "transcription_model": DEFAULT_TRANSCRIPTION_MODEL},
        )

    def _cmd_speak(self, args: list[str]) -> None:
        if len(args) > 1 or (args and args[0] not in {"answer", "reasoning", "all"}):
            raise MlxChatError("Usage: /speak [answer|reasoning|all]")
        if self.last_result is None:
            raise MlxChatError("There is no completed response to speak.")
        target = args[0] if args else "answer"
        if target == "reasoning":
            text = self.last_result.reasoning
        elif target == "all":
            text = "\n\n".join(
                part for part in (self.last_result.reasoning, self.last_result.answer) if part
            )
        else:
            text = self.last_result.answer
        try:
            with self.ui.audio_status("Speaking… F3 to stop"):
                stopped = self.ui.run_audio(
                    lambda stop_event: speak(text, stop_event=stop_event), stop_key="f3"
                )
        except KeyboardInterrupt:
            self.ui.info("Speech stopped.")
            return
        if stopped:
            self.ui.info("Speech stopped.")
        else:
            self.ui.info("Speech finished.")

    def _cmd_web(self, args: list[str]) -> None:
        if not args:
            raise MlxChatError("Usage: /web QUERY")
        query = " ".join(args)
        with self.ui.console.status(Text("Searching the web…", style="cyan")):
            result = self.web.search(query)
        self.ui.info(f"Found {len(result.sources)} web sources. Generating a cited answer…")
        self._generate(
            query,
            api_content=result.model_prompt(),
            metadata={
                "web_query": result.query,
                "web_sources": [source.as_dict() for source in result.sources],
            },
        )

    def _cmd_clear(self, args: list[str]) -> None:
        if args:
            raise MlxChatError("Usage: /clear")
        self.ui.clear()

    def _cmd_new(self, args: list[str]) -> None:
        if len(args) > 1:
            raise MlxChatError("Usage: /new [NAME]")
        new_name = normalize_session_name(args[0]) if args else None
        self._autosave()
        old_transcript = self.transcript
        old_result = self.last_result
        old_name = self.session_name
        self.transcript = []
        self.last_result = None
        self.session_name = new_name
        try:
            if new_name:
                self._save(new_name)
        except Exception:
            self.transcript = old_transcript
            self.last_result = old_result
            self.session_name = old_name
            raise
        label = "session " + repr(self.session_name) if self.session_name else "a new conversation"
        self.ui.success(f"Started {label}.")

    def _cmd_system(self, args: list[str]) -> None:
        if not args:
            self.ui.console.print(
                Panel(
                    Text(self.settings.system_prompt or DEFAULT_SYSTEM_PROMPT),
                    title="System prompt"
                    if self.settings.system_prompt
                    else "System prompt (default)",
                    border_style="bright_cyan",
                )
            )
            return
        value = " ".join(args)
        if value.lower() == "clear":
            value = ""
        elif value.startswith("@"):
            path = Path(value[1:]).expanduser()
            value = path.read_text(encoding="utf-8")
        self.settings.system_prompt = value
        self._autosave()
        self.ui.success(
            "System prompt updated." if value else "Default MacBook system prompt restored."
        )

    def _cmd_retry(self) -> None:
        removed_assistant = None
        previous_result = self.last_result
        if self.transcript and self.transcript[-1].get("role") == "assistant":
            removed_assistant = self.transcript.pop()
        if not self.transcript or self.transcript[-1].get("role") != "user":
            if removed_assistant is not None:
                self.transcript.append(removed_assistant)
            self.ui.warning("There is no completed turn to retry.")
            return
        if not self._generate():
            if removed_assistant is not None:
                self.transcript.append(removed_assistant)
            self.last_result = previous_result
            self._autosave()

    def _cmd_undo(self) -> None:
        if not self.transcript:
            self.ui.warning("Conversation is already empty.")
            return
        if self.transcript[-1].get("role") == "assistant":
            self.transcript.pop()
        if self.transcript and self.transcript[-1].get("role") == "user":
            self.transcript.pop()
        self.last_result = None
        self._autosave()
        self.ui.success("Removed the last turn.")

    def _cmd_settings(self) -> None:
        table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        table.add_column(style="cyan")
        table.add_column()
        values = [
            ("model", self.settings.model),
            ("endpoint", self.settings.base_url),
            ("thinking", "on" if self.settings.think else "off"),
            ("concise", "on" if self.settings.concise else "off"),
            ("temperature", f"{self.settings.temperature:g}"),
            ("top_p", f"{self.settings.top_p:g}"),
            ("top_k", str(self.settings.top_k)),
            ("min_p", f"{self.settings.min_p:g}"),
            ("max_tokens", str(self.settings.max_tokens)),
            ("repetition_penalty", f"{self.settings.repetition_penalty:g}"),
            ("system_prompt", self.settings.system_prompt or DEFAULT_SYSTEM_PROMPT),
            ("web_results", str(self.settings.web_results)),
            ("web_region", self.settings.web_region),
            ("web_safesearch", self.settings.web_safesearch),
        ]
        for key, item in values:
            table.add_row(Text(key), Text(str(item)))
        self.ui.console.print(Panel(table, title="Effective settings", border_style="bright_cyan"))

    def _set_number(
        self,
        key: str,
        args: list[str],
        convert: Any,
        minimum: float | None,
        maximum: float | None,
    ) -> None:
        display = key.replace("_", "-")
        if not args:
            self.ui.info(f"{display} = {getattr(self.settings, key)}")
            return
        try:
            value = convert(args[0])
        except (TypeError, ValueError) as exc:
            raise MlxChatError(f"{display} needs a number.") from exc
        if minimum is not None and value < minimum:
            raise MlxChatError(f"{display} must be at least {minimum}.")
        if maximum is not None and value > maximum:
            raise MlxChatError(f"{display} must be at most {maximum}.")
        setattr(self.settings, key, value)
        self.settings.validate()
        self._autosave()
        self.ui.success(f"{display} = {value}")

    def _cmd_stats(self) -> None:
        if self.last_result is None:
            self.ui.warning("No generation statistics yet.")
            return
        result = self.last_result
        table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        table.add_column(style="cyan")
        table.add_column(justify="right")
        table.add_row(Text("Elapsed"), Text(f"{result.elapsed:.3f} s"))
        if result.first_token_seconds is not None:
            table.add_row(Text("Time to first token"), Text(f"{result.first_token_seconds:.3f} s"))
        if result.tokens_per_second is not None:
            table.add_row(Text("Throughput"), Text(f"{result.tokens_per_second:.2f} token/s"))
        for key, value in result.usage.items():
            table.add_row(Text(key.replace("_", " ").title()), Text(str(value)))
        table.add_row(Text("Finish reason"), Text(result.finish_reason or "unknown"))
        self.ui.console.print(Panel(table, title="Last response", border_style="bright_cyan"))

    def _cmd_status(self) -> None:
        record = self.manager.matching_record()
        table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        table.add_column(style="cyan")
        table.add_column()
        table.add_row("Server", "running")
        table.add_row("Ownership", "managed" if record else "external")
        if record:
            table.add_row(Text("PID"), Text(str(record.pid)))
            table.add_row(Text("Started"), Text(record.started_at))
            table.add_row(Text("Log"), Text(record.log_path))
        table.add_row(Text("Model"), Text(self.settings.model))
        table.add_row(Text("Endpoint"), Text(self.settings.base_url))
        table.add_row(Text("Session"), Text(self.session_name or "unsaved"))
        table.add_row(Text("Messages"), Text(str(len(self.transcript))))
        self.ui.console.print(Panel(table, title="mlx-chat status", border_style="bright_cyan"))

    def _cmd_model(self, args: list[str]) -> None:
        if not args:
            presets = ", ".join(f"{name} ({model})" for name, model in MODEL_PRESETS.items())
            self.ui.info(f"Current model: {self.settings.model}. Presets: {presets}")
            return
        model = resolve_model_preset(args[0])
        if model == self.settings.model:
            self.ui.info("That model is already active.")
            return
        self._restart(model)

    @staticmethod
    def _stop_new_result(
        manager: ServerManager,
        result: StartResult | None,
        *,
        grace: float = 2.0,
    ) -> None:
        if (
            result is None
            or not result.managed
            or result.reused
            or result.pid is None
            or not result.instance_id
        ):
            return
        record = manager.matching_record()
        if record and record.pid == result.pid and record.instance_id == result.instance_id:
            stop_record(record, grace=grace)

    def _restart(self, model: str) -> None:
        self.ui.run_task(lambda: self._restart_server(model))

    def _restart_server(self, model: str) -> None:
        candidate = self.settings.merged(model=model)
        record = self.manager.matching_record()
        if not record and not self.start_result.managed:
            raise MlxChatError("This chat is attached to an external server; it cannot restart it.")
        old_model = self.settings.model
        old_server_args = list(record.server_args) if record else list(self.settings.server_args)
        self._autosave()
        new_result: StartResult | None = None
        new_manager: ServerManager | None = None
        try:
            if record:
                with self.ui.console.status(Text("Stopping the managed server…", style="cyan")):
                    stopped = stop_record(record)
                if not stopped.stopped and stopped.detail != "already stopped":
                    raise MlxChatError(f"Server restart failed: {stopped.detail}")

            self.settings.model = candidate.model
            new_manager = ServerManager(self.settings)
            new_client = ChatClient(self.settings)
            with self.ui.console.status(
                Text(
                    f"Starting {short_model(model)} (download/loading can take a while)…",
                    style="cyan",
                )
            ):
                new_result = new_manager.start(
                    ready_check=lambda _result: new_client.warmup(
                        timeout=self.settings.startup_timeout
                    )
                )
        except (Exception, KeyboardInterrupt) as switch_error:
            if new_manager is not None:
                self._stop_new_result(new_manager, new_result)

            # Restore the previously working model so a typo or download error
            # does not strand the live conversation without a server.
            self.settings.model = old_model
            self.settings.server_args = old_server_args
            self.manager = ServerManager(self.settings)
            self.client = ChatClient(self.settings)
            restored: StartResult | None = None
            try:
                with self.ui.console.status(
                    Text(f"Restoring {short_model(old_model)}…", style="yellow")
                ):
                    restored = self.manager.start(
                        ready_check=lambda _result: self.client.warmup(
                            timeout=self.settings.startup_timeout
                        )
                    )
                self.start_result = restored
            except (Exception, KeyboardInterrupt) as restore_error:
                self._stop_new_result(self.manager, restored)
                raise MlxChatError(
                    f"Could not start {model!r} ({switch_error}), and restoring "
                    f"{old_model!r} also failed ({restore_error}). Use /restart to retry."
                ) from switch_error
            if isinstance(switch_error, KeyboardInterrupt):
                raise
            raise MlxChatError(
                f"Could not start {model!r}: {switch_error}. Restored {old_model!r}."
            ) from switch_error

        assert new_manager is not None
        self.manager = new_manager
        self.client = new_client
        self.start_result = new_result
        self._autosave()
        self.ui.success(f"Ready with {model}.")

    def _session_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.session_name,
            "model": self.settings.model,
            "system_prompt": self.settings.system_prompt,
            "settings": {
                "think": self.settings.think,
                "concise": self.settings.concise,
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "top_k": self.settings.top_k,
                "min_p": self.settings.min_p,
                "max_tokens": self.settings.max_tokens,
            },
            "messages": self.transcript,
            "updated_at": utc_now(),
        }

    def _save(self, name: str) -> Path:
        previous = self.session_name
        self.session_name = name
        payload = self._session_payload()
        payload["name"] = name
        try:
            return save_session(name, payload)
        except Exception:
            self.session_name = previous
            raise

    def _autosave(self) -> None:
        if self.session_name:
            self._save(self.session_name)

    def _cmd_save(self, args: list[str]) -> None:
        name = args[0] if args else self.session_name
        if not name:
            raise MlxChatError("Usage: /save NAME")
        path = self._save(name)
        self.ui.success(f"Saved session {name!r} to {path}.")

    def _load(self, name: str, *, announce: bool = True) -> None:
        payload = load_session(name)
        messages = payload.get("messages", [])
        candidate_messages = [item for item in messages if isinstance(item, dict)]
        overrides: dict[str, Any] = {
            "system_prompt": payload.get("system_prompt") or "",
        }
        saved = payload.get("settings", {})
        if isinstance(saved, dict):
            for key in ("think", "concise", "temperature", "top_p", "top_k", "min_p", "max_tokens"):
                if key in saved:
                    overrides[key] = saved[key]
        candidate = self.settings.merged(**overrides)

        # Commit only after all saved values have passed validation.
        self.transcript = candidate_messages
        for key in (
            "system_prompt",
            "think",
            "concise",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
        ):
            setattr(self.settings, key, getattr(candidate, key))
        self.session_name = name
        self.last_result = None
        if announce:
            saved_model = payload.get("model")
            self.ui.success(f"Loaded session {name!r} ({len(self.transcript)} messages).")
            if saved_model and saved_model != self.settings.model:
                self.ui.warning(
                    f"It was created with {saved_model}; continuing with {self.settings.model}."
                )

    def _cmd_load(self, args: list[str]) -> None:
        if not args:
            raise MlxChatError("Usage: /load NAME")
        self._autosave()
        self._load(args[0])

    def _cmd_sessions(self) -> None:
        sessions = list_sessions()
        if not sessions:
            self.ui.info("No saved sessions yet. Use /save NAME.")
            return
        table = Table("Name", "Modified", "Path", box=box.SIMPLE, pad_edge=False)
        for name, path in sessions:
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            table.add_row(Text(name), Text(modified), Text(str(path)))
        self.ui.console.print(Panel(table, title="Saved sessions", border_style="bright_cyan"))

    def _cmd_history(self, args: list[str]) -> None:
        count = int(args[0]) if args else 6
        if count < 1:
            raise MlxChatError("History count must be positive.")
        if not self.transcript:
            self.ui.info("Conversation is empty.")
            return
        for item in self.transcript[-count:]:
            role = str(item.get("role", "message")).title()
            content = sanitize_terminal_text(str(item.get("content", "")))
            style = "green" if role == "User" else "bright_cyan"
            self.ui.console.print(Panel(Text(content), title=Text(role), border_style=style))

    def _last_assistant(self) -> dict[str, Any] | None:
        return next(
            (item for item in reversed(self.transcript) if item.get("role") == "assistant"),
            None,
        )

    def _cmd_copy(self, args: list[str]) -> None:
        item = self._last_assistant()
        if not item:
            raise MlxChatError("There is no assistant response to copy.")
        mode = args[0].lower() if args else "answer"
        if mode == "answer":
            value = str(item.get("content", ""))
        elif mode == "reasoning":
            value = str(item.get("reasoning", ""))
        elif mode == "all":
            value = (
                f"Reasoning\n\n{item.get('reasoning', '')}\n\nAnswer\n\n{item.get('content', '')}"
            )
        else:
            raise MlxChatError("Usage: /copy [answer|reasoning|all]")
        executable = shutil.which("pbcopy")
        if not executable:
            raise MlxChatError("pbcopy is not available on this system.")
        subprocess.run([executable], input=value, text=True, check=True)
        self.ui.success(f"Copied {mode} to the clipboard.")

    def _cmd_export(self, args: list[str]) -> None:
        format_name = args[0].lower() if args else "md"
        if format_name not in {"md", "json"}:
            raise MlxChatError("Usage: /export [md|json] [PATH]")
        path = (
            Path(args[1]).expanduser()
            if len(args) > 1
            else default_export_path(self.session_name, format_name)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if format_name == "json":
            path.write_text(
                json.dumps(self._session_payload(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            lines = [f"# {self.session_name or 'MLX Chat'}", ""]
            if self.settings.system_prompt:
                lines.extend(["## System", "", self.settings.system_prompt, ""])
            for item in self.transcript:
                role = str(item.get("role", "message")).title()
                if item.get("reasoning"):
                    lines.extend(["## Reasoning", "", str(item["reasoning"]), ""])
                lines.extend([f"## {role}", "", str(item.get("content", "")), ""])
            path.write_text("\n".join(lines), encoding="utf-8")
        self.ui.success(f"Exported conversation to {path}.")

    def _cmd_logs(self, args: list[str]) -> None:
        count = int(args[0]) if args else 60
        if count < 1 or count > 1000:
            raise MlxChatError("Log line count must be between 1 and 1000.")
        record = self.manager.matching_record()
        path = Path(record.log_path) if record else self.start_result.log_path
        if not path:
            self.ui.info("No server logs were found.")
            return
        value = sanitize_terminal_text(tail_log(path, lines=count))
        self.ui.console.print(
            Panel(Text(value or "(empty)"), title=Text(str(path)), border_style="bright_cyan")
        )

    def _cmd_stop(self) -> None:
        record = self.manager.matching_record()
        if not record:
            raise MlxChatError(
                "This is an external server, so mlx-chat will not stop it without ownership proof."
            )
        self._autosave()
        with self.ui.console.status(Text("Stopping MLX-LM and releasing memory…", style="cyan")):
            result = stop_record(record)
        if not result.stopped and result.detail != "already stopped":
            raise MlxChatError(result.detail)
        self.ui.server_stopped = True
        self.ui.success("Server stopped. Model memory has been released.")
        self.should_exit = True
