"""Interactive conversation controller and slash commands."""

from __future__ import annotations

import difflib
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import MODEL_PRESETS, resolve_model_preset
from .agent import (
    EXECUTE_REQUEST,
    PLAN_REQUEST,
    PROJECT_MEMORY_FILES,
    REVISE_REQUEST,
    AgentState,
    agent_system_message,
    load_project_memory,
    run_agent,
)
from .attachments import Attachment, attachment_prompt, merge_attachments, read_attachments
from .client import ChatClient, GenerationResult
from .commands import SLASH_COMMANDS
from .config import (
    Settings,
    app_home,
    list_sessions,
    load_session,
    normalize_session_name,
    save_session,
)
from .context import (
    DEFAULT_SYSTEM_PROMPT,
    TEMPLATE_RESERVE,
    select_context,
    summary_message,
    system_messages,
)
from .errors import MlxChatError
from .profiles import PROFILE_FIELDS, apply_profile, delete_profile, list_profiles, save_profile
from .server import (
    ServerManager,
    StartResult,
    health,
    stop_record,
    tail_log,
)
from .sessions import SESSION_SETTINGS, latest_result, restore_session
from .terminal import copy_text
from .tools import APPROVAL_MODES, ToolCall, Workspace
from .tts import speak
from .ui import TerminalUI, sanitize_terminal_text, short_model
from .voice import (
    DEFAULT_TRANSCRIPTION_MODEL,
    preload_transcription_model,
    record_microphone,
    transcribe_audio,
)
from .web import WebSearcher


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def display_path(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


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
        attachments: list[Attachment] | None = None,
        generation_overrides: dict[str, Any] | None = None,
        workspace: Path | None = None,
    ):
        self.settings = settings
        self.start_result = start_result
        self.manager = ServerManager(settings)
        self.client = ChatClient(settings)
        self.ui = TerminalUI(settings, plain=plain)
        self.web = WebSearcher(settings)
        self.transcript: list[dict[str, Any]] = []
        self.pending_attachments: list[Attachment] = []
        self.session_name = session_name
        self.last_result: GenerationResult | None = None
        self.should_exit = False
        self.voice_ready = False
        self.workspace_root = Path(workspace) if workspace is not None else Path.cwd()
        self.agent: AgentState | None = None
        if settings.agent:
            self._enable_agent(announce=False)

        if session_name:
            try:
                self._load(session_name, announce=False)
            except MlxChatError as exc:
                if "No saved session" not in str(exc):
                    raise

        if generation_overrides:
            candidate = self.settings.merged(**generation_overrides)
            for key in generation_overrides:
                setattr(self.settings, key, getattr(candidate, key))
        self.pending_attachments = attachments or []
        if self.pending_attachments:
            self._check_attachment_budget(self.pending_attachments, self.settings)
        if settings.preload_voice:
            self._preload_voice()

    def _preload_voice(self) -> None:
        """Warm Whisper alongside the chat model without making text chat unavailable."""
        try:
            with self.ui.startup_status("Preparing voice recognition…"):
                preload_transcription_model()
            self.voice_ready = True
        except (MlxChatError, OSError, ValueError) as exc:
            self.ui.warning(f"Voice is unavailable: {exc}")

    def api_messages(self) -> list[dict[str, Any]]:
        agent = getattr(self, "agent", None)
        messages = system_messages(self.settings)
        if agent is not None and agent.tools_supported:
            messages.append(agent_system_message(agent.workspace, agent.memory))
        items = self.transcript
        summary = self._latest_summary()
        if summary is not None:
            # Messages before a summary are kept for display and export, not sent.
            messages.append(summary_message(self.transcript[summary]["content"]))
            items = self.transcript[summary + 1 :]
        for item in items:
            role = item.get("role")
            content = item.get("api_content", item.get("content"))
            if role in {"user", "assistant"} and isinstance(content, str):
                message: dict[str, Any] = {"role": role, "content": content}
                if role == "assistant" and item.get("tool_calls"):
                    message["tool_calls"] = item["tool_calls"]
                messages.append(message)
            elif role == "tool" and isinstance(content, str):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("tool_call_id", ""),
                        "name": item.get("name", ""),
                        "content": content,
                    }
                )
        if agent is None:
            selection = select_context(messages, self.settings)
        else:
            selection = self._select_agent_context(messages)
        self.context_omitted_messages = selection.omitted_messages
        return selection.messages

    def _select_agent_context(self, messages: list[dict[str, Any]]):
        """Fit an agent turn, shortening its oldest tool outputs only when it must.

        Earlier messages are left byte-identical whenever possible so the server's
        prompt cache can reuse their KV state on the next step.
        """
        try:
            return select_context(messages, self.settings, bytes_per_token=3)
        except MlxChatError as exc:
            if "latest turn" not in str(exc):
                raise
        compacted = [dict(message) for message in messages]
        tools = [index for index, message in enumerate(compacted) if message["role"] == "tool"]
        for index in tools[:-1]:
            content = compacted[index]["content"]
            if len(content) > 600:
                compacted[index]["content"] = (
                    content[:400] + "\n… [older tool output shortened to fit the context]"
                )
            try:
                return select_context(compacted, self.settings, bytes_per_token=3)
            except MlxChatError as exc:
                if "latest turn" not in str(exc):
                    raise
        raise MlxChatError(
            "This agent turn no longer fits the context budget. Start a /new conversation, "
            "or raise --context-tokens if the model supports a longer context."
        )

    def _latest_summary(self) -> int | None:
        for index in range(len(self.transcript) - 1, -1, -1):
            if self.transcript[index].get("role") == "summary":
                return index
        return None

    def _enable_agent(self, *, announce: bool = True) -> None:
        workspace = Workspace(
            self.workspace_root,
            output_chars=self.settings.tool_output_chars,
            web=self.settings,
        )
        previous = getattr(self, "agent", None)
        self.agent = AgentState(
            workspace,
            approval=self.settings.approval,
            max_steps=self.settings.agent_max_steps,
            always_allowed=set(previous.always_allowed) if previous else set(),
            memory=load_project_memory(workspace.root),
        )
        self._sync_agent_status()
        if announce:
            self.ui.success(
                f"Agent is on in {display_path(workspace.root)} · approval: {self.agent.approval}."
            )

    def _sync_agent_status(self) -> None:
        agent = getattr(self, "agent", None)
        if agent is None:
            self.ui.agent_status = None
            return
        self.ui.agent_status = (
            f"{agent.workspace.root.name or agent.workspace.root} · {agent.approval}"
        )

    def _conversation_size(self) -> int:
        """Messages as the header counts them: tool steps do not add turns."""
        return 2 * sum(1 for item in self.transcript if item.get("role") == "user")

    def run(self) -> int:
        self.ui.enable_conversation()
        try:
            return self._run()
        finally:
            self.ui.set_context(self.session_name, self._conversation_size())
            self.ui.close_conversation()

    def _run(self) -> int:
        self.ui.set_context(self.session_name, self._conversation_size())
        self.ui.banner(
            reused=self.start_result.reused,
            managed=self.start_result.managed,
            voice_ready=self.voice_ready,
        )
        if self.transcript:
            self.ui.restore_transcript(self.transcript, replace=False)
        if self.pending_attachments:
            self._show_attachments()
        if self.session_name:
            self.ui.info(f"Session {self.session_name!r} is active and will auto-save.")

        while not self.should_exit:
            self.ui.set_context(self.session_name, self._conversation_size())
            self.ui.context_share = self._context_load() if self.transcript else 0.0
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
            self._maybe_compact()
            item: dict[str, Any] = {"role": "user", "content": prompt, "created_at": utc_now()}
            if self.pending_attachments:
                item["api_content"] = attachment_prompt(
                    api_content or prompt, self.pending_attachments
                )
                item["attachments"] = [entry.path for entry in self.pending_attachments]
            elif api_content is not None:
                item["api_content"] = api_content
            if metadata:
                item.update(metadata)
            self.transcript.append(item)
        if not self.transcript or self.transcript[-1].get("role") != "user":
            self.ui.warning("There is no user prompt to generate from.")
            return False

        if getattr(self, "agent", None) is not None:
            return self._generate_agent(prompt)

        try:
            messages = self.api_messages()
            if self.context_omitted_messages:
                self.ui.warning(
                    f"Context budget: {self.context_omitted_messages} older messages omitted "
                    "from this request. The full conversation is retained."
                )
            result = self.ui.render_generation(
                lambda callback: self.client.collect(
                    messages,
                    think=self.settings.think and not self.settings.concise,
                    on_event=callback,
                    cancel_event=getattr(self.ui, "cancellation_event", None),
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

        if prompt is not None:
            self.pending_attachments = []
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

    def _generate_agent(self, prompt: str | None) -> bool:
        assert self.agent is not None
        think = self.settings.think and not self.settings.concise
        start = len(self.transcript)
        warned = False

        def build_messages():
            nonlocal warned
            messages = self.api_messages()
            if self.context_omitted_messages and not warned:
                warned = True
                self.ui.warning(
                    f"Context budget: {self.context_omitted_messages} older messages omitted "
                    "from this request. The full conversation is retained."
                )
            return messages

        self.ui.defer_metrics = True
        try:
            run = run_agent(self.agent, ChatAgentHost(self, think), self.transcript, build_messages)
        except KeyboardInterrupt:
            if prompt is not None and len(self.transcript) == start:
                self.transcript.pop()
            elif len(self.transcript) > start:
                self._autosave()
            self.ui.warning("Generation cancelled; the server is still running.")
            return False
        except MlxChatError as exc:
            if prompt is not None and len(self.transcript) == start:
                self.transcript.pop()
            self.ui.error(str(exc))
            return False
        finally:
            self.ui.defer_metrics = False

        if prompt is not None:
            self.pending_attachments = []
        self.last_result = run.result
        self.ui.agent_summary(run)
        self._autosave()
        return True

    def _cmd_agent(self, args: list[str]) -> None:
        if len(args) > 1:
            raise MlxChatError("Usage: /agent [on|off]")
        mode = args[0].lower() if args else ("off" if self.agent else "on")
        if mode in {"on", "true", "1", "yes"}:
            self._enable_agent()
        elif mode in {"off", "false", "0", "no"}:
            self.agent = None
            self._sync_agent_status()
            self.ui.success("Agent is off. Replies are plain chat without tools.")
        else:
            raise MlxChatError("Usage: /agent [on|off]")

    def _cmd_approve(self, args: list[str]) -> None:
        if len(args) > 1 or (args and args[0].lower() not in APPROVAL_MODES):
            raise MlxChatError("Usage: /approve [ask|edits|all]")
        if not args:
            current = self.agent.approval if self.agent else self.settings.approval
            allowed = ", ".join(sorted(self.agent.always_allowed)) if self.agent else ""
            self.ui.info(
                f"Approval: {current}. ask = confirm edits and commands · edits = auto-apply "
                "file edits, confirm commands · all = run everything without asking."
                + (f"\nAlways allowed this session: {allowed}" if allowed else "")
            )
            return
        self.settings.approval = args[0].lower()
        if self.agent is not None:
            self.agent.approval = self.settings.approval
            if self.settings.approval == "ask":
                self.agent.always_allowed.clear()
        self._sync_agent_status()
        self.ui.success(f"Approval set to {self.settings.approval}.")

    # Context compaction

    def _turn_starts(self, start: int) -> list[int]:
        return [
            index
            for index in range(start, len(self.transcript))
            if self.transcript[index].get("role") == "user"
        ]

    def _context_load(self) -> float:
        """The share of the request budget the next request would use."""
        agent = getattr(self, "agent", None)
        budget = self.settings.context_tokens - self.settings.max_tokens - TEMPLATE_RESERVE
        try:
            messages = self.api_messages()
        except MlxChatError:
            return 1.0
        if self.context_omitted_messages:
            return 1.0
        selection = select_context(messages, self.settings, bytes_per_token=3 if agent else 1)
        return selection.estimated_tokens / max(1, budget)

    def _maybe_compact(self) -> None:
        if not hasattr(self, "client") or getattr(self.settings, "compact_at", 0) <= 0:
            return
        summary = self._latest_summary()
        if len(self._turn_starts(0 if summary is None else summary + 1)) < 3:
            return
        if self._context_load() < self.settings.compact_at / 100:
            return
        try:
            self._compact(automatic=True)
        except (MlxChatError, OSError) as exc:
            self.ui.warning(f"Could not summarize older messages: {exc}")

    def _compact(self, *, automatic: bool = False, keep_turns: int = 2) -> None:
        summary = self._latest_summary()
        start = 0 if summary is None else summary + 1
        turns = self._turn_starts(start)
        if len(turns) <= keep_turns:
            raise MlxChatError("There is not enough earlier conversation to summarize yet.")
        cut = turns[-keep_turns] if keep_turns else len(self.transcript)
        previous = self.transcript[summary]["content"] if summary is not None else ""
        older = self.transcript[start:cut]
        request = compaction_request(previous, older, self.settings)
        label = "Summarizing older messages to free context…"
        result = self.ui.run_task(
            lambda: self.client.collect(request, think=False),
            label=label,
        )
        text = result.answer.strip()
        if not text:
            raise MlxChatError("The model returned an empty summary.")
        item = {
            "role": "summary",
            "content": text,
            "covers": len(older)
            + (self.transcript[summary].get("covers", 0) if summary is not None else 0),
            "created_at": utc_now(),
        }
        self.transcript.insert(cut, item)
        self._autosave()
        self.ui.summary_notice(item, automatic=automatic)

    def _cmd_compact(self, args: list[str]) -> None:
        if args:
            raise MlxChatError("Usage: /compact")
        self._compact(keep_turns=1)

    # Plans and project memory

    def _cmd_plan(self, args: list[str]) -> None:
        if not args:
            plan = self.agent.workspace.plan if self.agent else []
            if not plan:
                self.ui.info("No plan yet. Use /plan TASK to plan before making changes.")
            else:
                self.ui.plan_card(plan)
            return
        if self.agent is None:
            raise MlxChatError("Plans need agent mode. Turn it on with /agent on.")
        task = " ".join(args)
        self.agent.read_only = True
        try:
            self.ui.user_message(f"/plan {task}")
            done = self._generate(task, api_content=PLAN_REQUEST.format(task=task))
            while done:
                decision, feedback = self.ui.ask_plan()
                if decision == "revise":
                    self.ui.user_message(feedback)
                    done = self._generate(
                        feedback, api_content=REVISE_REQUEST.format(feedback=feedback)
                    )
                    continue
                if decision == "run":
                    self.agent.read_only = False
                    self.ui.user_message("Run the plan.")
                    self._generate("Run the plan.", api_content=EXECUTE_REQUEST)
                else:
                    self.ui.info("Plan kept. Say “go” or describe changes when you are ready.")
                break
        finally:
            self.agent.read_only = False

    def _cmd_memory(self, args: list[str]) -> None:
        root = self.agent.workspace.root if self.agent else self.workspace_root
        if args and args[0] == "add":
            note = " ".join(args[1:]).strip()
            if not note:
                raise MlxChatError("Usage: /memory add TEXT")
            path = root / PROJECT_MEMORY_FILES[0]
            existing = path.read_text(encoding="utf-8") if path.exists() else "# Agent notes\n"
            separator = "" if existing.endswith("\n") else "\n"
            path.write_text(f"{existing}{separator}- {note}\n", encoding="utf-8")
            args = ["reload"]
            self.ui.success(f"Added to {display_path(path)}.")
        if args and args[0] == "reload":
            if self.agent is not None:
                self.agent.memory = load_project_memory(root)
            names = ", ".join(name for name, _ in load_project_memory(root)) or "none"
            self.ui.success(f"Project memory reloaded: {names}.")
            return
        if args:
            raise MlxChatError("Usage: /memory [add TEXT|reload]")
        memory = load_project_memory(root)
        if not memory:
            self.ui.info(
                "No project memory yet. Create AGENTS.md in the project (or use /memory add "
                "TEXT) with your test command, conventions, and anything the agent should know."
            )
            return
        for name, text in memory:
            self.ui.console.print(
                Panel(
                    Text(sanitize_terminal_text(text)),
                    title=Text(f"{name} · loaded into every agent turn"),
                    border_style="bright_cyan",
                )
            )

    def _cmd_tools(self) -> None:
        workspace = self.agent.workspace if self.agent else Workspace(self.workspace_root)
        approval = self.agent.approval if self.agent else self.settings.approval
        allowed = self.agent.always_allowed if self.agent else set()
        table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
        table.add_column("Tool", no_wrap=True)
        table.add_column("Needs approval", no_wrap=True)
        table.add_column("What it does")
        for tool in workspace.tools.values():
            needs = workspace.needs_approval(tool.name, approval, allowed)
            table.add_row(Text(tool.name), Text("yes" if needs else "no"), Text(tool.description))
        state = "on" if self.agent else "off"
        self.ui.console.print(
            Panel(
                table,
                title=f"Agent tools · {state} · {display_path(workspace.root)}",
                border_style="bright_cyan",
            )
        )

    def _handle_command(self, value: str) -> None:
        command = value.split()[0].lower() if value.split() else ""
        popup = getattr(self.ui, "command_popup", None)
        if popup is not None and command not in {
            "/voice",
            "/web",
            "/plan",
            "/compact",
            "/retry",
            "/load",
            "/undo",
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
            elif command == "/agent":
                self._cmd_agent(args)
            elif command == "/approve":
                self._cmd_approve(args)
            elif command == "/plan":
                self._cmd_plan(args)
            elif command == "/memory":
                self._cmd_memory(args)
            elif command == "/compact":
                self._cmd_compact(args)
            elif command == "/tools":
                if args:
                    raise MlxChatError("Usage: /tools")
                self._cmd_tools()
            elif command == "/web":
                self._cmd_web(args)
            elif command == "/attach":
                self._cmd_attach(args)
            elif command == "/detach":
                if args:
                    raise MlxChatError("Usage: /detach")
                self.pending_attachments = []
                self.ui.success("Pending attachments cleared.")
            elif command == "/profile":
                self._cmd_profile(args)
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
                self.pending_attachments = []
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

    def _show_attachments(self) -> None:
        if not self.pending_attachments:
            self.ui.info("No pending attachments. Use /attach PATH.")
            return
        files = "\n".join(
            f"{item.path} ({item.size} bytes{' · truncated' if item.truncated else ''})"
            for item in self.pending_attachments
        )
        self.ui.info(f"Selected file excerpts for the next prompt:\n{files}")

    def _check_attachment_budget(self, files, settings) -> None:
        select_context(
            [*system_messages(settings), {"role": "user", "content": attachment_prompt("", files)}],
            settings,
        )

    def _cmd_attach(self, args: list[str]) -> None:
        if args:
            files = merge_attachments(self.pending_attachments, read_attachments(args))
            self._check_attachment_budget(files, self.settings)
            self.pending_attachments = files
        self._show_attachments()

    def _cmd_profile(self, args: list[str]) -> None:
        if not args:
            self.ui.info("Available profiles: " + ", ".join(list_profiles()))
        elif len(args) == 2 and args[0] == "save":
            path = save_profile(args[1], self.settings)
            self.ui.success(f"Saved profile {path.stem!r} to {path}.")
        elif len(args) == 2 and args[0] == "delete":
            delete_profile(args[1])
            self.ui.success(f"Deleted profile {args[1]!r}.")
        elif len(args) == 1:
            candidate = apply_profile(args[0], self.settings)
            self._check_attachment_budget(self.pending_attachments, candidate)
            for key in PROFILE_FIELDS:
                setattr(self.settings, key, getattr(candidate, key))
            self._autosave()
            self.ui.success(f"Applied profile {args[0]!r}.")
        else:
            raise MlxChatError("Usage: /profile [NAME|save NAME|delete NAME]")

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
        if new_name and new_name in {name for name, _path in list_sessions()}:
            raise MlxChatError(f"Session {new_name!r} already exists. Use /load or another name.")
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
        self.pending_attachments = []
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
            "System prompt updated." if value else "Default Mac system prompt restored."
        )

    def _cmd_retry(self) -> None:
        removed: list[dict[str, Any]] = []
        previous_result = self.last_result
        while self.transcript and self.transcript[-1].get("role") in {"assistant", "tool"}:
            removed.insert(0, self.transcript.pop())
        if not self.transcript or self.transcript[-1].get("role") != "user":
            self.transcript.extend(removed)
            self.ui.warning("There is no completed turn to retry.")
            return
        if not self._generate():
            # Drop any partial agent steps from the failed retry before restoring.
            while self.transcript and self.transcript[-1].get("role") != "user":
                self.transcript.pop()
            self.transcript.extend(removed)
            self.last_result = previous_result
            self._autosave()

    def _cmd_undo(self) -> None:
        if not self.transcript:
            self.ui.warning("Conversation is already empty.")
            return
        while self.transcript and self.transcript[-1].get("role") in {"assistant", "tool"}:
            self.transcript.pop()
        if self.transcript and self.transcript[-1].get("role") == "user":
            self.transcript.pop()
        self.last_result = latest_result(self.transcript)
        self._autosave()
        restore = getattr(self.ui, "restore_transcript", None)
        if restore is not None:
            restore(self.transcript)
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
            ("agent", f"on · {display_path(self.agent.workspace.root)}" if self.agent else "off"),
            ("approval", self.agent.approval if self.agent else self.settings.approval),
            ("temperature", f"{self.settings.temperature:g}"),
            ("top_p", f"{self.settings.top_p:g}"),
            ("top_k", str(self.settings.top_k)),
            ("min_p", f"{self.settings.min_p:g}"),
            ("max_tokens", str(self.settings.max_tokens)),
            ("context_tokens", str(self.settings.context_tokens)),
            ("connect_timeout", f"{self.settings.connect_timeout} s"),
            ("request_timeout", f"{self.settings.request_timeout} s"),
            ("preload_voice", "on" if self.settings.preload_voice else "off"),
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
        candidate = self.settings.merged(**{key: value})
        setattr(self.settings, key, getattr(candidate, key))
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
        table.add_row(
            "Server", "running" if health(self.settings.host, self.settings.port) else "unavailable"
        )
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
            from .hardware import memory_gib, recommended_preset

            best = recommended_preset(memory_gib())
            presets = ", ".join(
                f"{name} ({model}{' · recommended for this Mac' if name == best else ''})"
                for name, model in MODEL_PRESETS.items()
            )
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

            # Roll back a failed model switch before accepting another prompt.
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
        if getattr(self, "agent", None) is not None:
            # A new model may support tool calling even if the previous one did not.
            self.agent.tools_supported = True
        self._autosave()
        self.ui.success(f"Ready with {model}.")

    def _session_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.session_name,
            "model": self.settings.model,
            "system_prompt": self.settings.system_prompt,
            "settings": {key: getattr(self.settings, key) for key in SESSION_SETTINGS},
            "messages": self.transcript,
            "updated_at": utc_now(),
        }

    def _save(self, name: str) -> Path:
        name = normalize_session_name(name)
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
        name = normalize_session_name(name)
        payload = load_session(name)
        restored = restore_session(payload, self.settings)

        # Validate before replacing the current session.
        self.transcript = restored.messages
        self.pending_attachments = []
        for key in ("system_prompt", *SESSION_SETTINGS):
            setattr(self.settings, key, getattr(restored.settings, key))
        self.session_name = name
        self.last_result = restored.last_result
        if announce:
            saved_model = restored.model
            self.ui.set_context(self.session_name, self._conversation_size())
            self.ui.restore_transcript(self.transcript)
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
        copy_text(value)
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
                if item.get("role") == "summary":
                    lines.extend(["## Summary of earlier messages", "", str(item["content"]), ""])
                    continue
                if item.get("role") == "tool":
                    lines.extend(
                        [
                            f"### Tool: {item.get('name', '')} · {item.get('summary', '')}",
                            "",
                            "```",
                            str(item.get("diff") or item.get("content", "")).rstrip(),
                            "```",
                            "",
                        ]
                    )
                    continue
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
                "This server is not tracked by mlx-chat. "
                "Stop it through the process that started it."
            )
        self._autosave()
        with self.ui.console.status(Text("Stopping MLX-LM and releasing memory…", style="cyan")):
            result = stop_record(record)
        if not result.stopped and result.detail != "already stopped":
            raise MlxChatError(result.detail)
        self.ui.server_stopped = True
        self.ui.success("Server stopped. Model memory has been released.")
        self.should_exit = True


class ChatAgentHost:
    """Connect the agent loop to the chat view."""

    def __init__(self, app: ChatApp, think: bool):
        self.app = app
        self.think = think

    def generate(self, messages, tools):
        app = self.app
        return app.ui.render_generation(
            lambda callback: app.client.collect(
                messages,
                think=self.think,
                on_event=callback,
                cancel_event=getattr(app.ui, "cancellation_event", None),
                tools=tools,
            ),
            think=self.think,
        )

    def approve(self, call: ToolCall, preview):
        return self.app.ui.ask_approval(call.name, call.raw_arguments, preview)

    def execute(self, call: ToolCall, run):
        from .rendering import TOOL_LABELS, tool_description

        label = f"{TOOL_LABELS.get(call.name, call.name)} " + tool_description(
            {"name": call.name, "arguments": call.raw_arguments}
        )
        return self.app.ui.run_tool(label.strip() + "…", run)

    def tool_finished(self, item):
        self.app.ui.tool_result(item)

    def notice(self, message: str) -> None:
        self.app.ui.warning(message)


def compaction_request(previous: str, items: list[dict[str, Any]], settings: Settings):
    """A plain request asking the model to condense older turns."""
    lines = []
    if previous:
        lines.append(f"[Earlier summary]\n{previous}")
    for item in items:
        role = item.get("role")
        content = str(item.get("api_content") or item.get("content") or "")
        if role == "tool":
            content = f"{item.get('name')}: {item.get('summary', '')}\n{content[:300]}"
        elif role == "assistant" and item.get("tool_calls"):
            names = ", ".join(call["function"]["name"] for call in item["tool_calls"])
            content = f"{content}\n(called {names})".strip()
        if content:
            lines.append(f"[{role}]\n{content[:4000]}")
    budget = (settings.context_tokens - settings.max_tokens - TEMPLATE_RESERVE) * 3 - 2000
    text = "\n\n".join(lines)
    if len(text) > budget:
        text = "…\n" + text[-budget:]
    return [
        {
            "role": "system",
            "content": "You write compact, faithful summaries of work sessions.",
        },
        {
            "role": "user",
            "content": (
                "Summarize this conversation so it can replace the original messages. Keep "
                "the user's goals and preferences, decisions, files changed, commands run "
                "and their results, errors, and anything left to do. Use short bullet points, "
                "at most 300 words.\n\n" + text
            ),
        },
    ]
