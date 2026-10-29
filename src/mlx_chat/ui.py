"""Terminal input and chat controls."""

from __future__ import annotations

from contextlib import contextmanager

from prompt_toolkit import PromptSession
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.filters import Condition, to_filter
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style as PromptStyle
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.style import Style as RichStyle
from rich.table import Table
from rich.text import Text

from . import tts, voice
from .commands import SLASH_COMMANDS as SLASH_COMMANDS
from .config import Settings, history_path
from .errors import MlxChatError
from .rendering import (
    AnswerPanel,
    GenerationPresentation,
    ThinkingTrace,
    approval_card,
    plan_checklist,
    plan_prompt_card,
    tool_description,
)
from .sessions import result_from_message
from .startup import StartupStatus
from .terminal import copy_text, paste_text
from .terminal import sanitize_terminal_text as sanitize_terminal_text
from .terminal import short_model as short_model
from .theme import ACCENT, BORDER, BRAND_START, DANGER, MUTED, SUCCESS, SURFACE, WARNING, gradient

APPROVAL_PLACEHOLDER = "↵ allow once · a always · n decline · or type what to do instead"
PLAN_PLACEHOLDER = "↵ run the plan · type changes to revise it · n keep it for later"


class TerminalUI(GenerationPresentation):
    def __init__(self, settings: Settings, *, plain: bool = False):
        force_terminal = None if not plain else False
        self.console = Console(force_terminal=force_terminal, no_color=plain)
        self._interactive_prompt = self.console.is_terminal and not plain
        self.settings = settings
        self._session_name: str | None = None
        self._message_count = 0
        self._conversation = None
        self.server_stopped = False
        self.agent_status: str | None = None
        self._banner_state = {"reused": False, "managed": True, "voice_ready": False}

        bindings = KeyBindings()

        editable = Condition(lambda: not self.prompt_session.default_buffer.read_only())
        selected = editable & Condition(
            lambda: self.prompt_session.default_buffer.selection_state is not None
        )

        @bindings.add("c-a", filter=editable, eager=True)
        def select_all(event):
            buffer = event.current_buffer
            buffer.complete_state = None
            buffer.cursor_position = 0
            buffer.start_selection()
            buffer.cursor_position = len(buffer.text)

        def copy_input(event, *, cut=False):
            buffer = event.current_buffer
            selection = buffer.selection_state
            data = buffer.cut_selection() if cut else buffer.copy_selection()
            if not cut:
                buffer.selection_state = selection
            event.app.clipboard.set_data(data)
            try:
                copy_text(data.text)
            except MlxChatError:
                self.warning("Could not copy input to the macOS clipboard.")

        @bindings.add("c-c", filter=selected, eager=True)
        def copy_selected(event):
            copy_input(event)

        @bindings.add("c-x", filter=selected, eager=True)
        def cut_selected(event):
            copy_input(event, cut=True)

        @bindings.add("c-v", filter=editable, eager=True)
        def paste(event):
            try:
                text = paste_text()
                event.app.clipboard.set_data(ClipboardData(text))
            except MlxChatError:
                text = event.app.clipboard.get_data().text
            if event.current_buffer.selection_state:
                event.current_buffer.cut_selection()
            event.current_buffer.insert_text(text)

        @bindings.add("c-u", filter=editable, eager=True)
        @bindings.add("escape", "backspace", filter=editable, eager=True)
        def clear_input(event):
            event.current_buffer.reset()

        @bindings.add("backspace", filter=selected, eager=True)
        @bindings.add("delete", filter=selected, eager=True)
        def delete_selected(event):
            event.current_buffer.cut_selection()

        @bindings.add(Keys.BracketedPaste, filter=selected)
        @bindings.add(Keys.Any, filter=selected)
        def replace_selected(event):
            event.current_buffer.cut_selection()
            event.current_buffer.insert_text(event.data)

        @bindings.add("escape", "enter")
        def _newline(event) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("escape")
        def _stop(event) -> None:
            event.app.exit(result="/stop")

        @bindings.add("c-l")
        def _clear(event) -> None:
            event.app.renderer.clear()

        @bindings.add("f1")
        def _voice(event) -> None:
            event.app.exit(result="/voice")

        @bindings.add("f2")
        def _speak(event) -> None:
            event.app.exit(result="/speak")

        prompt_style = PromptStyle.from_dict(
            {}
            if plain
            else {
                "user-border": BORDER,
                "user-title": f"bold {ACCENT}",
                "hint": MUTED,
                "composer-hint": MUTED,
                "input-placeholder": MUTED,
                "selected-text": f"bg:{BORDER} #ffffff",
                "composer": f"bg:{SURFACE} #e3e6eb",
                "bottom-toolbar": "bg:#262b36 #aab2c0",
                "completion-menu.completion": f"bg:{SURFACE} #d7dce5",
                "completion-menu.completion.current": f"bg:{BORDER} #ffffff bold",
            }
        )
        self.prompt_session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path())),
            completer=FuzzyWordCompleter(SLASH_COMMANDS, WORD=True),
            complete_while_typing=False,
            placeholder=[("class:input-placeholder", "Write a message… / for commands")],
            key_bindings=bindings,
            style=prompt_style,
            multiline=False,
            enable_open_in_editor=True,
            mouse_support=False,
            erase_when_done=self._interactive_prompt,
        )

        # Frame only the editable input; keep the status toolbar outside it.
        root = self.prompt_session.layout.container
        self._input_float_container = root.children[0].alternative_content
        self.prompt_session.layout.current_window.dont_extend_height = to_filter(True)

        input_container = root.children[0]
        root.children[0] = HSplit(
            [
                Window(height=1),
                VSplit(
                    [
                        Window(width=2),
                        Window(FormattedTextControl([("class:user-title", "● ")]), width=2),
                        input_container,
                        Window(width=2),
                    ],
                    height=lambda: min(
                        max(3, self._composer_input_rows()),
                        max(2, self.prompt_session.app.output.get_size().rows // 2 - 3),
                    ),
                ),
                Window(height=1),
                Window(height=1),
            ],
            style="class:composer",
        )

    def _composer_input_rows(self):
        from prompt_toolkit.utils import get_cwidth

        conversation = getattr(self, "_conversation", None)
        width = max(1, (conversation.composer_width() if conversation else self.console.width) - 6)
        return sum(
            max(1, (get_cwidth(line.expandtabs(4)) + width) // width)
            for line in self.prompt_session.default_buffer.text.split("\n")
        )

    def enable_conversation(self) -> None:
        if self._interactive_prompt and self._conversation is None:
            from .conversation import ConversationView

            self._conversation = ConversationView(self)

    def close_conversation(self) -> None:
        if self._conversation is not None:
            self._conversation.close()
            self._conversation = None

    def startup_status(self, message):
        return StartupStatus(self.console, self.settings.model, message)

    def exit_summary(self) -> None:
        title = Text("◆ mlx-chat", style=f"bold {ACCENT}")
        title.append("  ·  Chat closed", style=MUTED)
        details = Text(
            f"  {short_model(self.settings.model)}  ·  {self._message_count // 2} turns\n",
            style="dim",
        )
        details.append(f"  Session: {self._session_name or 'unsaved'}\n")
        details.append(
            "  Server stopped"
            if self.server_stopped
            else "  Server left running  ·  mlx-chat stop to release memory"
        )
        self.console.print()
        self.console.print(
            Panel(
                Group(title, details),
                box=box.ROUNDED,
                border_style=BORDER,
                padding=(0, 2),
                width=min(68, self.console.width),
            )
        )
        self.console.print()

    def set_context(self, session_name: str | None, message_count: int) -> None:
        self._session_name = session_name
        self._message_count = message_count

    def info_panel(self):
        """A slim one-line header: wordmark, model and mode on the left, state on the right."""
        thinking = self.settings.think and not self.settings.concise
        left = Text(no_wrap=True, overflow="ellipsis")
        left.append("◆ ", style=BRAND_START)
        for character, color in gradient("mlx-chat"):
            left.append(character, style=f"bold {color}")
        left.append("   ")
        left.append(short_model(self.settings.model), style="bold #e5e7eb")
        modes = []
        agent = getattr(self, "agent_status", None)
        modes.append(("agent " + agent, SUCCESS) if agent else ("chat", MUTED))
        if thinking:
            modes.append(("thinking", MUTED))
        if self.settings.concise:
            modes.append(("concise", MUTED))
        stt_ready = self._banner_state["voice_ready"] or voice.is_ready()
        if stt_ready or tts.is_ready():
            modes.append(("voice ready", ACCENT))
        for label, color in modes:
            left.append("  ·  ", style=BORDER)
            left.append(label, style=color)

        right = Text(no_wrap=True, overflow="ellipsis", justify="right")
        share = getattr(self, "context_share", None)
        if share is not None:
            share = max(0.0, min(1.0, share))
            filled = round(share * 8)
            color = SUCCESS if share < 0.6 else WARNING if share < 0.85 else DANGER
            right.append("context ", style=MUTED)
            right.append("▰" * filled, style=color)
            right.append("▱" * (8 - filled), style=BORDER)
            right.append(f" {share:.0%}", style=MUTED)
            right.append("  ·  ", style=BORDER)
        turns = self._message_count // 2
        right.append(
            f"{self._session_name or 'unsaved'} · {turns} turn{'' if turns == 1 else 's'}",
            style=MUTED,
        )
        bar = Table.grid(expand=True, padding=(0, 2))
        bar.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        bar.add_column(no_wrap=True, justify="right")
        bar.add_row(left, right)
        return Group(Padding(bar, (0, 2)), Rule(style=BORDER, characters="─"))

    def banner(self, *, reused: bool, managed: bool, voice_ready: bool = False) -> None:
        self._banner_state = {"reused": reused, "managed": managed, "voice_ready": voice_ready}
        if self._conversation is None:
            self.console.print()
            self.console.print(self.info_panel())
        if self._conversation is None:
            self.console.print(Text(self.conversation_hint(), justify="center", style="dim"))

    @staticmethod
    def conversation_hint() -> str:
        return "↵ send  ·  esc ↵ newline  ·  / commands  ·  /plan plans first  ·  F1 F2 F3 voice"

    @staticmethod
    def message_content(content, *, user=False):
        row = Table.grid(expand=True, padding=0)
        row.add_column(width=2, no_wrap=True)
        row.add_column(ratio=1, no_wrap=False, overflow="fold")
        row.add_row(Text("●" if user else "◆", style=ACCENT if user else MUTED), content)
        return row

    def chat_panel(self, content, *, user=False, marker=True, compact_bottom=False, **options):
        panel = AnswerPanel if compact_bottom else Panel
        return panel(
            self.message_content(content, user=user) if marker else content,
            box=box.Box("    \n" * 8),
            style=f"on {SURFACE}" if user else "",
            padding=(0, 1),
            **options,
        )

    def thinking_trace(self, text):
        trace = ThinkingTrace(text)
        if getattr(self, "_conversation", None) is not None:
            self._conversation.thinking_traces[id(trace)] = trace
        else:
            trace.expanded = True
        return trace

    def user_line(self, value: str) -> Panel:
        return self.chat_panel(
            Text(sanitize_terminal_text(value)),
            user=True,
        )

    @property
    def cancellation_event(self):
        return self._conversation.cancelled if self._conversation is not None else None

    def restore_transcript(self, messages, *, replace=True):
        if replace:
            self.clear()
        for message in messages:
            if message["role"] == "user":
                self.user_message(message["content"])
                if message.get("attachments"):
                    self.info("Attached excerpts: " + ", ".join(message["attachments"]))
            elif message["role"] == "tool":
                self.tool_result(message)
            elif message["role"] == "summary":
                self.summary_notice(message)
            else:
                result = result_from_message(message)
                self.final_generation(result, think=bool(result.reasoning), historical=True)

    def run_audio(self, operation, *, stop_key="f3"):
        if self._conversation is None:
            return operation(None)
        panel = self._conversation.pending
        self._conversation.audio_key = stop_key
        try:
            result = self._conversation.generate(
                lambda callback: operation(self._conversation.cancelled), lambda _: panel
            )
            self._conversation.finish_audio()
            return result
        finally:
            self._conversation.audio_key = None

    def run_task(self, operation, *, label: str | None = None):
        if self._conversation is None:
            return operation()
        panel = None
        if label:
            spinner = Spinner("dots", text=Text(label), style=MUTED)
            panel = self.chat_panel(spinner)
            panel.renderable = self.message_content(spinner)
        return self._conversation.generate(lambda callback: operation(), lambda _: panel)

    def summary_notice(self, item, *, automatic: bool = False) -> None:
        trace = self.thinking_trace(item.get("content", ""))
        trace.label = f"Summarized {item.get('covers', 0)} earlier messages" + (
            " to free context" if automatic else ""
        )
        self.console.print(trace)

    def plan_card(self, plan) -> None:
        self.console.print(self.chat_panel(plan_checklist(plan)))

    def ask_plan(self) -> tuple[str, str]:
        """After a plan: ("run" | "revise" | "keep", feedback)."""
        answer = self._ask(PLAN_PLACEHOLDER, plan_prompt_card(), "Run this plan? [Y/n/changes] ")
        value = answer.strip()
        lowered = value.lower()
        if lowered in {"", "y", "yes", "go", "run", "ok"}:
            return "run", ""
        if lowered in {"n", "no", "keep", "later"} or value.startswith("/"):
            return "keep", ""
        return "revise", value

    def _ask(self, placeholder: str, card, plain_question: str) -> str:
        conversation = self._conversation
        if conversation is None:
            self.console.print(card)
            if not self.console.is_terminal:
                return "n"
            return self.console.input(plain_question)
        buffer = self.prompt_session.default_buffer
        history = buffer.history
        conversation.approval = placeholder
        conversation.follow = True
        conversation.update(card)
        # Answers to questions are not chat input; keep them out of the prompt history.
        buffer.history = InMemoryHistory()
        try:
            return self.prompt_session.prompt("", default="", bottom_toolbar=None)
        finally:
            buffer.history = history
            conversation.approval = False
            conversation.update(None)

    def run_native(self, operation):
        if self._conversation is None:
            return operation()
        return self._conversation.run_native(operation)

    def run_tool(self, label: str, operation):
        """Run a tool with a spinner; Esc or Ctrl-C sets the event passed to it."""
        if self._conversation is None:
            return operation(None)
        spinner = Spinner("dots", text=Text(sanitize_terminal_text(label)), style=MUTED)
        panel = self.chat_panel(spinner)
        panel.renderable = self.message_content(spinner)
        conversation = self._conversation
        return conversation.generate(
            lambda _callback: operation(conversation.cancelled), lambda _: panel
        )

    def ask_approval(self, name: str, arguments_text: str, preview=None) -> tuple[str, str]:
        """Ask before a tool changes files or runs a command.

        Returns ("once" | "always" | "deny", feedback). Ctrl-C raises KeyboardInterrupt.
        """
        description = tool_description({"name": name, "arguments": arguments_text})
        card = approval_card(name, description, preview)
        answer = self._ask(APPROVAL_PLACEHOLDER, card, "Allow? [Y/a/n] ")
        value = answer.strip()
        lowered = value.lower()
        if lowered in {"", "y", "yes", "ok", "allow"}:
            return "once", ""
        if lowered in {"a", "always"}:
            return "always", ""
        if lowered in {"n", "no", "deny", "decline"} or value.startswith("/"):
            return "deny", ""
        return "deny", value

    def audio_status(self, value: str):
        if getattr(self, "_conversation", None) is not None:
            return self._conversation.status(value, user=value.startswith("/voice"))
        # Keep the spinner inside the message panel.
        spinner = Spinner("dots", text=Text(sanitize_terminal_text(value)), style=ACCENT)
        panel = self.chat_panel(Text(value))
        panel.renderable = self.message_content(spinner)
        return Live(panel, console=self.console, transient=True, refresh_per_second=10)

    def preview_command(self, command):
        if command == "/help":
            self.help()
        else:
            self.info(f"{command}\nPress Return to run this command.")

    @contextmanager
    def command_popup(self, title):
        conversation = getattr(self, "_conversation", None)
        if conversation is None:
            yield
        else:
            with conversation.popup(title):
                yield

    def voice_message(self, text):
        self.user_message(f"/voice {text}")

    def user_message(self, value: str) -> None:
        self.console.print(self.user_line(value))

    def prompt(self) -> str:
        conversation = getattr(self, "_conversation", None)
        if conversation is not None and conversation.submitted_inputs:
            value = conversation.submitted_inputs.popleft()
        else:
            try:
                value = self.prompt_session.prompt(
                    "",
                    default=conversation.draft_text if conversation is not None else "",
                    bottom_toolbar=None,
                    prompt_continuation=lambda width, _line, _wrap: " " * width,
                )
            except (KeyboardInterrupt, EOFError):
                if conversation is not None:
                    conversation.dismiss_preview()
                raise
        if conversation is not None:
            conversation.draft_text = ""
            conversation.dismiss_preview()
        if value.strip() and (
            not value.lstrip().startswith("/") or value.lstrip().startswith(("//", "/web "))
        ):
            self.user_message(value)
        return value

    def info(self, message: str) -> None:
        self.console.print(self.chat_panel(Text(sanitize_terminal_text(message))))

    def success(self, message: str) -> None:
        self.info(message)

    def warning(self, message: str) -> None:
        self.info(message)

    def error(self, message: str) -> None:
        self.info(message)

    def rule(self, title: str = "") -> None:
        self.console.rule(Text(title), style="dim")

    def clear(self) -> None:
        if getattr(self, "_conversation", None) is not None:
            self._conversation.items.clear()
            self._conversation.thinking_traces.clear()
            self._conversation.trace_item_indices.clear()
            self._conversation.selection = None
            self._conversation._history_width = None
            self._conversation.update(None)
            self._conversation.follow = True
            return
        output = self.console.file
        try:
            is_terminal = bool(output.isatty())
        except (AttributeError, OSError):
            is_terminal = False
        if not is_terminal:
            self.console.clear()
            return

        # Rich's clear() does not erase terminal scrollback (E3).
        output.write("\x1b[3J\x1b[H\x1b[2J")
        output.flush()

    def help(self) -> None:
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style=f"bold {ACCENT}",
            pad_edge=False,
        )
        table.add_column("Command", no_wrap=True)
        table.add_column("What it does")
        rows = [
            (
                "/attach [PATH ...]",
                "Attach text files/project excerpts to the next prompt, or list them",
            ),
            ("/detach", "Clear pending file attachments"),
            (
                "/profile [NAME|save NAME|delete NAME]",
                "Apply, save, delete, or list generation profiles",
            ),
            ("/agent [on|off]", "Let the model use tools in this folder, or plain chat"),
            ("/plan [TASK]", "Plan first with read-only tools, then run or revise the plan"),
            ("/approve [ask|edits|all]", "Ask before edits and commands, or auto-allow them"),
            ("/memory [add TEXT|reload]", "Show or add to the project's AGENTS.md memory"),
            ("/compact", "Summarize older messages to free context"),
            ("/tools", "List agent tools and which need approval"),
            ("/web QUERY", "Search the web and generate an answer with source links"),
            ("/think [on|off|toggle]", "Control Qwen thinking on future turns"),
            ("/concise [on|off|toggle]", "Prefer fast, brief direct answers"),
            ("/voice [seconds]", "Record a local voice prompt and transcribe it offline"),
            ("/speak [answer|reasoning|all]", "Read aloud in English with Kokoro"),
            ("/new [name]", "Start a clean conversation"),
            ("/reset", "Clear conversation context"),
            ("/system [text|@file|clear]", "Show or change the system prompt"),
            ("/retry", "Regenerate the last response"),
            ("/undo", "Remove the last user/assistant turn"),
            ("/settings", "Show generation settings"),
            ("/temperature N", "Set sampling temperature"),
            ("/top-p N · /top-k N", "Set sampling filters"),
            ("/max-tokens N", "Set the response token limit"),
            ("/stats", "Show metrics for the last response"),
            ("/status", "Show server and conversation status"),
            ("/model [name]", "Show or switch the loaded model"),
            ("/save NAME · /load NAME", "Save or restore a session"),
            ("/sessions", "List saved sessions"),
            ("/history [N]", "Show recent conversation turns"),
            ("/copy [answer|reasoning|all]", "Copy the last response with pbcopy"),
            ("/export [md|json] [path]", "Export this conversation"),
            ("/logs [N]", "Show recent MLX server logs"),
            ("/restart", "Restart the managed server"),
            ("/clear", "Blank the terminal; keep the conversation unchanged"),
            ("/stop", "Stop the managed server and exit"),
            ("/exit", "Exit chat; leave the server running"),
        ]
        for command, description in rows:
            cell = Text()
            for index, option in enumerate(command.split(" · ")):
                if index:
                    cell.append(" · ", style="dim")
                name = option.split()[0]
                insert = name + (" " if len(option.split()) > 1 else "")
                cell.append(
                    option,
                    style=RichStyle(color=ACCENT, underline=True, meta={"help_command": insert}),
                )
            table.add_row(cell, description)
        self.console.print(
            self.chat_panel(
                table,
                marker=False,
            )
        )
        self.generation_info(
            "// sends a literal slash · Esc+Enter adds a line\n"
            "F1 listen · F2 speak · F3 stop · Ctrl-C cancels"
        )
