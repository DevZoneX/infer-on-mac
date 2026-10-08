"""Rich and prompt-toolkit terminal presentation."""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.filters import Condition, to_filter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style as PromptStyle
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.style import Style as RichStyle
from rich.table import Table
from rich.text import Text

from . import tts, voice
from .client import GenerationResult, StreamEvent
from .config import Settings, history_path
from .resources import resource_snapshot
from .theme import ACCENT, BORDER, MUTED, SURFACE

ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

READ_ONLY_COMMANDS = {
    "/help",
    "/system",
    "/settings",
    "/stats",
    "/status",
    "/model",
    "/sessions",
    "/history",
    "/logs",
}

SLASH_COMMANDS = [
    "/help",
    "/web",
    "/think",
    "/concise",
    "/voice",
    "/speak",
    "/new",
    "/reset",
    "/clear",
    "/system",
    "/retry",
    "/undo",
    "/settings",
    "/temperature",
    "/top-p",
    "/top-k",
    "/max-tokens",
    "/stats",
    "/status",
    "/model",
    "/save",
    "/load",
    "/sessions",
    "/history",
    "/copy",
    "/export",
    "/logs",
    "/restart",
    "/stop",
    "/exit",
]


def sanitize_terminal_text(value: str) -> str:
    """Remove terminal control sequences while preserving newlines and tabs."""
    value = ANSI_RE.sub("", value)
    return "".join(
        char
        for char in value
        if char in {"\n", "\t"} or (ord(char) >= 32 and not 127 <= ord(char) <= 159)
    )


def short_model(model: str, maximum: int = 38) -> str:
    value = model.rsplit("/", 1)[-1]
    return value if len(value) <= maximum else f"{value[: maximum - 1]}…"


@dataclass
class RenderState:
    result: GenerationResult
    think: bool
    started: float


class ThinkingTrace(Panel):
    """A compact label whose trace can be expanded with the mouse."""

    def __init__(self, text=""):
        super().__init__(Text("Thinking"))
        self.text = text
        self.expanded = False
        self.loading = False
        self.spinner = Spinner("dots")

    def __rich_console__(self, console, options):
        label = Text()
        if self.loading:
            label.append(self.spinner.render(time.monotonic()))
        else:
            label.append("◆")
        label.append(" Thinking")
        label.append(
            "  ·  click to collapse" if self.expanded else "  ·  click to expand", style="dim"
        )
        label.stylize(RichStyle(color=MUTED, meta={"thinking_trace": id(self)}))
        content = [label]
        if self.expanded:
            content.append(
                Padding(Text(sanitize_terminal_text(self.text) or "Thinking…", style="dim"), (0, 2))
            )
        yield Padding(Group(*content), (1, 2, 0, 2))


class AnswerPanel(Panel):
    """Keep the response and its metadata together without a bottom spacer."""

    def __rich_console__(self, console, options):
        yield Padding(self.renderable, (1, 2, 0, 2))


class StartupStatus:
    """A calm startup card with one changing status line."""

    def __init__(self, console, model, message):
        self.console = console
        self.model = model
        self.message = message
        self.live = Live(self.render(), console=console, transient=True, refresh_per_second=8)

    def render(self):
        message = self.message.plain if isinstance(self.message, Text) else str(self.message)
        content = Group(
            Text("◆ mlx-chat", style=f"bold {ACCENT}"),
            Text(short_model(self.model), style="bold"),
            Text("Local intelligence · Apple Silicon", style="dim"),
            Text(""),
            Spinner("dots", text=Text(message, style=MUTED), style=ACCENT),
        )
        return Panel(
            content,
            box=box.ROUNDED,
            border_style=BORDER,
            padding=(1, 2),
            width=min(68, self.console.width),
        )

    def update(self, message):
        self.message = message
        self.live.update(self.render())

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, *args):
        return self.live.__exit__(*args)


class TerminalUI:
    def __init__(self, settings: Settings, *, plain: bool = False):
        force_terminal = None if not plain else False
        self.console = Console(force_terminal=force_terminal, no_color=plain)
        self._interactive_prompt = self.console.is_terminal and not plain
        self.settings = settings
        self._session_name: str | None = None
        self._message_count = 0
        self._conversation = None
        self.server_stopped = False
        self.command_previewer = self.preview_command
        self.last_preview_command = None
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
                subprocess.run(["pbcopy"], input=data.text, text=True, check=True, timeout=2)
            except (OSError, subprocess.SubprocessError):
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
                text = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True, check=True, timeout=2
                ).stdout
                event.app.clipboard.set_data(ClipboardData(text))
            except (OSError, subprocess.SubprocessError):
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
        def _newline(event) -> None:  # type: ignore[no-untyped-def]
            event.current_buffer.insert_text("\n")

        @bindings.add("escape")
        def _stop(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result="/stop")

        @bindings.add("c-l")
        def _clear(event) -> None:  # type: ignore[no-untyped-def]
            event.app.renderer.clear()

        @bindings.add("f1")
        def _voice(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result="/voice")

        @bindings.add("f2")
        def _speak(event) -> None:  # type: ignore[no-untyped-def]
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

    def info_panel(self) -> Panel:
        thinking = self.settings.think and not self.settings.concise
        stt_ready = self._banner_state["voice_ready"] or voice.is_ready()
        heading = Text(justify="center", no_wrap=True, overflow="ellipsis")
        heading.append(short_model(self.settings.model), style=f"bold {ACCENT}")
        heading.append(f"  ·  Thinking: {'on' if thinking else 'off'}", style=MUTED)
        if self.settings.concise:
            heading.append("  ·  Concise", style=MUTED)
        details = Table.grid(expand=True, padding=(0, 1))
        for _ in range(3):
            details.add_column(ratio=1, justify="center", no_wrap=True, overflow="ellipsis")
        audio_cells = []
        for label, model, ready in (
            ("TTS", tts.DEFAULT_TTS_MODEL, tts.is_ready()),
            ("STT", voice.DEFAULT_TRANSCRIPTION_MODEL, stt_ready),
        ):
            cell = Text(f"{label}: {short_model(model)} · ", style="dim")
            cell.append("ready" if ready else "not loaded", style=ACCENT if ready else "dim")
            audio_cells.append(cell)
        session = Text(
            f"Session: {self._session_name or 'unsaved'} · Turns: {self._message_count // 2}",
            style="dim",
        )
        details.add_row(*audio_cells, session)
        content = Group(heading, details)
        return Panel(
            content,
            title="[bold]mlx-chat[/bold] · local",
            title_align="left",
            border_style=BORDER,
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def banner(self, *, reused: bool, managed: bool, voice_ready: bool = False) -> None:
        self._banner_state = {"reused": reused, "managed": managed, "voice_ready": voice_ready}
        if self._conversation is None:
            self.console.print()
            self.console.print(self.info_panel())
        if self._conversation is None:
            self.console.print(Text(self.conversation_hint(), justify="center", style="dim"))

    @staticmethod
    def conversation_hint() -> str:
        return "Try /help · F1 listen · F2 speak · F3 stop · ↵ send · esc+↵ newline"

    @staticmethod
    def message_content(content, *, user=False):
        row = Table.grid(expand=True, padding=0)
        row.add_column(width=2, no_wrap=True)
        row.add_column(ratio=1, no_wrap=False, overflow="fold")
        row.add_row(Text("●" if user else "◆", style=ACCENT if user else MUTED), content)
        return row

    def chat_panel(
        self,
        content,
        *,
        title="",
        border_style="bright_cyan",
        marker=True,
        compact_bottom=False,
        **options,
    ) -> Panel:
        user = border_style == "green"
        background = SURFACE if user else None
        options.pop("box", None)
        options.pop("title_align", None)
        options["padding"] = (0, 1)
        panel_type = AnswerPanel if compact_bottom else Panel
        return panel_type(
            self.message_content(content, user=user) if marker else content,
            box=box.Box("    \n" * 8),
            style=f"on {background}" if background else "",
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
            title="●",
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def run_audio(self, operation, *, stop_key="f3"):
        if self._conversation is None:
            return operation(None)
        panel = self._conversation.pending
        self._conversation.audio_key = stop_key
        try:
            return self._conversation.generate(
                lambda callback: operation(self._conversation.cancelled), lambda _: panel
            )
        finally:
            self._conversation.audio_key = None

    def run_task(self, operation):
        if self._conversation is None:
            return operation()
        return self._conversation.generate(lambda callback: operation(), lambda _: None)

    def run_native(self, operation):
        if self._conversation is None:
            return operation()
        return self._conversation.run_native(operation)

    def audio_status(self, value: str):
        if getattr(self, "_conversation", None) is not None:
            return self._conversation.status(value, user=value.startswith("/voice"))
        # Put the animation in the panel's content row, between its borders.
        spinner = Spinner("dots", text=Text(sanitize_terminal_text(value)), style=ACCENT)
        panel = self.chat_panel(Text(value))
        panel.renderable = self.message_content(spinner)
        self._anchor_bottom(rows=3)
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

    def _anchor_bottom(self, rows: int = 4) -> None:
        """Reserve bottom rows in the normal screen buffer, preserving scrollback."""
        if (
            not getattr(self, "_interactive_prompt", False)
            or getattr(self, "_conversation", None) is not None
        ):
            return
        height = self.console.size.height
        rows = min(rows, max(1, height))
        output = self.console.file
        output.write(f"\x1b[{height};1H" + "\n" * rows + f"\x1b[{height - rows + 1};1H")
        output.flush()

    def prompt(self) -> str:
        self._anchor_bottom()
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
            self.last_preview_command = (
                conversation.popup_title
                if conversation.popup_preview and value.strip().lower() == conversation.popup_title
                else None
            )
            conversation.dismiss_preview()
        if value.strip() and (
            not value.lstrip().startswith("/") or value.lstrip().startswith(("//", "/web "))
        ):
            self.user_message(value)
        return value

    def _notice(self, symbol: str, style: str, message: str) -> None:
        self.console.print(
            self.chat_panel(
                Text(sanitize_terminal_text(message)),
                title=Text(f"{symbol} System", style="bright_cyan"),
                title_align="left",
                border_style="bright_cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def info(self, message: str) -> None:
        self._notice("◆", "cyan", message)

    def success(self, message: str) -> None:
        self._notice("✓", "green", message)

    def warning(self, message: str) -> None:
        self._notice("⚠", "yellow", message)

    def error(self, message: str) -> None:
        self._notice("Error:", "bold red", message)

    def rule(self, title: str = "") -> None:
        self.console.rule(Text(title), style="dim")

    def clear(self) -> None:
        if getattr(self, "_conversation", None) is not None:
            self._conversation.items.clear()
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

        # Match macOS' `clear` command: erase scrollback, move home, then
        # erase the visible viewport. Rich's Console.clear() omits E3, which
        # leaves the previous conversation reachable in terminal scrollback.
        output.write("\x1b[3J\x1b[H\x1b[2J")
        output.flush()

    def _live_text(
        self,
        value: str,
        *,
        style: str,
        width: int,
        max_lines: int,
        placeholder: str,
    ) -> Text:
        """Build a tail-following text viewport with a fixed line budget."""

        source = Text(sanitize_terminal_text(value) or placeholder, style=style)
        wrapped = source.wrap(self.console, max(1, width), overflow="fold")
        max_lines = max(1, max_lines)
        if len(wrapped) <= max_lines:
            return Text("\n").join(wrapped)

        if max_lines == 1:
            tail = wrapped[-1].copy()
            tail.truncate(max(1, width), overflow="ellipsis")
            return tail

        visible = wrapped[-(max_lines - 1) :]
        output = Text("…", style="dim")
        for line in visible:
            output.append("\n")
            output.append_text(line)
        return output

    def _live_markdown(
        self, value: str, *, width: int, max_lines: int, placeholder: str
    ) -> Text | Group:
        """Render streamed Markdown on every refresh, retaining link metadata."""
        source = sanitize_terminal_text(value)
        if not source:
            return Text(placeholder, style="dim italic")
        # Render the whole document first so headings, lists, code, and URLs
        # are styled while tokens are arriving. Keep its tail afterward so a
        # long generation continues to follow the newest text in a small pane.
        lines = self.console.render_lines(
            Markdown(source, code_theme="monokai", hyperlinks=True),
            self.console.options.update(width=max(1, width)),
            pad=False,
        )
        visible = lines[-max(1, max_lines) :]
        rendered_lines = []
        for line in visible:
            text = Text()
            for segment in line:
                text.append(segment.text, style=segment.style)
            rendered_lines.append(text)
        return Group(*rendered_lines)

    def _live_panel(
        self,
        value: str,
        *,
        title: str,
        style: str,
        border_style: str,
        height: int,
        placeholder: str,
    ) -> Panel:
        content_width = max(1, self.console.size.width - 4)
        content = (
            self._live_markdown(
                value,
                width=content_width,
                max_lines=max(1, height - 2),
                placeholder=placeholder,
            )
            if title.endswith("Answer[/bright_cyan]")
            else self._live_text(
                value,
                style=style,
                width=content_width,
                max_lines=max(1, height - 2),
                placeholder=placeholder,
            )
        )
        return self.chat_panel(
            content,
            title=title,
            title_align="left",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
            height=min(
                height,
                max(
                    3,
                    len(
                        self.console.render_lines(
                            content, self.console.options.update(width=content_width), pad=False
                        )
                    )
                    + 2,
                ),
            ),
        )

    def _live_group(self, state: RenderState) -> Group:
        height = self.console.size.height
        if getattr(self, "_interactive_prompt", False) and height >= 8:
            response = self._live_response_group(state, terminal_height=height - 3)
            return Group(response, self.chat_panel(Text("Generating… Ctrl-C to cancel")))
        return self._live_response_group(state)

    def _live_response_group(
        self, state: RenderState, *, terminal_height: int | None = None
    ) -> Group:
        result = state.result
        elapsed = time.monotonic() - state.started
        terminal_height = max(1, terminal_height or self.console.size.height)
        terminal_width = max(1, self.console.size.width)
        status = Text(
            f"  {elapsed:.1f}s · Ctrl-C to cancel",
            style="dim",
            overflow="ellipsis",
            no_wrap=True,
        )

        show_reasoning = state.think or bool(result.reasoning)
        answer_started = bool(result.answer)

        # Extremely short or narrow terminals can't accommodate panel borders.
        # Keep a compact tail plus status instead of allowing Rich to scroll.
        if terminal_height < 4 or terminal_width < 12:
            active = result.answer if answer_started else result.reasoning
            placeholder = "Waiting…" if answer_started else "Thinking…"
            preview_lines = terminal_height - 1 if terminal_height > 1 else 1
            preview = self._live_text(
                active,
                style="",
                width=terminal_width,
                max_lines=preview_lines,
                placeholder=placeholder,
            )
            return Group(preview, status) if terminal_height > 1 else Group(preview)

        panel_budget = terminal_height - 1  # reserve one row for elapsed time
        if show_reasoning and terminal_height >= 7:
            if answer_started:
                reasoning_height = max(3, (panel_budget * 2) // 5)
                answer_height = panel_budget - reasoning_height
                if answer_height < 3:
                    answer_height = 3
                    reasoning_height = panel_budget - answer_height
            else:
                answer_height = 3
                reasoning_height = panel_budget - answer_height
            reasoning_panel = self._live_panel(
                result.reasoning,
                title="[yellow]◆ Reasoning[/yellow]",
                style="dim italic",
                border_style="yellow",
                height=reasoning_height,
                placeholder="Thinking…",
            )
            answer_panel = self._live_panel(
                result.answer,
                title="[bright_cyan]◆ Answer[/bright_cyan]",
                style="",
                border_style="bright_cyan",
                height=answer_height,
                placeholder="Waiting for the answer…",
            )
            return Group(reasoning_panel, answer_panel, status)

        panel_height = max(3, panel_budget)
        if show_reasoning and not answer_started:
            panel = self._live_panel(
                result.reasoning,
                title="[yellow]◆ Reasoning[/yellow]",
                style="dim italic",
                border_style="yellow",
                height=panel_height,
                placeholder="Thinking…",
            )
        else:
            panel = self._live_panel(
                result.answer,
                title="[bright_cyan]◆ Answer[/bright_cyan]",
                style="",
                border_style="bright_cyan",
                height=panel_height,
                placeholder="Waiting for the answer…",
            )
        return Group(panel, status)

    def render_generation(
        self,
        generate: Callable[[Callable[[StreamEvent, GenerationResult], None]], GenerationResult],
        *,
        think: bool,
    ) -> GenerationResult:
        from rich.live import Live

        if getattr(self, "_conversation", None) is not None:
            trace = self.thinking_trace("")

            def preview(result):
                result = result or GenerationResult()
                panels = []
                if think or result.reasoning:
                    trace.text = result.reasoning
                    trace.loading = not bool(result.answer)
                    panels.append(trace)
                panels.append(
                    self.chat_panel(
                        Markdown(
                            sanitize_terminal_text(result.answer) or "Waiting…",
                            code_theme="monokai",
                            hyperlinks=True,
                        ),
                        title="◆ Answer",
                        compact_bottom=True,
                        title_align="left",
                        border_style="bright_cyan",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
                return Group(*panels)

            result = self._conversation.generate(generate, preview, allow_input=True)
            self.final_generation(result, think=think)
            return result

        if getattr(self, "_interactive_prompt", False):
            self._anchor_bottom(rows=7)
        preview = GenerationResult()
        state = RenderState(preview, think, time.monotonic())
        last_refresh = 0.0
        live = Live(
            self._live_group(state),
            console=self.console,
            auto_refresh=False,
            transient=True,
            vertical_overflow="ellipsis",
        )

        def on_event(_event: StreamEvent, result: GenerationResult) -> None:
            nonlocal last_refresh
            state.result = result
            now = time.monotonic()
            if now - last_refresh >= 0.065:
                live.update(self._live_group(state), refresh=True)
                last_refresh = now

        with live:
            try:
                result = generate(on_event)
            finally:
                # Rich forces `vertical_overflow="visible"` during Live.stop().
                # Leave it a one-line final renderable so an exception or a
                # very long response can never corrupt rows above the viewport.
                live.update(Text(""), refresh=True)
        self.final_generation(result, think=think)
        return result

    def final_generation(self, result: GenerationResult, *, think: bool) -> None:
        if result.reasoning:
            self.console.print(self.thinking_trace(result.reasoning))
        elif think:
            self.warning("The model returned no separate reasoning channel for this turn.")

        answer = sanitize_terminal_text(result.answer)
        if answer:
            renderable = Markdown(answer, code_theme="monokai", hyperlinks=True)
        else:
            renderable = Text("No final answer was returned.", style="italic dim")
        self.console.print(
            self.chat_panel(
                renderable,
                title="[bright_cyan]◆ Answer[/bright_cyan]",
                compact_bottom=True,
                title_align="left",
                border_style="bright_cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

        if result.finish_reason == "length" and result.reasoning and not result.answer:
            self.warning(
                "The token limit was reached during reasoning. Try /max-tokens 8192 and retry."
            )
        metrics = [f"{result.elapsed:.1f}s"]
        if result.first_token_seconds is not None:
            metrics.append(f"TTFT {result.first_token_seconds:.1f}s")
        if result.completion_tokens is not None:
            metrics.append(f"{result.completion_tokens} tokens")
        if result.tokens_per_second is not None:
            metrics.append(f"{result.tokens_per_second:.1f} tok/s")
        if result.finish_reason:
            metrics.append(f"finish: {result.finish_reason}")
        snapshot = resource_snapshot()
        server = next(
            (item for item in snapshot.processes if item.port == self.settings.port), None
        )
        if server is not None:
            if server.cpu_percent is not None:
                metrics.append(f"CPU {server.cpu_percent:.1f}%")
            if server.memory_mib is not None:
                metrics.append(f"RAM {server.memory_mib:,.0f} MiB")
        self.generation_info(" · ".join(metrics))

    def generation_info(self, value: str) -> None:
        metadata = Text("    " + value, style="dim", no_wrap=True, overflow="ellipsis")
        conversation = getattr(self, "_conversation", None)
        if conversation is not None:
            if conversation.popup_items is None or conversation.popup_preview:
                metadata.end = "\n\n\n"
            conversation.print_info(metadata)
        else:
            self.console.print(metadata)

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
            self.chat_panel(table, marker=False, border_style="bright_cyan", box=box.ROUNDED)
        )
        self.generation_info(
            "// sends a literal slash · Esc+Enter adds a line\n"
            "F1 listen · F2 speak · F3 stop · Ctrl-C cancels"
        )


def is_plain_terminal() -> bool:
    return "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb"
