"""Rich and prompt-toolkit terminal presentation."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from html import escape as html_escape

from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PromptStyle
from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .client import GenerationResult, StreamEvent
from .config import Settings, history_path
from .resources import resource_snapshot

ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

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


class TerminalUI:
    def __init__(self, settings: Settings, *, plain: bool = False):
        force_terminal = None if not plain else False
        self.console = Console(force_terminal=force_terminal, no_color=plain)
        self.settings = settings
        self._session_name: str | None = None
        self._message_count = 0

        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event) -> None:  # type: ignore[no-untyped-def]
            event.current_buffer.insert_text("\n")

        @bindings.add("c-l")
        def _clear(event) -> None:  # type: ignore[no-untyped-def]
            event.app.renderer.clear()

        @bindings.add("f1")
        def _voice(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result="/voice")

        @bindings.add("f3")
        def _speak(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result="/speak")

        prompt_style = PromptStyle.from_dict(
            {}
            if plain
            else {
                "prompt": "bold #5fd7ff",
                "bottom-toolbar": "bg:#262b36 #aab2c0",
                "completion-menu.completion": "bg:#262b36 #d7dce5",
                "completion-menu.completion.current": "bg:#5f5fff #ffffff bold",
            }
        )
        self.prompt_session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path())),
            completer=FuzzyWordCompleter(SLASH_COMMANDS, WORD=True),
            complete_while_typing=False,
            key_bindings=bindings,
            style=prompt_style,
            multiline=False,
            enable_open_in_editor=True,
            mouse_support=False,
        )

    def set_context(self, session_name: str | None, message_count: int) -> None:
        self._session_name = session_name
        self._message_count = message_count

    def banner(self, *, reused: bool, managed: bool, voice_ready: bool = False) -> None:
        state = "reused" if reused else "started"
        ownership = "managed" if managed else "external"
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            Text("● Ready", style="bold green"),
            Text(f"{state} · {ownership}", style="dim"),
        )
        grid.add_row(
            Text(short_model(self.settings.model), style="bold cyan"),
            Text(self.settings.base_url, style="blue"),
        )
        grid.add_row(
            Text(f"Thinking: {'ON' if self.settings.think else 'OFF'}", style="yellow"),
            Text(
                f"temp {self.settings.temperature:g} · top-p {self.settings.top_p:g} · "
                f"max {self.settings.max_tokens}",
                style="dim",
            ),
        )
        grid.add_row(
            Text(
                "Voice: ready" if voice_ready else "Voice: unavailable",
                style="green" if voice_ready else "yellow",
            ),
            Text(
                "Whisper Large v3 Turbo" if voice_ready else "install/setup required",
                style="dim",
            ),
        )
        self.console.print()
        self.console.print(
            Panel(
                grid,
                title="[bold]mlx-chat[/bold] · local Apple Silicon inference",
                subtitle="[dim]Local by default · /web is explicitly opt-in[/dim]",
                border_style="bright_blue",
                box=box.ROUNDED,
            )
        )
        self.console.print(
            "[dim]Type [bold]/help[/bold] for commands · Esc+Enter adds a line · "
            "Ctrl-C cancels · Ctrl-D exits[/dim]"
        )

    def _toolbar(self) -> HTML:
        if self.settings.concise:
            mode = " CONCISE "
        elif self.settings.think:
            mode = " THINK "
        else:
            mode = " DIRECT "
        session = self._session_name or "unsaved"
        return HTML(
            f" <b>{mode}</b>  {html_escape(short_model(self.settings.model, 26))}  "
            f"session: {html_escape(session)}  turns: {self._message_count // 2} "
        )

    def prompt(self) -> str:
        return self.prompt_session.prompt(
            HTML("<prompt>you ❯ </prompt>"),
            bottom_toolbar=self._toolbar,
            prompt_continuation=lambda width, _line, _wrap: " " * max(0, width - 2) + "· ",
        )

    def _notice(self, symbol: str, style: str, message: str) -> None:
        text = Text()
        text.append(f"{symbol} ", style=style)
        text.append(message)
        self.console.print(text)

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
        return Panel(
            content,
            title=title,
            title_align="left",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
            height=height,
        )

    def _live_group(self, state: RenderState) -> Group:
        result = state.result
        elapsed = time.monotonic() - state.started
        terminal_height = max(1, self.console.size.height)
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
            self.console.print(
                Panel(
                    Text(sanitize_terminal_text(result.reasoning), style="dim"),
                    title="[yellow]◆ Reasoning[/yellow]",
                    title_align="left",
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
        elif think:
            self.warning("The model returned no separate reasoning channel for this turn.")

        answer = sanitize_terminal_text(result.answer)
        if answer:
            renderable = Markdown(answer, code_theme="monokai", hyperlinks=True)
        else:
            renderable = Text("No final answer was returned.", style="italic dim")
        self.console.print(
            Panel(
                renderable,
                title="[bright_cyan]◆ Answer[/bright_cyan]",
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
        self.console.print(Text("  " + " · ".join(metrics), style="dim"))

    def help(self) -> None:
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
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
            table.add_row(command, description)
        self.console.print(
            Panel(table, title="mlx-chat commands", border_style="bright_blue", box=box.ROUNDED)
        )
        self.console.print(
            "[dim]Tip: start a message with // to send a literal /. "
            "Esc+Enter inserts a newline · F1 listens · F2 stops listening · "
            "F3 speaks · F4 stops speech · "
            "Ctrl-C stops audio.[/dim]"
        )


def is_plain_terminal() -> bool:
    return "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb"
