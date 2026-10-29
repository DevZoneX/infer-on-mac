"""Response rendering."""

from __future__ import annotations

import time
from collections.abc import Callable

from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.style import Style as RichStyle
from rich.syntax import Syntax
from rich.text import Text

from .client import GenerationResult, StreamEvent
from .resources import resource_snapshot
from .terminal import sanitize_terminal_text
from .theme import ACCENT, DANGER, MUTED, SUCCESS, WARNING, gradient

TOOL_LABELS = {
    "list_files": "List",
    "read_file": "Read",
    "search": "Search",
    "edit_file": "Edit",
    "write_file": "Write",
    "run_command": "Run",
    "find_files": "Find",
    "outline": "Outline",
    "update_plan": "Plan",
    "web_search": "Web search",
    "fetch_url": "Fetch",
}
DETAIL_LINES = 60


class ThinkingTrace(Panel):
    """A compact label whose trace can be expanded with the mouse."""

    def __init__(self, text=""):
        super().__init__(Text("Thinking"))
        self.text = text
        self.label = "Thinking"
        self.expanded = False
        self.loading = False
        self.spinner = Spinner("dots")

    def __rich_console__(self, console, options):
        label = Text()
        if self.loading:
            label.append(self.spinner.render(time.monotonic()))
        else:
            label.append("◆")
        label.append(f" {self.label}")
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


def tool_description(item) -> str:
    """The argument a person scans for: the command, path, or pattern."""
    from .tools import ToolCall, describe_call

    raw = item.get("arguments", "")
    call = ToolCall.from_api(
        {"id": "", "function": {"name": item.get("name", ""), "arguments": raw}}, 0
    )
    return describe_call(call) if not call.error else raw[:160]


def _clip_lines(text: str, limit: int) -> str:
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    return "\n".join([*lines[:limit], f"… {len(lines) - limit} more lines"])


class ToolTrace(Panel):
    """One tool call as a single line; click to expand its output.

    Edits open with their diff showing, since that is what the person reviews.
    """

    def __init__(self, item):
        super().__init__(Text(""))
        self.item = item
        self.expanded = item.get("ok", True) and (
            bool(item.get("diff")) or item.get("name") == "update_plan"
        )

    def plan(self):
        import json

        try:
            steps = json.loads(self.item.get("arguments") or "{}").get("steps")
        except (ValueError, AttributeError):
            return None
        return (
            steps if isinstance(steps, list) and all(isinstance(i, dict) for i in steps) else None
        )

    def details(self):
        if self.item.get("name") == "update_plan" and self.item.get("ok", True):
            plan = self.plan()
            return plan_checklist(plan) if plan else None
        if self.item.get("diff"):
            return Syntax(
                _clip_lines(self.item["diff"].rstrip(), DETAIL_LINES),
                "diff",
                theme="monokai",
                background_color="default",
                word_wrap=True,
            )
        content = sanitize_terminal_text(str(self.item.get("content", ""))).rstrip()
        return Text(_clip_lines(content, DETAIL_LINES), style="dim") if content else None

    def __rich_console__(self, console, options):
        item = self.item
        approval = item.get("approval")
        if approval == "denied":
            icon, color = "⊘", WARNING
        elif item.get("ok", True):
            icon, color = "✓", SUCCESS
        else:
            icon, color = "✗", DANGER
        meta = RichStyle(meta={"thinking_trace": id(self)})
        label = Text(no_wrap=True, overflow="ellipsis")
        label.append(f"{icon} ", style=color)
        label.append(TOOL_LABELS.get(item.get("name"), str(item.get("name", "tool"))), style="bold")
        description = sanitize_terminal_text(tool_description(item))
        if description:
            label.append(f"  {description}", style=ACCENT)
        summary = sanitize_terminal_text(str(item.get("summary", "")))
        if summary and summary != description:
            label.append(f"  ·  {summary}", style=MUTED)
        if approval in {"approved", "always"}:
            label.append("  ·  approved", style=MUTED)
        details = self.details()
        if details is not None:
            label.append("  ▾" if self.expanded else "  ▸", style=MUTED)
        label.stylize(meta)
        content = [label]
        if self.expanded and details is not None:
            content.append(Padding(details, (0, 0, 0, 2)))
        yield Padding(Group(*content), (1, 2, 0, 2))


PLAN_STYLES = {"done": ("●", SUCCESS), "in_progress": ("◐", ACCENT), "pending": ("○", MUTED)}


def plan_checklist(plan) -> Text:
    text = Text()
    for index, item in enumerate(plan):
        mark, color = PLAN_STYLES.get(item.get("status"), PLAN_STYLES["pending"])
        if index:
            text.append("\n")
        text.append(f"{mark} ", style=color)
        style = (
            "dim strike"
            if item.get("status") == "done"
            else ("bold" if item.get("status") == "in_progress" else "")
        )
        text.append(sanitize_terminal_text(str(item.get("step", ""))), style=style)
    return text


def plan_prompt_card():
    keys = Text()
    keys.append("◆ ", style=WARNING)
    keys.append("Run this plan?", style="bold")
    keys.append("   ↵ / y", style=f"bold {ACCENT}")
    keys.append(" run   ", style=MUTED)
    keys.append("n", style=f"bold {ACCENT}")
    keys.append(" keep for later   ", style=MUTED)
    keys.append("or type changes to revise it", style=MUTED)
    return Panel(keys, border_style=WARNING, box=box.ROUNDED, padding=(0, 1))


def approval_card(name: str, description: str, preview=None, *, always_label: str = ""):
    """The question shown while a tool call waits for the person."""
    title = Text()
    title.append("◆ ", style=WARNING)
    title.append(f"Allow {TOOL_LABELS.get(name, name)}?", style="bold")
    if description:
        title.append(f"  {sanitize_terminal_text(description)}", style=ACCENT)
    parts = [title]
    if preview is not None and preview.diff:
        parts.append(
            Padding(
                Syntax(
                    _clip_lines(preview.diff.rstrip(), DETAIL_LINES),
                    "diff",
                    theme="monokai",
                    background_color="default",
                    word_wrap=True,
                ),
                (1, 0, 0, 2),
            )
        )
    keys = Text()
    keys.append("↵ / y", style=f"bold {ACCENT}")
    keys.append(" allow once   ", style=MUTED)
    keys.append("a", style=f"bold {ACCENT}")
    keys.append(f" always allow {always_label or TOOL_LABELS.get(name, name)}   ", style=MUTED)
    keys.append("n / esc", style=f"bold {ACCENT}")
    keys.append(" decline   ", style=MUTED)
    keys.append("or type what to do instead", style=MUTED)
    parts.append(Padding(keys, (1, 0, 0, 2)))
    return Panel(
        Group(*parts),
        border_style=WARNING,
        box=box.ROUNDED,
        padding=(0, 1),
    )


class AnswerPanel(Panel):
    """Keep the response and its metadata together without a bottom spacer."""

    def __rich_console__(self, console, options):
        yield Padding(self.renderable, (1, 2, 0, 2))


SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class LivePreview:
    """A streaming preview: a body re-rendered only when it changes, and an animated footer."""

    def __init__(self, body, footer: Callable[[], Text]):
        self.body = body
        self.footer = footer

    def __rich_console__(self, console, options):
        yield self.body
        yield self.footer()


def live_status(result: GenerationResult, started: float, first: float | None) -> Text:
    """One animated line: what the model is doing, for how long, and how fast."""
    now = time.monotonic()
    if result.tool_calls:
        phase = "Preparing a tool call"
    elif result.answer:
        phase = "Writing"
    elif result.reasoning:
        phase = "Thinking"
    else:
        phase = "Reading the prompt"
    frame = SPINNER_FRAMES[int(now * 12) % len(SPINNER_FRAMES)]
    colors = [color for _character, color in gradient("x" * 12)]
    line = Text("\n    ")
    line.append(frame, style=colors[int(now * 6) % len(colors)])
    line.append(f" {phase}…", style=f"bold {ACCENT}")
    details = [f"{now - started:.1f}s"]
    if first is not None and now - first > 0.5:
        # About four characters per token; exact counts arrive with the final usage.
        tokens = (len(result.answer) + len(result.reasoning)) / 4
        details.append(f"~{tokens / (now - first):.0f} tok/s")
    details.append("esc to stop")
    line.append("   " + "  ·  ".join(details), style=MUTED)
    return line


class GenerationPresentation:
    def render_generation(
        self,
        generate: Callable[[Callable[[StreamEvent, GenerationResult], None]], GenerationResult],
        *,
        think: bool,
    ) -> GenerationResult:
        if getattr(self, "_conversation", None) is not None:
            trace = self.thinking_trace("")
            started = time.monotonic()
            first: list[float] = []

            def preview(result):
                result = result or GenerationResult()
                if not first and (result.answer or result.reasoning):
                    first.append(time.monotonic())
                panels = []
                if result.reasoning:
                    trace.text = result.reasoning
                    panels.append(trace)
                if result.answer:
                    panels.append(
                        self.chat_panel(
                            Markdown(
                                sanitize_terminal_text(result.answer),
                                code_theme="monokai",
                                hyperlinks=True,
                            ),
                            compact_bottom=True,
                        )
                    )
                return LivePreview(
                    Group(*panels),
                    lambda: live_status(result, started, first[0] if first else None),
                )

            result = self._conversation.generate(generate, preview, allow_input=True)
            self.final_generation(result, think=think)
            return result

        # Plain output is printed after generation completes.
        result = generate(lambda _event, _result: None)
        self.final_generation(result, think=think)
        return result

    def final_generation(
        self, result: GenerationResult, *, think: bool, historical: bool = False
    ) -> None:
        if result.reasoning:
            self.console.print(self.thinking_trace(result.reasoning))
        elif think and not result.tool_calls:
            self.warning("The model returned no separate reasoning channel for this turn.")

        answer = sanitize_terminal_text(result.answer)
        if result.tool_calls:
            # A tool step: its narration (if any) is shown; metrics come with the final answer.
            if answer.strip():
                self.console.print(
                    self.chat_panel(
                        Markdown(answer, code_theme="monokai", hyperlinks=True),
                        compact_bottom=True,
                    )
                )
            return
        if answer:
            renderable = Markdown(answer, code_theme="monokai", hyperlinks=True)
        else:
            renderable = Text("No final answer was returned.", style="italic dim")
        self.console.print(
            self.chat_panel(
                renderable,
                compact_bottom=True,
            )
        )

        if result.finish_reason == "length" and result.reasoning and not result.answer:
            self.warning(
                "The token limit was reached during reasoning. Try /max-tokens 8192 and retry."
            )
        metrics = [f"{result.elapsed:.1f}s"]
        if result.first_token_seconds is not None:
            metrics.append(f"first token {result.first_token_seconds:.1f}s")
        if result.completion_tokens is not None:
            metrics.append(f"{result.completion_tokens} tokens")
        if result.tokens_per_second is not None:
            metrics.append(f"{result.tokens_per_second:.1f} tok/s")
        if result.cached_tokens:
            metrics.append(f"{result.cached_tokens:,} cached")
        if result.finish_reason and result.finish_reason != "stop":
            metrics.append(f"finish: {result.finish_reason}")
        if getattr(self, "defer_metrics", False) and not historical:
            # The agent summary prints these together with the run totals.
            self.deferred_metrics = metrics
            return
        server = None if historical else resource_snapshot(self.settings.port)
        if server is not None:
            if server.cpu_percent is not None:
                metrics.append(f"CPU {server.cpu_percent:.1f}%")
            if server.memory_mib is not None:
                metrics.append(f"{server.memory_mib / 1024:.1f} GB in use")
        self.generation_info(" · ".join(metrics))

    def tool_trace(self, item):
        trace = ToolTrace(item)
        if getattr(self, "_conversation", None) is not None:
            self._conversation.thinking_traces[id(trace)] = trace
        else:
            trace.expanded = bool(item.get("diff"))
        return trace

    def tool_result(self, item) -> None:
        self.console.print(self.tool_trace(item))

    def agent_summary(self, run) -> None:
        """The final answer's metrics, then one line totalling the agent turn."""
        metrics = list(getattr(self, "deferred_metrics", None) or [])
        self.deferred_metrics = None
        server = resource_snapshot(self.settings.port)
        if server is not None:
            if server.cpu_percent is not None:
                metrics.append(f"CPU {server.cpu_percent:.1f}%")
            if server.memory_mib is not None:
                metrics.append(f"{server.memory_mib / 1024:.1f} GB in use")
        lines = [" · ".join(metrics)] if metrics else []
        if run.steps > 1 or run.tool_calls:
            parts = [
                f"{run.steps} steps",
                f"{run.tool_calls} tool {'call' if run.tool_calls == 1 else 'calls'}",
                f"{run.elapsed:.1f}s total",
            ]
            if run.prompt_tokens and run.cached_tokens:
                share = min(100, round(100 * run.cached_tokens / run.prompt_tokens))
                parts.append(f"{share}% of prompt tokens reused from cache")
            lines.append("agent · " + " · ".join(parts))
        if lines:
            self.generation_info("\n    ".join(lines))

    def generation_info(self, value: str) -> None:
        metadata = Text("    " + value, style="dim", no_wrap=True, overflow="ellipsis")
        conversation = getattr(self, "_conversation", None)
        if conversation is not None:
            if conversation.popup_items is None or conversation.popup_preview:
                metadata.end = "\n\n\n"
            conversation.print_info(metadata)
        else:
            self.console.print(metadata)
