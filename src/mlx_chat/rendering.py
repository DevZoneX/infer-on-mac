"""Response rendering."""

from __future__ import annotations

import time
from collections.abc import Callable

from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.style import Style as RichStyle
from rich.text import Text

from .client import GenerationResult, StreamEvent
from .resources import resource_snapshot
from .terminal import sanitize_terminal_text
from .theme import MUTED


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


class GenerationPresentation:
    def render_generation(
        self,
        generate: Callable[[Callable[[StreamEvent, GenerationResult], None]], GenerationResult],
        *,
        think: bool,
    ) -> GenerationResult:
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
                        compact_bottom=True,
                    )
                )
                return Group(*panels)

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
                compact_bottom=True,
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
        server = None if historical else resource_snapshot(self.settings.port)
        if server is not None:
            if server.cpu_percent is not None:
                metrics.append(f"CPU {server.cpu_percent:.1f}%")
            if server.memory_mib is not None:
                metrics.append(f"RSS {server.memory_mib:,.0f} MiB")
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
