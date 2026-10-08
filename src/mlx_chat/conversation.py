"""Scrollable conversation viewport above the existing input design."""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from weakref import WeakValueDictionary

from prompt_toolkit import ANSI
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.containers import ScrollOffsets
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.renderer import Renderer
from prompt_toolkit.utils import get_cwidth
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.style import Style
from rich.text import Text

from .theme import ACCENT, CANVAS, MUTED, POPUP, SELECTION

CONVERSATION_BACKGROUND = CANVAS
POPUP_BACKGROUND = POPUP


def edge_style(style, row, offset, height, total):
    """Blend clipped viewport edges into their background, retaining text metadata."""
    distance = 3
    if offset > 0:
        distance = min(distance, row - offset)
    if offset + height < total:
        distance = min(distance, offset + height - 1 - row)
    opacity = (0.25, 0.5, 0.8)[max(0, distance)] if distance < 3 else 1.0
    foreground = style.color.get_truecolor() if style.color else (205, 214, 244)
    bgcolor = style.bgcolor or Style(bgcolor=CONVERSATION_BACKGROUND).bgcolor
    background = bgcolor.get_truecolor()
    blended = tuple(
        round(bg + (fg - bg) * opacity) for fg, bg in zip(foreground, background, strict=False)
    )
    color = "#{:02x}{:02x}{:02x}".format(*blended)
    return style + Style(color=color, bgcolor=bgcolor)


class ConversationRenderer(Renderer):
    """Preserve the terminal screen and diff history across prompt/worker runs."""

    def __init__(self, previous):
        self.persistent = False
        super().__init__(
            previous.style,
            previous.output,
            full_screen=True,
            mouse_support=previous.mouse_support,
            cpr_not_supported_callback=previous.cpr_not_supported_callback,
        )
        self.persistent = True

    def reset(self, _scroll=False, leave_alternate_screen=True):
        if not self.persistent:
            super().reset(_scroll=_scroll, leave_alternate_screen=leave_alternate_screen)

    def erase(self, leave_alternate_screen=True):
        if not self.persistent:
            super().erase(leave_alternate_screen=leave_alternate_screen)

    def render(self, app, layout, is_done=False):
        if not (self.persistent and is_done):
            super().render(app, layout, is_done=is_done)

    def request_absolute_cursor_position(self):
        if self.persistent and self._in_alternate_screen:
            return
        super().request_absolute_cursor_position()

    def finish(self):
        self.persistent = False
        self.erase()


class ConversationControl(FormattedTextControl):
    """Keep the invisible cursor in the viewport and handle wheels on empty cells."""

    def __init__(self, view):
        self.view = view
        super().__init__(view.fragments, show_cursor=False)

    def create_content(self, width, height):
        self.view._viewport_rows = height
        self.view.prepare_lines()
        return UIContent(
            get_line=self.view.line_fragments,
            line_count=max(1, self.view.line_count),
            cursor_position=Point(0, self.view.scroll_position(self.view.window)),
            show_cursor=False,
        )

    def mouse_handler(self, mouse_event):
        self.view.mouse(mouse_event)
        return None


class ConversationView:
    def __init__(self, ui):
        self.ui = ui
        self.popup_items = None
        self.popup_preview = False
        self._preview_version = 0
        self._preview_text = None
        self._output_capture = ContextVar("command_output", default=None)
        self.popup_title = ""
        self.popup_offset = 0
        self.popup_waiting = False
        self.thinking_traces = WeakValueDictionary()
        self.trace_item_indices = {}
        self._item_spans = []
        self.selection = None
        self.selection_dragging = False
        self._popup_lines = []
        self.items = []
        self.pending = None
        self.follow = True
        self.offset = 0
        self.line_count = 0
        self.busy = False
        self.allow_input = False
        self.submitted_inputs = deque()
        self.draft_text = ""
        self.audio_key = None
        self.cancelled = threading.Event()
        self._revision = 0
        self._cache_key = None
        self._lines = []
        self._history_width = None
        self._history_line_count = 0
        self._rendered_item_count = 0
        self.window = Window(
            ConversationControl(self),
            wrap_lines=False,
            scroll_offsets=ScrollOffsets(),
            get_vertical_scroll=self.scroll_position,
            style=f"bg:{CONVERSATION_BACKGROUND}",
        )
        session = ui.prompt_session
        composer = session.layout.container
        composer.height = self.composer_height
        composer.width = self.composer_width
        self.composer = composer
        centered_composer = VSplit(
            [Window(width=Dimension(weight=1)), composer, Window(width=Dimension(weight=1))]
        )
        self.header = Window(
            FormattedTextControl(self.header_fragments, show_cursor=False),
            height=self.header_height,
            wrap_lines=False,
        )
        self.top_separator = HSplit(
            [
                Window(height=2),
                Window(
                    FormattedTextControl(self.hint_fragments, show_cursor=False),
                    height=1,
                    wrap_lines=False,
                ),
            ]
        )
        self.bottom_separator = Window(height=3)
        frame_style = f"bg:{CONVERSATION_BACKGROUND}"
        self.frame = HSplit(
            [
                Window(height=1, style=frame_style),
                VSplit(
                    [
                        Window(width=2, style=frame_style),
                        self.window,
                        Window(width=2, style=frame_style),
                    ]
                ),
                Window(height=1, style=frame_style),
            ]
        )
        # Menus need the whole screen's space, rather than the composer's
        # three-row allocation. Cursor-anchored floats then open above it.
        input_floats = ui._input_float_container
        self.layout = HSplit(
            [
                self.header,
                self.top_separator,
                self.frame,
                self.bottom_separator,
                centered_composer,
                Window(height=2),
            ],
            style=f"bg:{CANVAS}",
        )
        session.layout.container = FloatContainer(self.layout, floats=list(input_floats.floats))
        self.completion_floats = [item for item in session.layout.container.floats if item.ycursor]
        for item in self.completion_floats:
            item.ycursor = False
            item.bottom = 7
        popup_window = Window(
            FormattedTextControl(
                self.popup_fragments,
                get_cursor_position=lambda: Point(0, self.popup_offset),
            ),
            wrap_lines=False,
            style=f"bg:{POPUP_BACKGROUND}",
            get_vertical_scroll=lambda window: self.popup_offset,
        )
        session.layout.container.floats.append(
            Float(
                ConditionalContainer(
                    HSplit(
                        [
                            Window(
                                FormattedTextControl(self.popup_heading),
                                height=3,
                                style=f"bg:{POPUP_BACKGROUND}",
                            ),
                            popup_window,
                            Window(
                                FormattedTextControl(self.popup_footer),
                                height=self.popup_footer_height,
                                style=f"bg:{POPUP_BACKGROUND}",
                            ),
                        ]
                    ),
                    Condition(lambda: self.popup_items is not None),
                ),
                width=lambda: max(1, min(100, session.app.output.get_size().columns - 4)),
                height=self.popup_height,
                z_index=10,
                allow_cover_cursor=True,
            )
        )
        input_floats.floats = []
        session.reserve_space_for_menu = 0
        session._output = session.app.output
        session.app.full_screen = True
        session.app.renderer = ConversationRenderer(session.app.renderer)
        session.refresh_interval = 0.1
        session.app.refresh_interval = 0.1
        session.app.min_redraw_interval = 1 / 30
        session.mouse_support = True
        session.default_buffer.read_only = Condition(
            lambda: (self.busy and not self.allow_input) or self.popup_waiting
        )
        bindings = session.key_bindings

        @bindings.add("enter", filter=Condition(lambda: self.popup_waiting), eager=True)
        @bindings.add("escape", filter=Condition(lambda: self.popup_waiting), eager=True)
        @bindings.add("c-c", filter=Condition(lambda: self.popup_waiting), eager=True)
        def dismiss_popup(event):
            event.app.exit(result="")

        @bindings.add("pageup", filter=Condition(lambda: self.popup_waiting))
        @bindings.add("up", filter=Condition(lambda: self.popup_waiting))
        def popup_up(event):
            self.popup_offset = max(0, self.popup_offset - 3)

        @bindings.add("pagedown", filter=Condition(lambda: self.popup_waiting))
        @bindings.add("down", filter=Condition(lambda: self.popup_waiting))
        def popup_down(event):
            self.popup_offset = min(
                max(0, self.popup_line_count - self.popup_body_height()), self.popup_offset + 3
            )

        @bindings.add("pageup")
        def page_up(event):
            self.scroll(-self.viewport_height())

        @bindings.add("pagedown")
        def page_down(event):
            self.scroll(self.viewport_height())

        @bindings.add("c-end")
        def end(event):
            self.follow = True
            event.app.invalidate()

        @bindings.add("enter", filter=Condition(lambda: self.busy and self.allow_input))
        def submit_during_generation(event):
            text = event.current_buffer.text
            if text.strip():
                self.submitted_inputs.append(text)
                event.current_buffer.reset()
                self.cancelled.set()

        @bindings.add(
            "escape", "enter", filter=Condition(lambda: self.busy and not self.allow_input)
        )
        @bindings.add("enter", filter=Condition(lambda: self.busy and not self.allow_input))
        @bindings.add("f1", filter=Condition(lambda: self.busy))
        @bindings.add("f2", filter=Condition(lambda: self.busy))
        def ignore(event):
            pass

        @bindings.add("c-c", filter=Condition(lambda: self.busy))
        @bindings.add("escape", filter=Condition(lambda: self.busy))
        @bindings.add("c-d", filter=Condition(lambda: self.busy))
        def cancel(event):
            self.cancelled.set()

        @bindings.add("f3", filter=Condition(lambda: self.busy and self.audio_key is not None))
        def stop_audio(event):
            self.cancelled.set()

        session.default_buffer.on_text_changed += self.on_input_changed

        self._original_print = ui.console.print
        self._original_status = ui.console.status
        ui.console.print = self.print
        ui.console.status = self.status
        session.placeholder = lambda: [("class:input-placeholder", self.composer_placeholder())]

    def dismiss_preview(self):
        self._preview_version += 1
        self._preview_text = None
        if self.popup_preview:
            self.popup_preview = False
            self.popup_items = None
            self.ui.prompt_session.app.invalidate()

    def on_input_changed(self, buffer):
        app = self.ui.prompt_session.app
        if not app.is_running or self.popup_waiting or (self.busy and not self.allow_input):
            return
        from .ui import SLASH_COMMANDS

        command = buffer.text.strip().lower()
        if command not in SLASH_COMMANDS or command in {"/voice", "/web", "/retry"}:
            self.dismiss_preview()
            return
        if self._preview_text == command:
            return
        self.dismiss_preview()
        self._preview_text = command
        version = self._preview_version
        self.popup_preview = True
        self.popup_title = command
        self.popup_offset = 0
        self.popup_items = [Text("Preparing…", style="dim")]

        def apply(items):
            if version != self._preview_version or not self.popup_preview:
                return
            self.popup_items = items
            self._revision += 1
            app.invalidate()

        def collect():
            items = []
            token = self._output_capture.set(items)
            try:
                self.ui.command_previewer(command)
            except Exception as exc:
                items.append(Text(str(exc), style="dim"))
            finally:
                self._output_capture.reset(token)
            if app.loop is not None and app.is_running:
                with suppress(RuntimeError):
                    app.loop.call_soon_threadsafe(apply, items)

        threading.Thread(target=collect, daemon=True).start()
        app.invalidate()

    def composer_placeholder(self):
        if self.busy and self.allow_input:
            return "Add a message… Return to redirect"
        if self.busy:
            return "Working…"
        return "Write a message… / for commands"

    @contextmanager
    def popup(self, title):
        self.dismiss_preview()
        self.popup_items = []
        self.popup_title = title
        self.popup_offset = 0
        self.selection = None
        try:
            yield
            if self.popup_items:
                self.popup_waiting = True
                self.ui.prompt_session.app.run()
        finally:
            self.popup_waiting = False
            self.popup_items = None
            self.ui.prompt_session.app.invalidate()

    def popup_width(self):
        return max(1, min(100, self.ui.prompt_session.app.output.get_size().columns - 4))

    def popup_heading(self):
        width = self.popup_width()
        names = {"/help": "Commands", "/system": "System prompt", "/speak": "Read aloud"}
        title = names.get(self.popup_title, self.popup_title.lstrip("/").capitalize())
        left = f"  {title}"
        right = "esc quit  " if self.popup_preview else "esc  "
        middle = (
            "// literal / · Esc+Enter newline · F1 listen · F2 speak · F3 stop"
            if self.popup_title == "/help"
            else ""
        )
        available = max(0, width - get_cwidth(left) - len(right) - 4)
        if len(middle) > available:
            middle = middle[: max(0, available - 1)] + "…" if available else ""
        gap = max(0, width - get_cwidth(left) - len(right) - len(middle))
        return [
            ("", " " * width + "\n"),
            ("bold fg:#e5e9f0", left),
            ("", " " * (gap // 2)),
            ("fg:#a5b1c5", middle),
            ("", " " * (gap - gap // 2)),
            ("fg:#a5b1c5", right),
            ("", "\n" + " " * width),
        ]

    def popup_info(self):
        return [
            item
            for item in (self.popup_items or [])
            if isinstance(item, Text) and item.style == "dim"
        ]

    def popup_footer_height(self):
        return 3 if self.popup_info() and self.popup_title != "/help" else 1

    def popup_footer(self):
        info = self.popup_info()
        if not info or self.popup_title == "/help":
            return [("", " " * self.popup_width())]
        text = Text("\n".join(item.plain.strip() for item in info), style="#b3bfd0")
        return self.popup_render(Padding(text, (1, 2, 0, 2)))

    def popup_height(self):
        self.popup_fragments()
        return max(
            1,
            min(
                self.popup_line_count + 3 + self.popup_footer_height(),
                32,
                self.ui.prompt_session.app.output.get_size().rows - 6,
            ),
        )

    def popup_body_height(self):
        return max(1, self.popup_height() - 3 - self.popup_footer_height())

    def selection_mouse(self, event, *, popup=False):
        if event.button != MouseButton.LEFT:
            return
        point = (event.position.y, event.position.x)
        if event.event_type == MouseEventType.MOUSE_DOWN:
            self.selection = (popup, point, point)
            self.selection_dragging = True
        elif event.event_type == MouseEventType.MOUSE_MOVE and self.selection_dragging:
            self.selection = (popup, self.selection[1], point)
        elif event.event_type == MouseEventType.MOUSE_UP and self.selection_dragging:
            self.selection = (popup, self.selection[1], point)
            self.selection_dragging = False
            if self.selection[1] == point and popup and self.select_help_command(point):
                self.selection = None
                return
            if self.selection[1] == point and not popup and self.toggle_trace(point):
                self.selection = None
                return
            text = self.selected_text()
            if text:
                try:
                    subprocess.run(["pbcopy"], input=text, text=True, check=True, timeout=2)
                except (OSError, subprocess.SubprocessError):
                    self.ui.warning("Could not copy the selected text to the clipboard.")
        self.ui.prompt_session.app.invalidate()

    def select_help_command(self, point):
        if self.popup_title != "/help" or not (self.popup_waiting or self.popup_preview):
            return False
        row, column = point
        if not 0 <= row < len(self._popup_lines):
            return False
        x = 0
        for segment in self._popup_lines[row]:
            width = get_cwidth(segment.text)
            command = segment.style.meta.get("help_command") if segment.style else None
            if x <= column < x + width and command:
                if self.popup_preview:
                    from prompt_toolkit.document import Document

                    self.dismiss_preview()
                    self.ui.prompt_session.default_buffer.document = Document(command, len(command))
                else:
                    self.draft_text = command
                    self.ui.prompt_session.app.exit(result="")
                return True
            x += width
        return False

    def toggle_trace(self, point):
        row, column = point
        if row >= len(self._lines):
            return False
        x = 0
        for segment in self._lines[row]:
            width = get_cwidth(segment.text)
            token = (segment.style.meta or {}).get("thinking_trace") if segment.style else None
            if x <= column < x + width and token in self.thinking_traces:
                trace = self.thinking_traces[token]
                trace.expanded = not trace.expanded
                index = self.trace_item_indices.get(token)
                if index is not None and index < len(self._item_spans):
                    start, end = self._item_spans[index]
                    lines = self.ui.console.render_lines(
                        self.items[index],
                        self.ui.console.options.update(width=self._history_width),
                        pad=False,
                    )
                    self._lines[start:end] = lines
                    delta = len(lines) - (end - start)
                    self._item_spans[index] = (start, start + len(lines))
                    for later in range(index + 1, len(self._item_spans)):
                        left, right = self._item_spans[later]
                        self._item_spans[later] = (left + delta, right + delta)
                    self._history_line_count += delta
                self.selection = None
                self._revision += 1
                self.ui.prompt_session.app.invalidate()
                return True
            x += width
        return False

    def selected_text(self):
        if self.selection is None:
            return ""
        popup, start, end = self.selection
        start, end = sorted((start, end))
        lines = self._popup_lines if popup else self._lines
        selected = []
        for row in range(start[0], min(end[0] + 1, len(lines))):
            value = "".join(segment.text for segment in lines[row])
            left = start[1] if row == start[0] else 0
            right = end[1] if row == end[0] else float("inf")
            column = 0
            chars = []
            for char in value:
                width = get_cwidth(char)
                if left <= column < right:
                    chars.append(char)
                column += width
            selected.append("".join(chars).rstrip())
        return "\n".join(selected)

    def selection_fragments(self, text, style, row, column, *, popup=False):
        result = []
        bounds = (
            sorted(self.selection[1:]) if self.selection and self.selection[0] == popup else None
        )
        if not bounds or row < bounds[0][0] or row > bounds[1][0]:
            formatted = to_formatted_text(ANSI(style.render(text, color_system="truecolor")))
            return [
                (part[0], part[1], self.popup_mouse if popup else self.mouse) for part in formatted
            ], column + get_cwidth(text)
        for char in text:
            selected = bounds and bounds[0] <= (row, column) < bounds[1]
            char_style = style + Style(color="#ffffff", bgcolor=SELECTION) if selected else style
            formatted = to_formatted_text(ANSI(char_style.render(char, color_system="truecolor")))
            result.extend(
                (part[0], part[1], self.popup_mouse if popup else self.mouse) for part in formatted
            )
            column += get_cwidth(char)
        return result, column

    def popup_mouse(self, event):
        self.selection_mouse(event, popup=True)
        if event.event_type == MouseEventType.SCROLL_UP:
            self.popup_offset = max(0, self.popup_offset - 3)
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            self.popup_offset = min(
                max(0, self.popup_line_count - self.popup_body_height()), self.popup_offset + 3
            )
        self.ui.prompt_session.app.invalidate()

    def popup_render(self, content, *, body=False):
        lines = self.ui.console.render_lines(
            content, self.ui.console.options.update(width=self.popup_width()), pad=True
        )
        if body:
            self.popup_line_count = len(lines)
            self._popup_lines = lines
        fragments = []
        for number, line in enumerate(lines):
            column = 0
            for segment in line:
                style = (segment.style or Style()) + Style(bgcolor=POPUP_BACKGROUND)
                if body:
                    parts, column = self.selection_fragments(
                        segment.text, style, number, column, popup=True
                    )
                    fragments.extend(parts)
                else:
                    fragments.extend(
                        to_formatted_text(
                            ANSI(style.render(segment.text, color_system="truecolor"))
                        )
                    )
            if number < len(lines) - 1:
                fragments.append(("", "\n", self.popup_mouse))
        return fragments

    def popup_fragments(self):
        info = self.popup_info()
        content = Group(
            *(item for item in (self.popup_items or []) if item not in info),
            *([self.pending] if self.pending is not None and not self.popup_preview else []),
        )
        return self.popup_render(Padding(content, (0, 2)), body=True)

    def hint_fragments(self):
        width = self.ui.prompt_session.app.output.get_size().columns
        text = self.ui.conversation_hint()
        text = text[: max(0, width)]
        return [("class:hint", " " * max(0, (width - get_cwidth(text)) // 2) + text)]

    def header_height(self):
        size = self.ui.prompt_session.app.output.get_size()
        lines = self.ui.console.render_lines(
            self.ui.info_panel(),
            self.ui.console.options.update(width=max(1, size.columns)),
            pad=False,
        )
        return min(len(lines), max(3, size.rows // 3))

    def header_fragments(self):
        width = self.ui.prompt_session.app.output.get_size().columns
        lines = self.ui.console.render_lines(
            self.ui.info_panel(), self.ui.console.options.update(width=max(1, width)), pad=False
        )
        fragments = []
        for number, line in enumerate(lines):
            for segment in line:
                style = segment.style or Style()
                fragments.extend(
                    to_formatted_text(ANSI(style.render(segment.text, color_system="truecolor")))
                )
            if number < len(lines) - 1:
                fragments.append(("", "\n"))
        return fragments

    def composer_width(self):
        columns = self.ui.prompt_session.app.output.get_size().columns
        return max(1, min(104, columns - 4, max(36, int(columns * 0.88))))

    def composer_height(self):
        size = self.ui.prompt_session.app.output.get_size()
        height = min(max(6, size.rows // 2), max(3, self.ui._composer_input_rows()) + 3)
        for item in getattr(self, "completion_floats", []):
            item.bottom = height + 2
        return Dimension.exact(height)

    def normal_print(self, *objects, **kwargs):
        if self._output_capture.get() is not None:
            return self.print(*objects, **kwargs)
        return self._original_print(*objects, **kwargs)

    def close(self):
        self.dismiss_preview()
        self.ui.prompt_session.app.renderer.finish()
        self.ui.console.print = self.normal_print
        self.ui.console.status = self._original_status
        self.ui.exit_summary()

    @contextmanager
    def status(self, value, *, user=False, **kwargs):
        spinner = Spinner("dots", text=Text(str(value)), style=ACCENT if user else MUTED)
        panel = (
            self.ui.user_line(str(value))
            if user
            else self.ui.chat_panel(
                spinner,
                title="◆ System",
                title_align="left",
                border_style="bright_cyan",
                padding=(0, 1),
            )
        )
        panel.renderable = self.ui.message_content(spinner, user=user)
        self.update(panel)
        try:
            yield
        finally:
            self.update(None)

    def print(self, *objects, **kwargs):
        renderables = [
            self.ui.console.render_str(item) if isinstance(item, str) else item for item in objects
        ]
        content = Group(*renderables)
        capture = self._output_capture.get()
        target = (
            capture
            if capture is not None
            else self.popup_items
            if self.popup_items is not None and not self.popup_preview
            else self.items
        )
        if target is self.items:
            for item in renderables:
                if id(item) in self.thinking_traces:
                    self.trace_item_indices[id(item)] = len(target)
        target.append(
            content
            if all(isinstance(item, Panel) for item in renderables)
            else self.ui.chat_panel(content)
        )
        self._revision += 1
        self.ui.prompt_session.app.invalidate()

    def print_info(self, content):
        """Append subdued metadata without turning it into a response bubble."""
        capture = self._output_capture.get()
        target = (
            capture
            if capture is not None
            else self.popup_items
            if self.popup_items is not None and not self.popup_preview
            else self.items
        )
        target.append(content)
        self._revision += 1
        self.ui.prompt_session.app.invalidate()

    def update(self, renderable):
        self.pending = renderable
        self._revision += 1
        self.ui.prompt_session.app.invalidate()

    def viewport_height(self):
        if getattr(self, "_viewport_rows", None) is not None:
            return max(1, self._viewport_rows)
        info = self.window.render_info
        return max(1, info.window_height if info else self.ui.console.height - 5)

    def scroll_position(self, window):
        bottom = max(0, self.line_count - self.viewport_height())
        self.offset = bottom if self.follow else min(bottom, self.offset)
        return self.offset

    def scroll(self, delta):
        if self.follow:
            self.offset = max(0, self.line_count - self.viewport_height())
        self.offset = max(
            0, min(max(0, self.line_count - self.viewport_height()), self.offset + delta)
        )
        self.follow = delta > 0 and self.offset >= self.line_count - self.viewport_height()
        self.ui.prompt_session.app.invalidate()

    def mouse(self, event):
        self.selection_mouse(event)
        if event.event_type == MouseEventType.SCROLL_UP:
            self.scroll(-3)
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll(3)

    def prepare_lines(self):
        width = max(1, self.ui.prompt_session.app.output.get_size().columns - 4)
        animated = self.pending is not None and self.popup_items is None
        key = (
            width,
            self._revision,
            int(time.monotonic() * 10) if animated else 0,
            self.popup_items is not None,
        )
        if key == self._cache_key:
            return
        if width != self._history_width or len(self.items) < self._rendered_item_count:
            self._lines = []
            self._history_line_count = 0
            self._rendered_item_count = 0
            self._item_spans = []
            self._history_width = width
        del self._lines[self._history_line_count :]
        options = self.ui.console.options.update(width=width)
        for item in self.items[self._rendered_item_count :]:
            start = len(self._lines)
            self._lines.extend(self.ui.console.render_lines(item, options, pad=False))
            self._item_spans.append((start, len(self._lines)))
        self._rendered_item_count = len(self.items)
        self._history_line_count = len(self._lines)
        if animated:
            self._lines.extend(self.ui.console.render_lines(self.pending, options, pad=False))
        self.line_count = len(self._lines)
        self._cache_key = key

    def line_fragments(self, number):
        if number >= self.line_count:
            return []
        offset = self.scroll_position(self.window)
        height = self.viewport_height()
        result = []
        line = self._lines[number]

        column = 0
        for segment in line:
            style = edge_style(segment.style or Style(), number, offset, height, self.line_count)
            if self.popup_items is not None:
                foreground = style.color.get_truecolor()
                background = style.bgcolor.get_truecolor()
                dim_fg = "#{:02x}{:02x}{:02x}".format(*(round(c * 0.35) for c in foreground))
                dim_bg = "#{:02x}{:02x}{:02x}".format(*(round(c * 0.65) for c in background))
                style += Style(color=dim_fg, bgcolor=dim_bg)
            parts, column = self.selection_fragments(segment.text, style, number, column)
            result.extend(parts)
        return result

    def fragments(self):
        """Expose only visible rows; the control requests history lines lazily."""
        self.prepare_lines()
        offset = self.scroll_position(self.window)
        end = min(self.line_count, offset + self.viewport_height())
        result = []
        for number in range(offset, end):
            result.extend(self.line_fragments(number))
            if number + 1 < end:
                result.append(("", "\n", self.mouse))
        return result

    def run_native(self, operation):
        """Keep native inference on the caller thread; render the UI separately."""
        self.busy = True
        self.cancelled.clear()
        app = self.ui.prompt_session.app
        ready = threading.Event()
        errors = []

        def display():
            try:
                app.run(pre_run=ready.set, handle_sigint=False)
            except BaseException as exc:
                errors.append(exc)
                ready.set()

        screen = threading.Thread(target=display, daemon=True)
        screen.start()
        ready.wait()
        try:
            if errors:
                raise errors[0]
            return operation()
        finally:
            if app.loop is not None:
                app.loop.call_soon_threadsafe(app.exit, "")
            screen.join()
            self.busy = False
            self.update(None)

    def generate(self, generate, preview, *, allow_input=False):
        self.busy = True
        self.allow_input = allow_input
        self.cancelled.clear()
        self.follow = True
        self.update(preview(None))
        outcome = []
        app = self.ui.prompt_session.app

        last_preview = 0.0

        def callback(event, result):
            nonlocal last_preview
            if self.cancelled.is_set():
                raise KeyboardInterrupt
            now = time.monotonic()
            if now - last_preview >= 1 / 30:
                self.update(preview(result))
                last_preview = now

        def worker():
            try:
                outcome.append((True, generate(callback)))
            except BaseException as exc:
                outcome.append((False, exc))
            finally:
                app.loop.call_soon_threadsafe(app.exit, "")

        thread = threading.Thread(target=worker, daemon=True)
        try:
            self.ui.prompt_session.prompt(
                "",
                default=self.draft_text if allow_input else "",
                bottom_toolbar=None,
                pre_run=thread.start,
            )
        finally:
            if allow_input:
                self.draft_text = self.ui.prompt_session.default_buffer.text
            self.cancelled.set()
            thread.join()
            self.busy = False
            self.allow_input = False
            self.update(None)
        if not outcome[0][0]:
            raise outcome[0][1]
        return outcome[0][1]
