"""Scrollable conversation viewport above the existing input design."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from prompt_toolkit import ANSI
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.containers import ScrollOffsets
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.renderer import Renderer
from prompt_toolkit.utils import get_cwidth
from rich.console import Group
from rich.panel import Panel
from rich.spinner import Spinner
from rich.style import Style
from rich.text import Text

CONVERSATION_BACKGROUND = "#181e2a"


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
        content = super().create_content(width, height)
        content.cursor_position = Point(0, self.view.scroll_position(self.view.window))
        return content

    def mouse_handler(self, mouse_event):
        if mouse_event.event_type in {MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN}:
            self.view.mouse(mouse_event)
            return None
        return super().mouse_handler(mouse_event)


class ConversationView:
    def __init__(self, ui):
        self.ui = ui
        self.popup_items = None
        self.popup_title = ""
        self.popup_offset = 0
        self.popup_waiting = False
        self.items = []
        self.pending = None
        self.follow = True
        self.offset = 0
        self.line_count = 0
        self.busy = False
        self.audio_key = None
        self.cancelled = threading.Event()
        self._revision = 0
        self._cache_key = None
        self._lines = []
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
            [self.header, self.top_separator, self.frame, self.bottom_separator, composer]
        )
        session.layout.container = FloatContainer(self.layout, floats=list(input_floats.floats))
        self.completion_floats = [item for item in session.layout.container.floats if item.ycursor]
        for item in self.completion_floats:
            item.ycursor = False
            item.bottom = 3
        popup_window = Window(
            FormattedTextControl(
                self.popup_fragments,
                get_cursor_position=lambda: Point(0, self.popup_offset),
            ),
            wrap_lines=False,
            style="bg:#22313d",
            get_vertical_scroll=lambda window: self.popup_offset,
        )
        session.layout.container.floats.append(
            Float(
                ConditionalContainer(popup_window, Condition(lambda: self.popup_items is not None)),
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
        session.mouse_support = True
        session.default_buffer.read_only = Condition(lambda: self.busy or self.popup_waiting)
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
                max(0, self.popup_line_count - self.popup_height()), self.popup_offset + 3
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

        @bindings.add("escape", "enter", filter=Condition(lambda: self.busy or self.popup_waiting))
        @bindings.add("enter", filter=Condition(lambda: self.busy))
        @bindings.add("f1", filter=Condition(lambda: self.busy))
        @bindings.add("f3", filter=Condition(lambda: self.busy))
        def ignore(event):
            pass

        @bindings.add("c-c", filter=Condition(lambda: self.busy))
        @bindings.add("escape", filter=Condition(lambda: self.busy))
        @bindings.add("c-d", filter=Condition(lambda: self.busy))
        def cancel(event):
            self.cancelled.set()

        @bindings.add("f2", filter=Condition(lambda: self.busy and self.audio_key == "f2"))
        @bindings.add("f4", filter=Condition(lambda: self.busy and self.audio_key == "f4"))
        def stop_audio(event):
            self.cancelled.set()

        self._original_print = ui.console.print
        self._original_status = ui.console.status
        ui.console.print = self.print
        ui.console.status = self.status
        session.placeholder = lambda: "Working… Ctrl-C to cancel" if self.busy else ""

    @contextmanager
    def popup(self, title):
        self.popup_items = []
        self.popup_title = title
        self.popup_offset = 0
        try:
            yield
            if self.popup_items:
                self.popup_waiting = True
                self.ui.prompt_session.app.run()
        finally:
            self.popup_waiting = False
            self.popup_items = None
            self.ui.prompt_session.app.invalidate()

    def popup_height(self):
        self.popup_fragments()
        return max(
            1, min(self.popup_line_count, 28, self.ui.prompt_session.app.output.get_size().rows - 6)
        )

    def popup_mouse(self, event):
        if event.event_type == MouseEventType.SCROLL_UP:
            self.popup_offset = max(0, self.popup_offset - 3)
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            self.popup_offset = min(
                max(0, self.popup_line_count - self.popup_height()), self.popup_offset + 3
            )
        self.ui.prompt_session.app.invalidate()

    def popup_fragments(self):
        width = max(1, min(100, self.ui.prompt_session.app.output.get_size().columns - 4))
        content = Group(
            Text(f" {self.popup_title} · Enter/Esc closes · ↑/↓ scroll", style="bold cyan"),
            *(self.popup_items or []),
            *([self.pending] if self.pending is not None else []),
        )
        lines = self.ui.console.render_lines(
            content, self.ui.console.options.update(width=width), pad=True
        )
        self.popup_line_count = len(lines)
        fragments = []
        for line in lines:
            for segment in line:
                style = segment.style or Style()
                fragments.extend(
                    (fragment[0], fragment[1], self.popup_mouse)
                    for fragment in to_formatted_text(
                        ANSI(style.render(segment.text, color_system="truecolor"))
                    )
                )
            fragments.append(("", "\n"))
        return fragments

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

    def composer_height(self):
        size = self.ui.prompt_session.app.output.get_size()
        width = max(1, size.columns - 4)
        rows = sum(
            max(1, (get_cwidth(line.expandtabs(4)) + width) // width)
            for line in self.ui.prompt_session.default_buffer.text.split("\n")
        )
        height = min(max(3, size.rows // 2), rows + 2)
        for item in getattr(self, "completion_floats", []):
            item.bottom = height
        return Dimension.exact(height)

    def close(self):
        self.ui.prompt_session.app.renderer.finish()
        self.ui.console.print = self._original_print
        self.ui.console.status = self._original_status
        self._original_print(Group(*self.items))

    @contextmanager
    def status(self, value, *, user=False, **kwargs):
        spinner = Spinner("dots", text=Text(str(value)), style="green" if user else "bright_cyan")
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
        target = self.popup_items if self.popup_items is not None else self.items
        target.append(
            content
            if all(isinstance(item, Panel) for item in renderables)
            else self.ui.chat_panel(content)
        )
        self._revision += 1
        self.ui.prompt_session.app.invalidate()

    def print_info(self, content):
        """Append subdued metadata without turning it into a response bubble."""
        target = self.popup_items if self.popup_items is not None else self.items
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
        if event.event_type == MouseEventType.SCROLL_UP:
            self.scroll(-3)
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll(3)

    def fragments(self):
        width = max(1, self.ui.prompt_session.app.output.get_size().columns - 4)
        animated = self.pending is not None
        key = (width, self._revision, int(time.monotonic() * 10) if animated else 0)
        if key != self._cache_key:
            content = list(self.items)
            if self.pending is not None and self.popup_items is None:
                content.append(self.pending)
            self._lines = self.ui.console.render_lines(
                Group(*content), self.ui.console.options.update(width=max(1, width)), pad=False
            )
            self._cache_key = key
        self.line_count = len(self._lines)
        offset = self.scroll_position(self.window)
        height = self.viewport_height()
        result = []
        for number, line in enumerate(self._lines):
            for segment in line:
                style = edge_style(
                    segment.style or Style(), number, offset, height, self.line_count
                )
                formatted = to_formatted_text(
                    ANSI(style.render(segment.text, color_system="truecolor"))
                )
                if formatted:
                    result.append((formatted[0][0], fragment_list_to_text(formatted), self.mouse))
            if number < len(self._lines) - 1:
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

    def generate(self, generate, preview):
        self.busy = True
        self.cancelled.clear()
        self.follow = True
        self.update(preview(None))
        outcome = []
        app = self.ui.prompt_session.app

        def callback(event, result):
            if self.cancelled.is_set():
                raise KeyboardInterrupt
            self.update(preview(result))

        def worker():
            try:
                outcome.append((True, generate(callback)))
            except BaseException as exc:
                outcome.append((False, exc))
            finally:
                app.loop.call_soon_threadsafe(app.exit, "")

        thread = threading.Thread(target=worker, daemon=True)
        try:
            self.ui.prompt_session.prompt("", bottom_toolbar=None, pre_run=thread.start)
        finally:
            self.cancelled.set()
            thread.join()
            self.busy = False
            self.update(None)
        if not outcome[0][0]:
            raise outcome[0][1]
        return outcome[0][1]
