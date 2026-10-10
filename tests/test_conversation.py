from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.filters import to_filter
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console, Group
from rich.text import Text

from mlx_chat.client import GenerationResult, StreamEvent
from mlx_chat.config import Settings
from mlx_chat.ui import TerminalUI


def test_fullscreen_conversation_keeps_input_below_history_and_streams(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    output = io.StringIO()
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=output, force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        ui.console.print(Text("Earlier conversation"))
        assert output.getvalue() == ""
        assert ui.prompt_session.app.full_screen
        assert ui.prompt_session.app.renderer.full_screen
        assert ui.prompt_session.layout.container.content.children[0] is view.header
        assert ui.prompt_session.layout.container.content.children[2] is view.frame

        def generate(callback):
            result = GenerationResult(answer="Streaming answer")
            callback(StreamEvent("content", text="Streaming answer"), result)
            return result

        result = view.generate(
            generate, lambda result: Text(result.answer if result else "Waiting")
        )
        assert result.answer == "Streaming answer"
        assert view.pending is None
        assert not view.busy
        ui.user_message("My prompt")
        pipe.send_text("/system\n")
        assert ui.prompt() == "/system"
        ui.close_conversation()
    assert "Earlier conversation" not in output.getvalue()
    assert "My prompt" not in output.getvalue()
    assert "Chat closed" in output.getvalue()


def test_scrolling_up_pauses_follow_until_returning_to_bottom():
    from mlx_chat.conversation import ConversationView

    view = object.__new__(ConversationView)
    view.window = SimpleNamespace(render_info=SimpleNamespace(window_height=10))
    view.ui = SimpleNamespace(
        prompt_session=SimpleNamespace(app=SimpleNamespace(invalidate=lambda: None))
    )
    view.line_count = 100
    view.offset = 90
    view.follow = True
    view.scroll(-20)
    assert view.offset == 70
    assert not view.follow
    view.line_count = 120
    assert view.scroll_position(view.window) == 70
    view.scroll(100)
    assert view.follow
    assert view.offset == 110


def test_audio_stop_keys_are_handled_while_operation_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        for key, stop_key in (("\x1bOR", "f3"), ("\x1b[13~", "f3")):
            pipe.send_text(key)
            with ui.audio_status("Audio operation"):
                assert ui.run_audio(lambda stopped: stopped.wait(1), stop_key=stop_key)
        ui.close_conversation()


def test_input_height_is_small_and_grows_with_multiline_text(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        ui.prompt_session.default_buffer.validate_while_typing = to_filter(False)
        assert view.composer_height().preferred == 6
        ui.prompt_session.default_buffer.text = "one\ntwo\nthree"
        assert view.composer_height().preferred == 6
        ui.prompt_session.default_buffer.text = "x" * 200
        assert view.composer_height().preferred > 3
        assert view.composer_height().preferred <= 20
        ui.close_conversation()


def test_mouse_wheel_moves_rendered_viewport_without_cursor_reset(monkeypatch, tmp_path):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        ui.console.print(Text("\n".join(f"History row {i}" for i in range(150))))
        positions = []
        headers = []

        def wheel():
            headers.append("".join(fragment[1] for fragment in view.header_fragments()))
            positions.append(view.window.vertical_scroll)
            view.window.content.mouse_handler(
                MouseEvent(
                    position=Point(79, 1),
                    event_type=MouseEventType.SCROLL_UP,
                    button=MouseButton.NONE,
                    modifiers=frozenset(),
                )
            )

        def finish():
            headers.append("".join(fragment[1] for fragment in view.header_fragments()))
            positions.append(view.window.vertical_scroll)
            pipe.send_text("/exit\n")

        def ready():
            ui.prompt_session.app.loop.call_later(0.05, wheel)
            ui.prompt_session.app.loop.call_later(0.1, finish)

        ui.prompt_session.prompt("", pre_run=ready)
        assert positions[0] > 0
        assert positions[1] == positions[0] - 3
        assert not view.follow
        assert headers[0] == headers[1]
        assert "mlx-chat" in headers[0]
        assert "Thinking:" in headers[0]
        assert "Ready" not in "".join(fragment[1] for fragment in view.fragments())
        ui.close_conversation()


def test_edge_fade_only_dims_clipped_edges_and_preserves_links():
    from rich.style import Style

    from mlx_chat.conversation import CONVERSATION_BACKGROUND, edge_style

    original = Style(color="white", link="https://example.com", meta={"label": "code"})
    top = edge_style(original, row=20, offset=20, height=10, total=100)
    middle = edge_style(original, row=24, offset=20, height=10, total=100)
    bottom = edge_style(original, row=29, offset=20, height=10, total=100)
    assert top.color.get_truecolor().red < middle.color.get_truecolor().red
    assert bottom.color.get_truecolor().red < middle.color.get_truecolor().red
    assert top.bgcolor.name == CONVERSATION_BACKGROUND
    assert top.link == original.link
    assert top.meta == original.meta
    # At the actual start/end of a short conversation there is no hidden content.
    first = edge_style(original, row=0, offset=0, height=10, total=5)
    last = edge_style(original, row=4, offset=0, height=10, total=5)
    assert first.color.get_truecolor() == original.color.get_truecolor()
    assert last.color.get_truecolor() == original.color.get_truecolor()


def test_native_transcription_runs_on_main_thread_with_ui_active(monkeypatch, tmp_path):
    import threading

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()

        def transcribe():
            assert threading.current_thread() is threading.main_thread()
            assert ui.prompt_session.app.is_running
            return "Transcribed question"

        with ui.audio_status("Transcribing…"):
            assert ui.run_native(transcribe) == "Transcribed question"
        assert not ui._conversation.busy
        ui.close_conversation()


def test_prompt_and_generation_transitions_do_not_clear_or_leave_screen(monkeypatch, tmp_path):
    class TrackingOutput(DummyOutput):
        def __init__(self):
            self.enters = 0
            self.leaves = 0
            self.clears = 0

        def enter_alternate_screen(self):
            self.enters += 1

        def quit_alternate_screen(self):
            self.leaves += 1

        def erase_screen(self):
            self.clears += 1

        def erase_down(self):
            self.clears += 1

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    output = TrackingOutput()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        pipe.send_text("Hello\n")
        assert ui.prompt() == "Hello"
        initial_clears = output.clears
        ui._conversation.generate(lambda callback: "result", lambda result: Text("Streaming"))
        pipe.send_text("/help\n")
        assert ui.prompt() == "/help"
        assert output.enters == 1
        assert output.leaves == 0
        assert output.clears == initial_clears
        ui.close_conversation()
        assert output.leaves == 1


def test_restart_stops_starts_and_warms_server_with_interface_running(monkeypatch, tmp_path):
    from mlx_chat import chat

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    events = []
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        app = chat.ChatApp.__new__(chat.ChatApp)
        app.ui = ui
        app.settings = ui.settings
        app._autosave = lambda: None
        record = SimpleNamespace(server_args=[])
        app.manager = SimpleNamespace(matching_record=lambda: record)
        app.start_result = SimpleNamespace(managed=True)
        started = SimpleNamespace(managed=True, reused=False, pid=None)

        def stop(record):
            assert ui.prompt_session.app.is_running
            events.append("stop")
            return SimpleNamespace(stopped=True)

        def start(*, ready_check):
            assert ui.prompt_session.app.is_running
            events.append("start")
            ready_check(started)
            return started

        new_manager = SimpleNamespace(start=start)
        new_client = SimpleNamespace(warmup=lambda **kw: events.append("warmup"))
        monkeypatch.setattr(chat, "stop_record", stop)
        monkeypatch.setattr(chat, "ServerManager", lambda settings: new_manager)
        monkeypatch.setattr(chat, "ChatClient", lambda settings: new_client)
        app._restart(app.settings.model)
        assert events == ["stop", "start", "warmup"]
        assert app.manager is new_manager
        assert app.client is new_client
        assert app.start_result is started
        assert not ui._conversation.busy
        ui.close_conversation()


def test_conversation_preserves_message_backgrounds_through_edge_fade():
    from rich.style import Style

    from mlx_chat.conversation import edge_style

    for background in ("#20352b", "#22313d", "#343024"):
        original = Style(color="white", bgcolor=background)
        for row in (20, 24, 29):
            rendered = edge_style(original, row=row, offset=20, height=10, total=100)
            assert rendered.bgcolor == original.bgcolor


def test_help_uses_response_marker_and_hint_stays_above_history(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=120, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        ui.banner(reused=False, managed=True)
        assert not view.items
        hint = view.hint_fragments()[0][1]
        assert hint.strip() == ui.conversation_hint()
        ui.user_message("/help")
        ui.help()
        lines = ui.console.render_lines(Group(*view.items), ui.console.options, pad=False)
        text = "\n".join("".join(segment.text for segment in line) for line in lines)
        assert "● /help" in text
        assert "◆" not in text
        assert "mlx-chat commands" not in text
        assert ui.conversation_hint() not in text
        tip = view.items[-1]
        assert isinstance(tip, Text)
        assert tip.plain.startswith("    // sends")
        assert tip.style == "dim"
        assert tip.no_wrap


def test_answer_metrics_are_unboxed_dim_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=120, height=40),
    )
    monkeypatch.setattr("mlx_chat.rendering.resource_snapshot", lambda _port: None)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        ui.final_generation(
            GenerationResult(answer="Hello", elapsed=0.5, first_token_seconds=0.1), think=False
        )
        metadata = ui._conversation.items[-1]
        assert isinstance(metadata, Text)
        assert metadata.style == "dim"
        assert metadata.no_wrap
        assert "0.5s" in metadata.plain
        assert "TTFT 0.1s" in metadata.plain
        assert "◆" not in metadata.plain
        lines = ui.console.render_lines(metadata, ui.console.options, pad=False)
        assert all(
            segment.style is None or segment.style.bgcolor is None
            for line in lines
            for segment in line
        )


def test_slash_completion_menu_opens_above_bottom_composer(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        rendered = []

        def capture():
            screen = ui.prompt_session.app.renderer._last_screen
            rows = {
                y: "".join(cell.char for _, cell in sorted(cells.items()))
                for y, cells in screen.data_buffer.items()
            }
            rendered.append(rows)
            ui.prompt_session.app.exit(result="/s")

        def ready():
            pipe.send_text("/s\t")
            ui.prompt_session.app.loop.call_later(0.15, capture)

        ui.prompt_session.prompt("", pre_run=ready)
        rows = rendered[0]
        assert not any("Window too small" in row for row in rows.values())
        menu_rows = [y for y, row in rows.items() if "/speak" in row or "/settings" in row]
        assert menu_rows
        composer_top = max(y for y, row in rows.items() if "●" in row)
        assert all(y < composer_top for y in menu_rows)


def test_command_popup_captures_help_without_changing_conversation(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        ui.user_message("Existing message")
        before = list(view.items)
        pipe.send_text("\x1b")
        with ui.command_popup("/help"):
            ui.help()
            assert view.items == before
            text = "".join(fragment[1] for fragment in view.popup_fragments())
            heading = "".join(fragment[1] for fragment in view.popup_heading())
            footer = "".join(fragment[1] for fragment in view.popup_footer())
            assert "Commands" in heading
            assert "scroll" not in heading
            assert "// sends" not in text
            assert "// sends" not in footer
            assert "// literal /" in heading
            assert "close" not in heading
        assert view.popup_items is None
        assert view.items == before
        ui.voice_message("Hello from microphone")
        lines = ui.console.render_lines(view.items[-1], ui.console.options, pad=False)
        text = "".join(segment.text for line in lines for segment in line)
        assert "● /voice Hello from microphone" in text


def test_popup_design_keeps_title_and_footer_separate_from_scrollable_body(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=100, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        ui.user_message("Message behind popup")
        before = view.fragments()
        view.popup_items = []
        view.popup_title = "/help"
        ui.help()
        heading = view.popup_heading()
        footer = view.popup_footer()
        body = view.popup_fragments()
        assert "esc" in "".join(part[1] for part in heading)
        assert "close" not in "".join(part[1] for part in heading)
        assert "Enter/Esc" not in "".join(part[1] for part in heading)
        assert "scroll" not in "".join(part[1] for part in heading)
        assert any("bg:#23282e" in part[0] for part in body)
        assert view.fragments() != before
        view.popup_offset = 10
        assert view.popup_heading() == heading
        assert view.popup_footer() == footer
        view.popup_items = None
        assert view.fragments() == before


@pytest.mark.parametrize("popup", [False, True])
def test_drag_selects_and_copies_while_mouse_scrolling_stays_enabled(monkeypatch, tmp_path, popup):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.filters import is_true
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from rich.segment import Segment

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    copied = []
    monkeypatch.setattr(
        "mlx_chat.terminal.subprocess.run", lambda args, **kw: copied.append(kw["input"])
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        lines = [[Segment("Hello world")], [Segment("Second line")]]
        view._lines = lines
        view._popup_lines = lines
        for kind, x, y in [
            (MouseEventType.MOUSE_DOWN, 0, 0),
            (MouseEventType.MOUSE_MOVE, 5, 0),
            (MouseEventType.MOUSE_UP, 5, 0),
        ]:
            view.selection_mouse(
                MouseEvent(
                    position=Point(x, y),
                    event_type=kind,
                    button=MouseButton.LEFT,
                    modifiers=frozenset(),
                ),
                popup=popup,
            )
        assert copied == ["Hello"]
        assert is_true(ui.prompt_session.mouse_support)
        from rich.style import Style

        fragments, _ = view.selection_fragments(
            "Hello world", Style(color="white"), 0, 0, popup=popup
        )
        assert any("bg:#506174" in part[0] for part in fragments[:5])
        view.selection = (popup, (1, 6), (0, 6))
        assert view.selected_text() == "world\nSecond"


def test_long_chat_reuses_history_and_formats_only_viewport(monkeypatch, tmp_path):
    from unittest.mock import Mock

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        view._viewport_rows = 20
        for index in range(2000):
            ui.user_message(f"History message {index}")
        view.prepare_lines()
        render = Mock(wraps=ui.console.render_lines)
        monkeypatch.setattr(ui.console, "render_lines", render)
        formatting = Mock(wraps=view.line_fragments)
        monkeypatch.setattr(view, "line_fragments", formatting)
        view.fragments()
        assert render.call_count == 0
        assert formatting.call_count <= 20
        assert view.line_count == 6000
        view.update(Text("Streaming preview"))
        view.fragments()
        assert render.call_count == 1
        assert render.call_args.args[0] is view.pending
        view.update(Text("Longer streaming preview"))
        view.fragments()
        assert render.call_count == 2
        render.reset_mock()
        ui.user_message("New message")
        view.update(None)
        view.fragments()
        assert sum(call.args[0] is view.items[-1] for call in render.call_args_list) == 1
        assert not any(
            any(call.args[0] is old for old in view.items[:-1]) for call in render.call_args_list
        )
        assert view.line_count == 6003
        ui.clear()
        ui.user_message("After clear")
        view.fragments()
        assert view.line_count == 3
        assert "After clear" in "".join(part[1] for part in view.fragments())


@pytest.mark.parametrize("submit", [True, False])
def test_input_during_generation_redirects_or_preserves_draft_and_partial_answer(
    monkeypatch, tmp_path, submit
):
    import time

    from mlx_chat.client import ChatClient

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    monkeypatch.setattr("mlx_chat.rendering.resource_snapshot", lambda _port: None)
    client = ChatClient(Settings())

    def stream(*args, **kwargs):
        yield StreamEvent("content", text="Partial answer")
        time.sleep(0.15)
        yield StreamEvent("content", text=" continues")

    monkeypatch.setattr(client, "stream_chat", stream)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        pipe.send_text("New direction" + ("\n" if submit else "\x03"))
        result = ui.render_generation(
            lambda cb: client.collect([], think=False, on_event=cb), think=False
        )
        assert result.finish_reason == "cancelled"
        assert result.answer.startswith("Partial answer")
        lines = ui.console.render_lines(
            Group(*ui._conversation.items), ui.console.options, pad=False
        )
        assert "Partial answer" in "".join(segment.text for line in lines for segment in line)
        if submit:
            assert list(ui._conversation.submitted_inputs) == ["New direction"]
        else:
            assert ui._conversation.draft_text == "New direction"
            pipe.send_text("\n")
        assert ui.prompt() == "New direction"
        assert not ui._conversation.submitted_inputs
        assert not ui._conversation.busy


@pytest.mark.parametrize("draft", ["Hello", "\n".join("line" for _ in range(40))])
def test_composer_is_centered_inset_and_has_bottom_breathing_room(monkeypatch, tmp_path, draft):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        captured = []

        def capture():
            screen = ui.prompt_session.app.renderer._last_screen
            captured.append(
                {
                    y: "".join(cell.char for _, cell in sorted(cells.items()))
                    for y, cells in screen.data_buffer.items()
                }
            )
            ui.prompt_session.app.exit(result="")

        def ready():
            ui.prompt_session.app.loop.call_later(0.05, capture)

        ui.prompt_session.prompt("", default=draft, pre_run=ready)
        assert 36 <= view.composer_width() < 80
        assert view.composer_height().preferred <= 20
        rows = captured[0]
        assert not any("Window too small" in row for row in rows.values())
        input_y, input_row = next((y, row) for y, row in rows.items() if "●" in row)
        assert input_row.startswith(" " * 6)
        assert input_y < max(rows) - 1
        assert not any("return send" in row for row in rows.values())


def test_thinking_click_expands_trace_and_stats_touch_answer(monkeypatch, tmp_path):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    monkeypatch.setattr("mlx_chat.rendering.resource_snapshot", lambda _port: None)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        ui.final_generation(
            GenerationResult(
                answer="Final answer", reasoning="First thought\nSecond thought", elapsed=0.7
            ),
            think=True,
        )
        view.prepare_lines()

        def text_rows():
            return ["".join(segment.text for segment in line) for line in view._lines]

        original = text_rows()
        assert not original[0].strip()
        assert "◆ Thinking" in original[1]
        assert "click to expand" in original[1]
        assert not original[2].strip()
        assert not any("First thought" in line for line in original)
        answer_row = next(index for index, line in enumerate(original) if "Final answer" in line)
        assert "0.7s" in original[answer_row + 1]
        assert original[answer_row + 1].startswith("    ")

        def click():
            for kind in (MouseEventType.MOUSE_DOWN, MouseEventType.MOUSE_UP):
                view.mouse(
                    MouseEvent(
                        position=Point(4, 1),
                        event_type=kind,
                        button=MouseButton.LEFT,
                        modifiers=frozenset(),
                    )
                )
            view.prepare_lines()

        click()
        assert any("First thought" in line for line in text_rows())
        assert any("click to collapse" in line for line in text_rows())
        assert any("Second thought" in line for line in text_rows())
        click()
        assert text_rows() == original
        hint = ui.conversation_hint()
        assert all(key in hint for key in ("F1", "F2", "F3"))


@pytest.mark.parametrize(
    "command,expected", [("/voice", "/voice "), ("/settings", "/settings"), ("/load", "/load ")]
)
def test_clicking_help_command_places_it_in_composer_without_running_it(
    monkeypatch, tmp_path, command, expected
):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from prompt_toolkit.utils import get_cwidth

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=100, height=40),
    )
    monkeypatch.setattr(
        "mlx_chat.terminal.subprocess.run",
        lambda *args, **kwargs: pytest.fail("Clicking a command must not copy or execute it"),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation

        def click():
            for y, line in enumerate(view._popup_lines):
                x = 0
                for segment in line:
                    if segment.style and segment.style.meta.get("help_command") == expected:
                        for kind in (MouseEventType.MOUSE_DOWN, MouseEventType.MOUSE_UP):
                            view.popup_mouse(
                                MouseEvent(
                                    position=Point(x + 1, y),
                                    event_type=kind,
                                    button=MouseButton.LEFT,
                                    modifiers=frozenset(),
                                )
                            )
                        return
                    x += get_cwidth(segment.text)
            pytest.fail(f"No clickable {command} found")

        original_run = ui.prompt_session.app.run
        monkeypatch.setattr(
            ui.prompt_session.app,
            "run",
            lambda: original_run(
                pre_run=lambda: ui.prompt_session.app.loop.call_later(0.05, click)
            ),
        )
        with ui.command_popup("/help"):
            ui.help()
        assert view.draft_text == expected
        assert not view.items
        assert not view.submitted_inputs
        monkeypatch.setattr(ui.prompt_session.app, "run", original_run)
        pipe.send_text("\n")
        assert ui.prompt() == expected


def test_generation_stats_have_two_blank_rows_before_next_user_message(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        ui.generation_info("3.0s · TTFT 0.1s")
        ui.user_message("Next question")
        ui._conversation.prepare_lines()
        rows = ["".join(segment.text for segment in line) for line in ui._conversation._lines]
        stats_row = next(i for i, row in enumerate(rows) if "TTFT" in row)
        next_row = next(i for i, row in enumerate(rows) if "Next question" in row)
        assert next_row - stats_row >= 4
        assert all(not row.strip() for row in rows[stats_row + 1 : next_row])


@pytest.mark.parametrize("stopped", [False, True])
def test_close_restores_terminal_with_compact_summary_without_chat(monkeypatch, tmp_path, stopped):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    output = io.StringIO()
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(
            file=output, force_terminal=True, width=100, height=40, color_system=None
        ),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        ui.user_message("Private chat text")
        ui.console.print(ui.chat_panel(Text("Private answer")))
        ui.set_context("saved-session", 8)
        ui.server_stopped = stopped
        ui.close_conversation()
        assert ui._conversation is None
        text = output.getvalue()
        assert "Chat closed" in text
        assert "4 turns" in text
        assert "saved-session" in text
        assert ("Server stopped" if stopped else "Server left running") in text
        assert "Private chat text" not in text
        assert "Private answer" not in text
        assert len(text.splitlines()) <= 8


def test_composer_has_visual_placeholder_and_mac_paste_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=100, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        captured = []

        def capture():
            screen = ui.prompt_session.app.renderer._last_screen
            captured.append(
                "".join(cell.char for row in screen.data_buffer.values() for cell in row.values())
            )
            ui.prompt_session.app.exit(result="")

        ui.prompt_session.prompt(
            "", pre_run=lambda: ui.prompt_session.app.loop.call_later(0.05, capture)
        )
        assert "Write a message" in captured[0]
        assert "⌘V" not in captured[0]
        assert not ui.prompt_session.default_buffer.text
        view.mode = "generation"
        assert "redirect" in view.composer_placeholder()


@pytest.mark.parametrize("command", ["/help", "/settings", "/system"])
def test_command_preview_opens_while_typing_and_closes_when_edited(monkeypatch, tmp_path, command):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=100, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        view = ui._conversation
        observed = []
        attempts = []

        def inspect_preview():
            attempts.append(True)
            if (
                not view.popup_preview
                or not view.popup_items
                or (
                    isinstance(view.popup_items[0], Text)
                    and view.popup_items[0].plain == "Preparing…"
                )
            ):
                if len(attempts) < 100:
                    ui.prompt_session.app.loop.call_later(0.01, inspect_preview)
                    return
                ui.prompt_session.app.exit(exception=AssertionError("Preview did not open"))
                return
            observed.append(view.popup_title)
            assert not view.popup_waiting
            assert not ui.prompt_session.default_buffer.read_only()
            assert ui.prompt_session.default_buffer.text == command
            assert not view.items
            pipe.send_text("\x7f")
            ui.prompt_session.app.loop.call_later(0.05, inspect_closed)

        def inspect_closed():
            assert view.popup_items is None
            assert not view.popup_preview
            assert ui.prompt_session.default_buffer.text == command[:-1]
            pipe.send_text("\n")

        def ready():
            pipe.send_text(command)
            ui.prompt_session.app.loop.call_later(0.05, inspect_preview)

        result = ui.prompt_session.prompt("", pre_run=ready)
        assert result == command[:-1]
        assert observed == [command]
        assert not view.items


def test_escape_quits_with_one_key_even_when_live_help_is_open(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=100, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()

        def ready():
            pipe.send_text("/help")
            ui.prompt_session.app.loop.call_later(0.1, lambda: pipe.send_text("\x1b"))

        assert ui.prompt_session.prompt("", pre_run=ready) == "/stop"
        ui._conversation.dismiss_preview()


def test_f3_speech_popup_closes_after_one_second_without_enter(monkeypatch, tmp_path):
    import time

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(
            file=io.StringIO(),
            force_terminal=True,
            width=80,
            height=40,
        ),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        started = time.monotonic()
        try:
            pipe.send_text("\x1bOR")
            with ui.command_popup("/speak"):
                with ui.audio_status("Speaking… F3 to stop"):
                    assert ui.run_audio(lambda stopped: stopped.wait(2), stop_key="f3")
                ui.info("Speech stopped.")
            elapsed = time.monotonic() - started
            assert 0.9 <= elapsed < 1.8
            assert ui._conversation.popup_items is None
            assert not ui._conversation.popup_waiting
            assert ui._conversation._audio_popup_timer is None
        finally:
            ui.close_conversation()


def test_f3_hides_popup_even_while_audio_worker_finishes(monkeypatch, tmp_path):
    import threading
    import time

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(
            file=io.StringIO(),
            force_terminal=True,
            width=80,
            height=40,
        ),
    )
    release = threading.Event()
    observed = []
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        app = ui.prompt_session.app

        def operation(stopped):
            assert stopped.wait(1)

            def check():
                observed.append(ui._conversation.popup_items is None)
                release.set()

            app.loop.call_soon_threadsafe(app.loop.call_later, 1.15, check)
            assert release.wait(2)
            return True

        try:
            pipe.send_text("\x1bOR")
            started = time.monotonic()
            with ui.command_popup("/speak"), ui.audio_status("Speaking… F3 to stop"):
                assert ui.run_audio(operation, stop_key="f3")
            assert observed == [True]
            assert time.monotonic() - started < 1.8
            assert ui._conversation.mode == "input"
        finally:
            release.set()
            ui.close_conversation()


@pytest.mark.parametrize("press_f3", [False, True])
def test_finished_speech_popup_dismisses_automatically(monkeypatch, tmp_path, press_f3):
    import time

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(
            file=io.StringIO(),
            force_terminal=True,
            width=80,
            height=40,
        ),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        try:
            started = time.monotonic()
            with ui.command_popup("/speak"):
                with ui.audio_status("Speaking… F3 to stop"):
                    ui.run_audio(lambda stopped: False)
                ui.info("Speech finished.")
                if press_f3:
                    pipe.send_text("\x1bOR")
            assert 0.9 <= time.monotonic() - started < 1.8
            assert ui._conversation.popup_items is None
        finally:
            ui.close_conversation()
