from __future__ import annotations

import io
from types import SimpleNamespace

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
    assert "Earlier conversation" in output.getvalue()
    assert output.getvalue().count("My prompt") == 1


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
        for key, stop_key in (("\x1bOQ", "f2"), ("\x1bOS", "f4")):
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
        assert view.composer_height().preferred == 3
        ui.prompt_session.default_buffer.text = "one\ntwo\nthree"
        assert view.composer_height().preferred == 5
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
        assert tip.plain.startswith("  Tip: start")
        assert tip.style == "dim"
        assert tip.no_wrap


def test_answer_metrics_are_unboxed_dim_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=120, height=40),
    )
    monkeypatch.setattr(
        "mlx_chat.ui.resource_snapshot", lambda: type("Snapshot", (), {"processes": []})()
    )
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
        composer_top = max(y for y, row in rows.items() if "● You" in row)
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
            assert "/help" in text
            assert "Tip: start" in text
        assert view.popup_items is None
        assert view.items == before
        ui.voice_message("Hello from microphone")
        lines = ui.console.render_lines(view.items[-1], ui.console.options, pad=False)
        text = "".join(segment.text for line in lines for segment in line)
        assert "● /voice Hello from microphone" in text
