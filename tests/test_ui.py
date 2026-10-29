from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.text import Text

from mlx_chat.client import GenerationResult, StreamEvent
from mlx_chat.theme import SURFACE
from mlx_chat.ui import TerminalUI, sanitize_terminal_text, short_model


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_terminal_sanitization_strips_ansi_and_control_characters():
    unsafe = "safe\x1b[31m red\x1b[0m\x1b[2J\n\tline\x00\x07\x7f\x85 café"

    assert sanitize_terminal_text(unsafe) == "safe red\n\tline café"


def test_terminal_sanitization_preserves_printable_unicode_newlines_and_tabs():
    text = "héllo 👋\n\tindented\rremoved"

    assert sanitize_terminal_text(text) == "héllo 👋\n\tindentedremoved"


def test_clear_erases_scrollback_and_visible_terminal():
    output = TTYBuffer()
    ui = object.__new__(TerminalUI)
    ui.console = Console(file=output, force_terminal=True)

    ui.clear()

    assert output.getvalue() == "\x1b[3J\x1b[H\x1b[2J"


def test_plain_generation_renders_completed_response_once_without_live_ui(monkeypatch):
    monkeypatch.setattr("rich.live.Live", lambda *a, **kw: pytest.fail("No live UI in plain mode"))
    ui = object.__new__(TerminalUI)
    final = []
    ui.final_generation = lambda result, **kw: final.append(result)
    result = GenerationResult(answer="Completed answer")

    def generate(callback):
        callback(StreamEvent("content", text="Partial"), result)
        return result

    assert ui.render_generation(generate, think=False) is result
    assert final == [result]


def test_plain_generation_propagates_errors_without_output():
    ui = object.__new__(TerminalUI)
    ui.final_generation = lambda *a, **kw: pytest.fail("Do not render a failed response")
    with pytest.raises(KeyboardInterrupt):
        ui.render_generation(lambda cb: (_ for _ in ()).throw(KeyboardInterrupt), think=False)


@pytest.mark.parametrize(
    ("model", "maximum", "expected"),
    [
        ("org/model", 38, "model"),
        ("model", 5, "model"),
        ("org/abcdefgh", 5, "abcd…"),
    ],
)
def test_short_model(model, maximum, expected):
    assert short_model(model, maximum) == expected


def test_f1_starts_voice_and_f2_speaks(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from mlx_chat.config import Settings

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        commands = []
        event = SimpleNamespace(
            app=SimpleNamespace(exit=lambda **kw: commands.append(kw["result"]))
        )
        bindings = ui.prompt_session.key_bindings
        bindings.get_bindings_for_keys(("f1",))[-1].handler(event)
        bindings.get_bindings_for_keys(("f2",))[-1].handler(event)
        assert commands == ["/voice", "/speak"]
        assert not any(
            binding.keys == ("f6",) for binding in bindings.get_bindings_for_keys(("f6",))
        )
        assert bindings.get_bindings_for_keys(("c-v",))


def test_user_prompt_commits_text_once_and_audio_commands_use_inline_status(monkeypatch, tmp_path):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from mlx_chat.config import Settings

    output = io.StringIO()
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **_: Console(file=output, force_terminal=True, color_system=None),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        pipe.send_text("Hello there\n")
        assert ui.prompt() == "Hello there"
        assert output.getvalue().count("Hello there") == 1
        assert "● Hello there" in output.getvalue()
        assert "You" not in output.getvalue()
        assert "╭" not in output.getvalue()
        assert "╰" not in output.getvalue()
        pipe.send_text("/voice\n")
        assert ui.prompt() == "/voice"
        pipe.send_text("/speak\n")
        assert ui.prompt() == "/speak"
        assert "/voice" not in output.getvalue()
        assert "/speak" not in output.getvalue()


def test_user_boxes_use_full_width_and_grow_vertically():
    ui = object.__new__(TerminalUI)
    ui.console = Console(file=io.StringIO(), width=50, color_system=None)
    heights = []
    for value in ("hello", "x" * 100, "one\ntwo\nthree"):
        lines = ui.console.render_lines(ui.user_line(value), ui.console.options)
        assert all(sum(len(segment.text) for segment in line) == 50 for line in lines)
        heights.append(len(lines))
    assert heights[0] == 3
    assert heights[1] > heights[0]
    assert heights[2] == 5
    live = ui.audio_status("Listening… F3 to finish")
    ui.console.print(live.renderable)
    rows = ui.console.file.getvalue().splitlines()
    spinner_row = next(row for row in rows if "Listening" in row)
    assert len(spinner_row) == 50
    assert not any(char in "".join(rows) for char in "│╭╰")
    assert ui.user_line("hello").style == f"on {SURFACE}"
    assert ui.chat_panel(Text("answer")).style == ""


def test_escape_stops_server_and_escape_enter_still_inserts_newline(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from mlx_chat.config import Settings

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        results = []
        inserted = []
        event = SimpleNamespace(
            app=SimpleNamespace(exit=lambda **kw: results.append(kw["result"])),
            current_buffer=SimpleNamespace(insert_text=inserted.append),
        )
        bindings = ui.prompt_session.key_bindings
        bindings.get_bindings_for_keys(("escape",))[-1].handler(event)
        assert results == ["/stop"]
        bindings.get_bindings_for_keys(("escape", "c-m"))[0].handler(event)
        assert inserted == ["\n"]
        assert results == ["/stop"]


@pytest.mark.parametrize(
    ("think", "concise", "expected"),
    [(False, False, "off"), (True, False, "on"), (True, True, "off")],
)
def test_info_panel_shows_explicit_thinking_and_audio_readiness(
    monkeypatch, think, concise, expected
):
    from mlx_chat import tts, voice
    from mlx_chat.config import Settings

    ui = object.__new__(TerminalUI)
    ui.settings = Settings(think=think, concise=concise)
    ui._session_name = None
    ui._message_count = 6
    ui._banner_state = {"voice_ready": False}
    monkeypatch.setattr(voice, "_model_ready", False)
    monkeypatch.setattr(tts, "_engine", None)
    output = io.StringIO()
    console = Console(file=output, width=160, color_system=None)
    console.print(ui.info_panel())
    content = output.getvalue()
    assert len(content.splitlines()) == 2
    assert "Ready" not in content
    assert content.index("mlx-chat") < content.index("Qwen3-4B-4bit") < content.index("unsaved")
    assert ("thinking" in content) == (expected == "on")
    assert "Direct" not in content
    assert "3 turns" in content
    assert "voice ready" not in content
    monkeypatch.setattr(voice, "_model_ready", True)
    monkeypatch.setattr(tts, "_engine", object())
    output.seek(0)
    output.truncate(0)
    ui.agent_status = "proj · ask"
    ui.context_share = 0.5
    console.print(ui.info_panel())
    content = output.getvalue()
    assert "voice ready" in content
    assert "agent proj · ask" in content
    assert "context ▰▰▰▰▱▱▱▱ 50%" in content


def test_thinking_spinner_animates_and_label_remains_clickable(monkeypatch):
    from mlx_chat.ui import ThinkingTrace

    trace = ThinkingTrace("A thought")
    trace.loading = True
    console = Console(file=io.StringIO(), width=80)
    monkeypatch.setattr("mlx_chat.rendering.time.monotonic", lambda: 100.0)
    first = console.render_lines(trace, console.options, pad=False)
    monkeypatch.setattr("mlx_chat.rendering.time.monotonic", lambda: 100.2)
    second = console.render_lines(trace, console.options, pad=False)
    first_text = "".join(segment.text for line in first for segment in line)
    second_text = "".join(segment.text for line in second for segment in line)
    assert first_text != second_text
    assert "Thinking" in second_text
    assert "click to expand" in second_text
    assert "A thought" not in second_text
    assert any(
        segment.style and segment.style.meta.get("thinking_trace") == id(trace)
        for line in second
        for segment in line
    )
    trace.loading = False
    trace.expanded = True
    final = console.render_lines(trace, console.options, pad=False)
    assert (
        "".join(segment.text for segment in final[1]).strip() == "◆ Thinking  ·  click to collapse"
    )
    assert "A thought" in "".join(segment.text for line in final for segment in line)


@pytest.mark.parametrize(
    "edit,expected",
    [
        ("\x01replacement", "replacement"),
        ("\x01\x7f", ""),
        ("\x15", ""),
        ("\x01\x1b[200~pasted\x1b[201~", "pasted"),
    ],
)
def test_composer_select_all_replace_delete_and_clear(monkeypatch, tmp_path, edit, expected):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from mlx_chat.config import Settings

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        pipe.send_text("original text" + edit + "\n")
        assert ui.prompt() == expected


def test_composer_copy_preserves_selection_and_cut_paste_use_clipboard(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from mlx_chat.config import Settings

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(file=io.StringIO(), force_terminal=True, width=80, height=40),
    )
    clipboard = []

    def clipboard_command(args, **kwargs):
        if args == ["pbcopy"]:
            clipboard.append(kwargs["input"])
            return SimpleNamespace()
        return SimpleNamespace(stdout=clipboard[-1])

    monkeypatch.setattr("mlx_chat.terminal.subprocess.run", clipboard_command)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings())
        ui.enable_conversation()
        pipe.send_text("copy me\x01\x03\x7fnew\n")
        assert ui.prompt() == "new"
        assert clipboard == ["copy me"]
        pipe.send_text("cut me\x01\x18\x16\n")
        assert ui.prompt() == "cut me"
        assert clipboard[-1] == "cut me"
