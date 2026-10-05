from __future__ import annotations

import io
import time

import pytest
from rich.console import Console
from rich.text import Text

from mlx_chat.client import GenerationResult, StreamEvent
from mlx_chat.ui import RenderState, TerminalUI, sanitize_terminal_text, short_model


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


@pytest.mark.parametrize("height", [3, 4, 6, 7, 10, 24])
def test_live_preview_never_exceeds_terminal_height_and_follows_tail(height):
    output = io.StringIO()
    ui = object.__new__(TerminalUI)
    ui.console = Console(
        file=output,
        width=42,
        height=height,
        force_terminal=False,
        color_system=None,
    )
    result = GenerationResult(
        reasoning="EARLY-REASONING\n" + "reasoning text " * 100 + "\nLATE-REASONING",
        answer="EARLY-ANSWER\n" + "answer text " * 100 + "\nLATE-ANSWER",
    )

    ui.console.print(
        ui._live_group(RenderState(result, True, time.monotonic())),
        end="",
    )
    rendered = output.getvalue()
    rendered_height = rendered.count("\n") + int(bool(rendered and not rendered.endswith("\n")))

    assert rendered_height <= height
    assert "EARLY-ANSWER" not in rendered
    assert "LATE-ANSWER" in rendered
    if height >= 7:
        assert "LATE-REASONING" in rendered


def test_live_renderer_is_bounded_and_cleared_before_exit(monkeypatch):
    instances = []
    final = []

    class FakeLive:
        def __init__(self, renderable, **options):
            self.options = options
            self.updates = [renderable]
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, renderable, *, refresh):
            assert refresh is True
            self.updates.append(renderable)

    monkeypatch.setattr("rich.live.Live", FakeLive)
    ui = object.__new__(TerminalUI)
    ui.console = Console(file=io.StringIO(), width=40, height=8, force_terminal=False)
    ui.final_generation = lambda result, *, think: final.append((result, think))

    def generate(callback):
        result = GenerationResult(answer="complete answer")
        callback(StreamEvent("content", text="complete answer"), result)
        return result

    result = ui.render_generation(generate, think=True)

    live = instances[0]
    assert live.options["vertical_overflow"] == "ellipsis"
    assert live.options["auto_refresh"] is False
    assert isinstance(live.updates[-1], Text)
    assert live.updates[-1].plain == ""
    assert final == [(result, True)]


def test_live_renderer_is_cleared_when_generation_is_interrupted(monkeypatch):
    updates = []

    class FakeLive:
        def __init__(self, renderable, **_options):
            updates.append(renderable)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, renderable, *, refresh):
            assert refresh is True
            updates.append(renderable)

    monkeypatch.setattr("rich.live.Live", FakeLive)
    ui = object.__new__(TerminalUI)
    ui.console = Console(file=io.StringIO(), width=40, height=8, force_terminal=False)

    with pytest.raises(KeyboardInterrupt):
        ui.render_generation(lambda _callback: (_ for _ in ()).throw(KeyboardInterrupt), think=True)

    assert isinstance(updates[-1], Text)
    assert updates[-1].plain == ""


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


def test_f1_starts_voice_and_f3_speaks(monkeypatch, tmp_path):
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
        bindings.get_bindings_for_keys(("f1",))[0].handler(event)
        bindings.get_bindings_for_keys(("f3",))[0].handler(event)
        assert commands == ["/voice", "/speak"]
        assert not bindings.get_bindings_for_keys(("f6",))
        assert not bindings.get_bindings_for_keys(("c-v",))


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
    live = ui.audio_status("Listening… F2 to finish")
    ui.console.print(live.renderable)
    rows = ui.console.file.getvalue().splitlines()
    spinner_row = next(row for row in rows if "Listening" in row)
    assert len(spinner_row) == 50
    assert not any(char in "".join(rows) for char in "│╭╰")
    assert ui.user_line("hello").style == "on #20352b"
    assert ui.chat_panel(Text("answer")).style == "on #22313d"


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
        bindings.get_bindings_for_keys(("escape",))[0].handler(event)
        assert results == ["/stop"]
        bindings.get_bindings_for_keys(("escape", "c-m"))[0].handler(event)
        assert inserted == ["\n"]
        assert results == ["/stop"]


def test_streaming_answer_starts_small_and_grows_to_viewport():
    ui = object.__new__(TerminalUI)
    ui.console = Console(file=io.StringIO(), width=50, height=24, color_system=None)
    heights = []
    for answer in ("Hello", "\n\n".join(f"Line {i}" for i in range(6)), "long text " * 500):
        group = ui._live_group(
            RenderState(GenerationResult(answer=answer), False, time.monotonic())
        )
        lines = ui.console.render_lines(group, ui.console.options)
        heights.append(len(lines))
    assert heights[0] == 4
    assert heights[0] < heights[1] < heights[2]
    assert heights[2] <= 24


def test_generation_keeps_user_box_below_answer_and_within_screen():
    ui = object.__new__(TerminalUI)
    ui._interactive_prompt = True
    output = io.StringIO()
    ui.console = Console(file=output, width=50, height=24, color_system=None)
    group = ui._live_group(
        RenderState(
            GenerationResult(answer="word " * 500, reasoning="thinking " * 300),
            True,
            time.monotonic(),
        )
    )
    ui.console.print(group)
    rendered = output.getvalue()
    assert "Answer" not in rendered
    assert "You" not in rendered
    assert "◆" in rendered
    assert "Generating…" in rendered
    assert rendered.count("\n") <= 24


def test_bottom_anchor_reserves_rows_without_clearing_history():
    ui = object.__new__(TerminalUI)
    ui._interactive_prompt = True
    output = io.StringIO()
    ui.console = Console(file=output, width=50, height=24)
    ui._anchor_bottom()
    assert output.getvalue() == "\x1b[24;1H\n\n\n\n\x1b[21;1H"


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
    assert len(content.splitlines()) == 4
    assert "Ready" not in content
    assert content.index("TTS:") < content.index("STT:") < content.index("Session:")
    assert f"Thinking: {expected}" in content
    assert "Direct" not in content
    assert "Turns: 3" in content
    assert "STT: whisper-large-v3-turbo · not loaded" in content
    assert "TTS: Kokoro-82M-bf16 · not loaded" in content
    monkeypatch.setattr(voice, "_model_ready", True)
    monkeypatch.setattr(tts, "_engine", object())
    output.seek(0)
    output.truncate(0)
    console.print(ui.info_panel())
    content = output.getvalue()
    assert "STT: whisper-large-v3-turbo · ready" in content
    assert "TTS: Kokoro-82M-bf16 · ready" in content
