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
