import warnings
from unittest.mock import ANY, Mock

import pytest

from mlx_chat import tts
from mlx_chat.errors import MlxChatError


def test_clean_text():
    assert tts.speech_text("Hello 👋 — café! ❤️") == "Hello — café!"
    assert tts.speech_text("Ready  🚀   now") == "Ready now"
    assert (
        tts.speech_text(
            "<|assistant|>## **Hello**\n[friend](https://example.com) `world`\n"
            "```py\nprint(1)\n```<|end|>"
        )
        == "Hello friend world"
    )


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_speech_omits_unfinished_code_blocks(fence):
    assert tts.speech_text(f"Here is the example:\n{fence}python\nprint('hello')") == (
        "Here is the example:"
    )


def test_speak_uses_only_english_kokoro(monkeypatch):
    engine = Mock()
    monkeypatch.setattr(tts, "_load_engine", lambda: engine)
    play = Mock()
    monkeypatch.setattr(tts, "_play_stream", play)
    tts.speak("**Hello** 👋")
    engine.generate_stream.assert_called_once_with("Hello", voice="af_heart", sample_rate=24000)
    play.assert_called_once_with(engine.generate_stream.return_value, ANY)


def test_empty_text_does_not_load_model(monkeypatch):
    load = Mock()
    monkeypatch.setattr(tts, "_load_engine", load)
    with pytest.raises(MlxChatError, match="no text"):
        tts.speak("❤️ 🚀")
    load.assert_not_called()


def test_errors_are_actionable(monkeypatch):
    engine = Mock()
    monkeypatch.setattr(tts, "_load_engine", lambda: engine)
    engine.generate_stream.side_effect = ImportError("kokoro_mlx")
    with pytest.raises(MlxChatError, match="uv sync"):
        tts.speak("Hello")
    engine.generate_stream.side_effect = RuntimeError("device unavailable")
    with pytest.raises(MlxChatError, match="Speech output failed"):
        tts.speak("Hello")


def test_ctrl_c_propagates_after_owned_playback_cleanup(monkeypatch):
    engine = Mock()
    monkeypatch.setattr(tts, "_load_engine", lambda: engine)
    monkeypatch.setattr(tts, "_play_stream", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        tts.speak("Hello")


def test_only_known_torch_warning_is_hidden_and_filter_is_restored(monkeypatch):
    def emit_warnings(*args, **kwargs):
        warnings.warn_explicit(
            "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.",
            FutureWarning,
            filename="torch/jit/_script.py",
            lineno=1491,
            module="torch.jit._script",
        )
        warnings.warn("Unrelated speech warning", UserWarning, stacklevel=1)
        warnings.warn_explicit(
            "Another torch warning",
            FutureWarning,
            filename="torch/jit/_script.py",
            lineno=1,
            module="torch.jit._script",
        )

    engine = Mock()
    engine.generate_stream.side_effect = emit_warnings
    monkeypatch.setattr(tts, "_play_stream", lambda *args: None)
    monkeypatch.setattr(tts, "_load_engine", lambda: engine)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        before = list(warnings.filters)
        tts.speak("Hello")
        assert warnings.filters == before
    assert [str(item.message) for item in captured] == [
        "Unrelated speech warning",
        "Another torch warning",
    ]


def test_f3_stops_active_speech_through_playback_event(monkeypatch):
    from prompt_toolkit.input import create_pipe_input

    from mlx_chat import keyboard

    engine = Mock()
    observed = []

    def play(chunks, stopped):
        observed.append(stopped.wait(1))

    monkeypatch.setattr(tts, "_play_stream", play)
    monkeypatch.setattr(tts, "_load_engine", lambda: engine)
    monkeypatch.setattr(keyboard.sys.stdin, "isatty", lambda: True)
    with create_pipe_input() as pipe:
        monkeypatch.setattr(keyboard, "create_input", lambda: pipe)
        pipe.send_text("\x1bOR")
        assert tts.speak("Hello") is True
    # F3 can arrive during loading, in which case playback is skipped entirely.
    assert observed in ([], [True])


def test_stop_listener_handles_f3_encodings_and_restores_input(monkeypatch):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.keys import Keys

    from mlx_chat import keyboard

    monkeypatch.setattr(keyboard.sys.stdin, "isatty", lambda: True)
    for sequence in ("\x1bOR", "\x1b[13~", "\x03"):
        with create_pipe_input() as pipe:
            monkeypatch.setattr(keyboard, "create_input", lambda: pipe)
            with keyboard.stop_key_listener(Keys.F3) as stopped:
                pipe.send_text("\x1bOQ")  # F2 must not stop speech.
                assert not stopped.wait(0.05)
                pipe.send_text(sequence)
                assert stopped.wait(1)


@pytest.mark.parametrize("length", [480, 1440])
def test_stop_fades_remaining_audio_to_zero_without_a_sample_jump(monkeypatch, length):
    import sys
    import threading
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    stopped = threading.Event()
    writes = []
    source = np.full(length, 0.7, dtype=np.float32)

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, data):
            writes.append(data.copy().ravel())
            if np.any(data) and not stopped.is_set():
                stopped.set()

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **kw: Stream())
    )
    tts._play_stream(iter([source, source]), stopped)
    audible = [piece for piece in writes if np.any(piece)]
    assert len(audible) == 2
    original, fade = audible
    assert original[-1] == pytest.approx(fade[0])
    assert fade[-1] == 0
    assert np.all(np.diff(fade) <= 0)
    assert len(fade) == int(tts.SAMPLE_RATE * tts.FADE_SECONDS)
    assert np.all(writes[-1] == 0)


def test_stop_does_not_wait_for_a_blocked_synthesis_chunk(monkeypatch):
    import sys
    import threading
    import time
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    stopped = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def chunks():
        try:
            entered.set()
            release.wait(3)
            yield np.ones(480, dtype=np.float32)
        finally:
            finished.set()

    class Stream:
        def __enter__(self):
            assert entered.wait(1)
            stopped.set()
            return self

        def __exit__(self, *_args):
            return False

        def write(self, data):
            assert np.all(data == 0)

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **kw: Stream())
    )
    started = time.monotonic()
    try:
        tts._play_stream(chunks(), stopped)
        assert time.monotonic() - started < 0.5
        assert not finished.is_set()
    finally:
        release.set()
        assert finished.wait(1)


def test_ctrl_c_fades_before_closing_output_stream(monkeypatch):
    import sys
    import threading
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    stopped = threading.Event()
    writes = []
    closed = []
    interrupted = False

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed.append(True)
            return False

        def write(self, data):
            nonlocal interrupted
            if np.any(data) and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            writes.append(data.copy())

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **kw: Stream())
    )
    with pytest.raises(KeyboardInterrupt):
        tts._play_stream(iter([np.full(1440, 0.7, dtype=np.float32)]), stopped)
    assert stopped.is_set()
    assert closed == [True]
    assert writes[-2][-1, 0] == 0
    assert np.all(writes[-1] == 0)


def test_cancel_before_first_write_does_not_play_a_new_audio_tail(monkeypatch):
    import queue
    import sys
    import threading
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    stopped = threading.Event()
    original_queue = queue.Queue
    writes = []

    class CancellingQueue(original_queue):
        def get(self, *args, **kwargs):
            value = super().get(*args, **kwargs)
            stopped.set()
            return value

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, data):
            writes.append(data.copy())

    monkeypatch.setattr(tts.queue, "Queue", CancellingQueue)
    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **kw: Stream())
    )
    tts._play_stream(iter([np.full(1440, 0.7, dtype=np.float32)]), stopped)
    assert writes
    assert all(np.all(data == 0) for data in writes)
