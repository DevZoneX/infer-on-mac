from prompt_toolkit.input import create_pipe_input

from mlx_chat.voice import recording_stop_requested


def test_recording_stop_handles_f2_and_ctrl_c_without_blocking():
    with create_pipe_input() as keyboard:
        assert not recording_stop_requested(keyboard)
        keyboard.send_text("\x1bOP")
        assert not recording_stop_requested(keyboard)
        keyboard.send_text("x")
        assert not recording_stop_requested(keyboard)
        keyboard.send_text("\x1bOQ")
        assert recording_stop_requested(keyboard)
        keyboard.send_text("\x1b[12~")
        assert recording_stop_requested(keyboard)
        keyboard.send_text("\x03")
        assert recording_stop_requested(keyboard)
    assert not recording_stop_requested(None)


def test_conversation_recording_waits_for_f2_and_transcribes_captured_audio(monkeypatch):
    import sys
    import threading
    from types import SimpleNamespace

    import pytest

    np = pytest.importorskip("numpy")

    from mlx_chat import voice

    stopped = threading.Event()
    blocks = []
    output_paths = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            # More than two seconds of silence must not end F1/F2 recording.
            block = np.zeros((size, 1), dtype=np.float32)
            blocks.append(block)
            if len(blocks) == 25:
                stopped.set()
            return block, False

    monkeypatch.setattr(voice.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(InputStream=lambda **kw: Stream())
    )
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(
            write=lambda path, audio, rate: output_paths.append((path, audio.shape, rate))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper",
        SimpleNamespace(transcribe=lambda path, **kw: {"text": "My recorded question"}),
    )
    assert voice.transcribe_microphone(stop_event=stopped) == "My recorded question"
    assert len(blocks) == 25
    assert output_paths[0][1] == (40000, 1)
    assert output_paths[0][2] == 16000
    assert not output_paths[0][0].exists()


def test_recording_stops_after_speech_not_during_initial_silence(monkeypatch):
    import sys
    import threading
    from types import SimpleNamespace

    import pytest

    from mlx_chat import voice

    np = pytest.importorskip("numpy")
    levels = iter([0.0] * 25 + [0.1] * 3 + [0.0] * 12)
    reads = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            level = next(levels)
            reads.append(level)
            return np.full((size, 1), level, dtype=np.float32), False

    monkeypatch.setattr(voice.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(InputStream=lambda **kw: Stream())
    )
    stop = threading.Event()
    audio = voice.record_microphone(stop_event=stop)
    assert len(reads) == 40
    assert audio.shape == (64000, 1)
    assert not stop.is_set()
