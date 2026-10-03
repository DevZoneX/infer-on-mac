import warnings
from unittest.mock import Mock

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


def test_speak_uses_only_english_kokoro(monkeypatch):
    engine = Mock()
    monkeypatch.setattr(tts, "_load_engine", lambda: engine)
    tts.speak("**Hello** 👋")
    engine.speak.assert_called_once_with("Hello", voice="af_heart", stream=True)


def test_empty_text_does_not_load_model(monkeypatch):
    load = Mock()
    monkeypatch.setattr(tts, "_load_engine", load)
    with pytest.raises(MlxChatError, match="no text"):
        tts.speak("❤️ 🚀")
    load.assert_not_called()


def test_errors_are_actionable(monkeypatch):
    engine = Mock()
    monkeypatch.setattr(tts, "_load_engine", lambda: engine)
    engine.speak.side_effect = ImportError("kokoro_mlx")
    with pytest.raises(MlxChatError, match="uv sync"):
        tts.speak("Hello")
    engine.speak.side_effect = RuntimeError("device unavailable")
    with pytest.raises(MlxChatError, match="Speech output failed"):
        tts.speak("Hello")


def test_ctrl_c_stops_audio(monkeypatch):
    sd = pytest.importorskip("sounddevice")
    stop = Mock()
    monkeypatch.setattr(sd, "stop", stop)
    engine = Mock()
    engine.speak.side_effect = KeyboardInterrupt
    monkeypatch.setattr(tts, "_load_engine", lambda: engine)
    with pytest.raises(KeyboardInterrupt):
        tts.speak("Hello")
    stop.assert_called_once()


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
    engine.speak.side_effect = emit_warnings
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
