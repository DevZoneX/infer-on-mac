"""Simple local English speech with Kokoro."""

from __future__ import annotations

import re
import threading
import unicodedata
import warnings
from contextlib import nullcontext

from prompt_toolkit.keys import Keys

from .errors import MlxChatError
from .keyboard import stop_key_listener

DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_TTS_VOICE = "af_heart"
_engine: object | None = None
_engine_lock = threading.Lock()
EMOJI_RE = re.compile(
    r"[\U0001F000-\U0001FAFF\U0001FC00-\U0001FFFD\U00002600-\U000027BF\u200d\ufe0e\ufe0f\u20e3]+"
)


def speech_text(text: str) -> str:
    """Read prose, not emoji, Markdown, code blocks, URLs, or model tokens."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"<\|[^>]*\|>|\[/?(?:INST|SYS)\]|</?s>", "", text)
    text = re.sub(r"```.*?(?:```|$)|~~~.*?(?:~~~|$)", " ", text, flags=re.DOTALL)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)", "", text)
    text = EMOJI_RE.sub("", text)
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) not in {"So", "Sk", "Cf", "Cc"} or char.isspace()
    )
    return re.sub(r"\s+", " ", re.sub(r"[*_`~|]", "", text)).strip()


def is_ready() -> bool:
    """Whether the speech model has successfully loaded in this process."""
    return _engine is not None


def _load_engine() -> object:
    global _engine
    with _engine_lock:
        if _engine is None:
            from huggingface_hub.utils import disable_progress_bars
            from kokoro_mlx import KokoroTTS

            disable_progress_bars()
            _engine = KokoroTTS.from_pretrained(DEFAULT_TTS_MODEL)
        return _engine


def speak(text: str, *, stop_event=None) -> bool:
    """Speak locally; return True if F3 or Ctrl-C stopped playback."""
    text = speech_text(text)
    if not text or not any(char.isalnum() for char in text):
        raise MlxChatError("There is no text to speak.")
    try:
        # Kokoro's phonemizer still uses this deprecated PyTorch API. Hide only
        # that known upstream notice, not other warnings or speech errors.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.jit\.script` is deprecated\..*",
                category=FutureWarning,
                module=r"torch\.jit\._script",
            )
            with (
                nullcontext(stop_event) if stop_event is not None else stop_key_listener(Keys.F3)
            ) as stopped:
                engine = _load_engine()
                if not stopped.is_set():
                    engine.speak(text, voice=DEFAULT_TTS_VOICE, stream=True, stop_event=stopped)
                return stopped.is_set()
    except ImportError as exc:
        raise MlxChatError("Speech output is not installed. Run `uv sync --extra voice`.") from exc
    except KeyboardInterrupt:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        raise
    except Exception as exc:
        raise MlxChatError(f"Speech output failed: {exc}") from exc
