"""Kokoro speech output."""

from __future__ import annotations

import queue
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
_synthesis_lock = threading.Lock()
SAMPLE_RATE = 24000
FADE_SECONDS = 0.02
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


def _play_stream(chunks, stopped):
    import numpy as np
    import sounddevice as sd

    audio = queue.Queue(maxsize=2)
    shutdown = threading.Event()
    errors = []

    def put(chunk):
        while not stopped.is_set() and not shutdown.is_set():
            try:
                audio.put(chunk, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        try:
            # A cancelled native call may still finish; do not overlap model use.
            with _synthesis_lock:
                try:
                    if stopped.is_set() or shutdown.is_set():
                        return
                    for chunk in chunks:
                        if stopped.is_set() or shutdown.is_set():
                            break
                        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        if len(chunk) and not put(chunk):
                            break
                finally:
                    close = getattr(chunks, "close", None)
                    if close is not None:
                        close()
        except Exception as exc:
            errors.append(exc)
        finally:
            put(None)

    frames = int(SAMPLE_RATE * FADE_SECONDS)
    silence = np.zeros(frames, dtype=np.float32)
    remaining = silence[:0]
    last_sample = 0.0
    wrote_audio = False

    def fade(stream):
        tail = np.empty(frames, dtype=np.float32)
        count = min(frames, len(remaining)) if wrote_audio else 0
        tail[:count] = remaining[:count]
        tail[count:] = tail[count - 1] if count else last_sample
        gain = (1 + np.cos(np.linspace(0, np.pi, frames, dtype=np.float32))) / 2
        tail *= gain
        stream.write(tail.reshape(-1, 1))
        stream.write(silence.reshape(-1, 1))

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=frames,
            latency="low",
        ) as stream:
            try:
                while not stopped.is_set():
                    try:
                        chunk = audio.get(timeout=0.01)
                    except queue.Empty:
                        if not stopped.is_set():
                            stream.write(silence.reshape(-1, 1))
                            last_sample = 0.0
                            wrote_audio = False
                        continue
                    if chunk is None:
                        break
                    for offset in range(0, len(chunk), frames):
                        remaining = chunk[offset:]
                        if stopped.is_set():
                            break
                        piece = remaining[:frames]
                        last_sample = float(piece[-1])
                        wrote_audio = True
                        remaining = remaining[len(piece) :]
                        stream.write(piece.reshape(-1, 1))
                fade(stream)
            except KeyboardInterrupt:
                stopped.set()
                fade(stream)
                raise
    finally:
        shutdown.set()
    if errors and not stopped.is_set():
        raise errors[0]


def speak(text: str, *, stop_event=None) -> bool:
    """Speak locally; return True if F3 or Ctrl-C stopped playback."""
    text = speech_text(text)
    if not text or not any(char.isalnum() for char in text):
        raise MlxChatError("There is no text to speak.")
    try:
        # Suppress the known Kokoro/PyTorch warning, not playback errors.
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
                    _play_stream(
                        engine.generate_stream(
                            text, voice=DEFAULT_TTS_VOICE, sample_rate=SAMPLE_RATE
                        ),
                        stopped,
                    )
                return stopped.is_set()
    except ImportError as exc:
        raise MlxChatError("Speech output is not installed. Run `uv sync --extra voice`.") from exc
    except Exception as exc:
        raise MlxChatError(f"Speech output failed: {exc}") from exc
