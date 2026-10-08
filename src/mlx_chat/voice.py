"""Offline microphone capture and MLX Whisper transcription."""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from contextlib import ExitStack, closing
from pathlib import Path

from prompt_toolkit.input import create_input
from prompt_toolkit.keys import Keys

from .errors import MlxChatError
from .keyboard import stop_requested

DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/whisper-large-v3-turbo"
_model_ready = False


def is_ready() -> bool:
    """Whether Whisper has successfully loaded in this process."""
    return _model_ready


def preload_transcription_model(model: str = DEFAULT_TRANSCRIPTION_MODEL) -> None:
    """Load Whisper into MLX's cache so the first voice prompt is responsive."""
    global _model_ready
    try:
        import mlx.core as mx
        from huggingface_hub.utils import disable_progress_bars
        from mlx_whisper.transcribe import ModelHolder
    except ImportError as exc:
        raise MlxChatError(
            "Voice support is not installed. Run `uv sync --extra dev --extra voice`."
        ) from exc
    # Keep Hugging Face's raw progress bar out of the Rich terminal UI; the
    # caller presents a single loading status instead.
    disable_progress_bars()
    # MLX Whisper transcribes with fp16 by default. Preload at that same dtype
    # so ModelHolder can be reused instead of leaving an incompatible fp32
    # model in its cache.
    ModelHolder.get_model(model, mx.float16)
    _model_ready = True


def recording_stop_requested(keyboard) -> bool:
    """Stop recording on F3 or Ctrl-C, without blocking microphone capture."""
    return stop_requested(keyboard, Keys.F3)


def record_microphone(*, seconds: int = 30, stop_event=None):
    """Record a bounded prompt locally, stopping early after sustained silence.

    Imports are intentionally delayed: regular text chat needs neither audio
    dependencies nor microphone permission.
    """
    if not 1 <= seconds <= 180:
        raise MlxChatError("Voice recording duration must be between 1 and 180 seconds.")
    if shutil.which("ffmpeg") is None:
        raise MlxChatError(
            "Voice transcription needs FFmpeg. Install it with `brew install ffmpeg`, "
            "then retry /voice."
        )
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise MlxChatError(
            "Voice support is not installed. Run `uv sync --extra voice` (or rerun ./setup.sh)."
        ) from exc

    sample_rate = 16_000
    block_size = 1_600
    chunks: list[object] = []
    silent_blocks = 0
    heard_speech = False
    try:
        with ExitStack() as stack:
            keyboard = None
            if stop_event is None and sys.stdin.isatty():
                keyboard = stack.enter_context(closing(create_input()))
                stack.enter_context(keyboard.raw_mode())
            stream = stack.enter_context(
                sd.InputStream(
                    samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_size
                )
            )
            # Permission/device setup can take time; start the recording budget
            # only after the stream has actually opened.
            started = time.monotonic()
            while time.monotonic() - started < seconds:
                if (stop_event is not None and stop_event.is_set()) or recording_stop_requested(
                    keyboard
                ):
                    break
                block, _overflowed = stream.read(block_size)
                chunks.append(block.copy())
                rms = float(np.sqrt(np.mean(np.square(block))))
                if rms >= 0.012:
                    heard_speech = True
                    silent_blocks = 0
                else:
                    silent_blocks += 1
                # Finish after 1.2 seconds of silence following actual speech.
                # Initial silence never ends capture; F3 remains a manual stop.
                if heard_speech and silent_blocks >= 12:
                    break
    except KeyboardInterrupt:
        # Ctrl-C deliberately ends capture and retains the recorded prompt.
        pass
    except Exception as exc:
        raise MlxChatError(
            "Could not record from the microphone. Grant microphone access to your terminal/Codex "
            f"and check the selected input device. Audio device error: {exc}"
        ) from exc
    if not chunks:
        raise MlxChatError("No audio was captured.")
    return np.concatenate(chunks, axis=0)


def transcribe_audio(audio, *, model: str = DEFAULT_TRANSCRIPTION_MODEL) -> str:
    """Run Whisper on the caller's thread, matching main-thread model preload."""
    global _model_ready
    try:
        import mlx_whisper
        import soundfile as sf
    except ImportError as exc:
        raise MlxChatError("Voice support is not installed. Run `uv sync --extra voice`.") from exc
    sample_rate = 16_000
    with tempfile.NamedTemporaryFile(
        prefix="mlx-chat-voice-", suffix=".wav", delete=False
    ) as handle:
        path = Path(handle.name)
    try:
        sf.write(path, audio, sample_rate)
        output = mlx_whisper.transcribe(str(path), path_or_hf_repo=model)
        _model_ready = True
    except Exception as exc:
        raise MlxChatError(f"Voice transcription failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    text = str(output.get("text", "")).strip() if isinstance(output, dict) else ""
    if not text:
        raise MlxChatError("I could not detect speech. Try speaking closer to the microphone.")
    return text


def transcribe_microphone(
    *, seconds: int = 30, model: str = DEFAULT_TRANSCRIPTION_MODEL, stop_event=None
) -> str:
    return transcribe_audio(record_microphone(seconds=seconds, stop_event=stop_event), model=model)
