"""Offline microphone capture and MLX Whisper transcription."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from .errors import MlxChatError

DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/whisper-large-v3-turbo"


def preload_transcription_model(model: str = DEFAULT_TRANSCRIPTION_MODEL) -> None:
    """Load Whisper into MLX's cache so the first voice prompt is responsive."""
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


def transcribe_microphone(*, seconds: int = 30, model: str = DEFAULT_TRANSCRIPTION_MODEL) -> str:
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
        import mlx_whisper
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise MlxChatError(
            "Voice support is not installed. Run `uv sync --extra voice` (or rerun ./setup.sh)."
        ) from exc

    sample_rate = 16_000
    block_size = 1_600
    chunks: list[object] = []
    silent_blocks = 0
    started = time.monotonic()
    try:
        with sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_size
        ) as stream:
            while time.monotonic() - started < seconds:
                block, _overflowed = stream.read(block_size)
                chunks.append(block.copy())
                rms = float(np.sqrt(np.mean(np.square(block))))
                silent_blocks = silent_blocks + 1 if rms < 0.012 else 0
                # Avoid clipping a speaker's first words, then finish promptly
                # after 1.2 seconds of quiet.
                if len(chunks) >= 20 and silent_blocks >= 12:
                    break
    except KeyboardInterrupt:
        # Ctrl-C deliberately ends capture and retains the recorded prompt.
        pass
    except Exception as exc:
        raise MlxChatError(
            "Could not record from the microphone. Grant microphone access to your terminal/Codex "
            "and check the selected input device."
        ) from exc
    if not chunks:
        raise MlxChatError("No audio was captured.")
    audio = np.concatenate(chunks, axis=0)
    with tempfile.NamedTemporaryFile(
        prefix="mlx-chat-voice-", suffix=".wav", delete=False
    ) as handle:
        path = Path(handle.name)
    try:
        sf.write(path, audio, sample_rate)
        output = mlx_whisper.transcribe(str(path), path_or_hf_repo=model)
    except Exception as exc:
        raise MlxChatError(f"Voice transcription failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    text = str(output.get("text", "")).strip() if isinstance(output, dict) else ""
    if not text:
        raise MlxChatError("I could not detect speech. Try speaking closer to the microphone.")
    return text
