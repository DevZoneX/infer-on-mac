"""Terminal text and clipboard."""

import os
import re
import subprocess

from .errors import MlxChatError

ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def sanitize_terminal_text(value: str) -> str:
    """Remove terminal control sequences while preserving newlines and tabs."""
    value = ANSI_RE.sub("", value)
    return "".join(
        char
        for char in value
        if char in {"\n", "\t"} or (ord(char) >= 32 and not 127 <= ord(char) <= 159)
    )


def copy_text(text: str) -> None:
    try:
        subprocess.run(["pbcopy"], input=text, text=True, check=True, timeout=2)
    except (OSError, subprocess.SubprocessError) as exc:
        raise MlxChatError("Could not copy text to the macOS clipboard.") from exc


def paste_text() -> str:
    try:
        return subprocess.run(
            ["pbpaste"], capture_output=True, text=True, check=True, timeout=2
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise MlxChatError("Could not read the macOS clipboard.") from exc


def short_model(model: str, maximum: int = 38) -> str:
    value = model.rsplit("/", 1)[-1]
    return value if len(value) <= maximum else f"{value[: maximum - 1]}…"


def is_plain_terminal() -> bool:
    return "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb"
