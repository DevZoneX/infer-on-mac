"""Configuration, runtime paths, and saved chat sessions."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from . import DEFAULT_MODEL
from .errors import MlxChatError

APP_DIR_NAME = "mlx-chat"
SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def app_home() -> Path:
    """Return the application data directory, honoring MLX_CHAT_HOME for tests/CI."""
    override = os.environ.get("MLX_CHAT_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / APP_DIR_NAME


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    host: str = "127.0.0.1"
    port: int = 8080
    think: bool = False
    concise: bool = False
    preload_voice: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 4096
    repetition_penalty: float = 1.0
    system_prompt: str = ""
    startup_timeout: int = 900
    request_timeout: int = 3600
    connect_timeout: int = 10
    context_tokens: int = 32768
    web_results: int = 5
    web_timeout: int = 10
    web_region: str = "us-en"
    web_safesearch: str = "moderate"
    server_args: list[str] | None = None

    def __post_init__(self) -> None:
        if self.server_args is None:
            self.server_args = []
        self.validate()

    def validate(self) -> None:
        for key in ("model", "host", "web_region", "system_prompt", "web_safesearch"):
            value = getattr(self, key)
            if not isinstance(value, str) or (key != "system_prompt" and not value.strip()):
                label = {"model": "Model name", "host": "Host"}.get(key, key)
                raise MlxChatError(f"{label} must be non-empty text.")
        for key in ("think", "concise", "preload_voice"):
            if not isinstance(getattr(self, key), bool):
                raise MlxChatError(f"{key} must be a boolean.")
        numbers = {
            "port": ("Port", int, 1, 65535),
            "temperature": ("Temperature", (int, float), 0, 5),
            "top_p": ("top_p", (int, float), 0, 1),
            "top_k": ("top_k", int, 0, None),
            "min_p": ("min_p", (int, float), 0, 1),
            "max_tokens": ("max_tokens", int, 1, None),
            "repetition_penalty": ("repetition_penalty", (int, float), 0, None),
            "startup_timeout": ("Timeouts", int, 1, None),
            "request_timeout": ("Timeouts", int, 1, None),
            "connect_timeout": ("connect_timeout", int, 1, None),
            "context_tokens": ("context_tokens", int, 1, None),
            "web_results": ("web_results", int, 1, 10),
            "web_timeout": ("web_timeout", int, 1, 60),
        }
        for key, (label, kind, minimum, maximum) in numbers.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, kind):
                raise MlxChatError(
                    f"{label} must be an integer." if kind is int else f"{label} must be a number."
                )
            if (
                (isinstance(value, float) and not math.isfinite(value))
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                limit = (
                    f"between {minimum} and {maximum}"
                    if maximum is not None
                    else (f"at least {minimum}")
                )
                raise MlxChatError(f"{label} must be finite and {limit}.")
        if self.context_tokens <= self.max_tokens + 256:
            raise MlxChatError("context_tokens must exceed max_tokens by more than 256 tokens.")
        if self.web_safesearch not in {"on", "moderate", "off"}:
            raise MlxChatError("web_safesearch must be on, moderate, or off.")
        if not isinstance(self.server_args, list) or not all(
            isinstance(item, str) for item in self.server_args
        ):
            raise MlxChatError("server_args must be a list of strings.")
        lifecycle_flags = {"--model", "--host", "--port"}
        for item in self.server_args:
            flag = item.split("=", 1)[0]
            if flag in {"-m", "-p"} or any(
                lifecycle_flag.startswith(flag) for lifecycle_flag in lifecycle_flags
            ):
                raise MlxChatError(
                    f"{flag} must use mlx-chat's own option so lifecycle state stays accurate."
                )

    @property
    def base_url(self) -> str:
        display_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        if ":" in display_host and not display_host.startswith("["):
            display_host = f"[{display_host}]"
        return f"http://{display_host}:{self.port}"

    def merged(self, **overrides: Any) -> Settings:
        data = asdict(self)
        data.update({key: value for key, value in overrides.items() if value is not None})
        return Settings(**data)


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def config_path() -> Path:
    return app_home() / "config.json"


def load_settings() -> Settings:
    path = config_path()
    if not path.exists():
        return Settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("configuration root must be a JSON object")
        allowed = {item.name for item in fields(Settings)}
        data = {key: value for key, value in raw.items() if key in allowed}
        return Settings(**data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, MlxChatError) as exc:
        raise MlxChatError(f"Could not read configuration at {path}: {exc}") from exc


def save_settings(settings: Settings) -> None:
    settings.validate()
    _atomic_json_write(config_path(), asdict(settings))


def sessions_dir() -> Path:
    return app_home() / "sessions"


def normalize_session_name(name: str) -> str:
    clean = name.strip()
    if not SESSION_NAME_RE.fullmatch(clean):
        raise MlxChatError(
            "Session names must start with a letter or number and contain only "
            "letters, numbers, dots, dashes, or underscores (64 characters max)."
        )
    return clean


def save_session(name: str, payload: dict[str, Any]) -> Path:
    clean = normalize_session_name(name)
    path = sessions_dir() / f"{clean}.json"
    _atomic_json_write(path, payload)
    return path


def load_session(name: str) -> dict[str, Any]:
    clean = normalize_session_name(name)
    path = sessions_dir() / f"{clean}.json"
    if not path.exists():
        raise MlxChatError(f"No saved session named {clean!r}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MlxChatError(f"Could not load session {clean!r}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise MlxChatError(f"Session {clean!r} is not valid.")
    return value


def list_sessions() -> list[tuple[str, Path]]:
    directory = sessions_dir()
    if not directory.exists():
        return []
    return [(item.stem, item) for item in sorted(directory.glob("*.json"))]


def history_path() -> Path:
    path = app_home() / "prompt_history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
