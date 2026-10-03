"""Configuration, runtime paths, and saved chat sessions."""

from __future__ import annotations

import contextlib
import json
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
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 4096
    repetition_penalty: float = 1.0
    system_prompt: str = ""
    startup_timeout: int = 900
    request_timeout: int = 3600
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
        if not isinstance(self.model, str) or not self.model.strip():
            raise MlxChatError("Model name cannot be empty.")
        if not isinstance(self.host, str) or not self.host.strip():
            raise MlxChatError("Host cannot be empty.")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise MlxChatError("Port must be an integer between 1 and 65535.")
        if not 1 <= self.port <= 65535:
            raise MlxChatError("Port must be between 1 and 65535.")
        if not isinstance(self.think, bool):
            raise MlxChatError("think must be a boolean.")
        if not isinstance(self.concise, bool):
            raise MlxChatError("concise must be a boolean.")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise MlxChatError("Temperature must be a number between 0 and 5.")
        if not 0 <= self.temperature <= 5:
            raise MlxChatError("Temperature must be between 0 and 5.")
        if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
            raise MlxChatError("top_p must be a number between 0 and 1.")
        if not 0 <= self.top_p <= 1:
            raise MlxChatError("top_p must be between 0 and 1.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise MlxChatError("top_k must be a non-negative integer.")
        if self.top_k < 0:
            raise MlxChatError("top_k cannot be negative.")
        if isinstance(self.min_p, bool) or not isinstance(self.min_p, (int, float)):
            raise MlxChatError("min_p must be a number between 0 and 1.")
        if not 0 <= self.min_p <= 1:
            raise MlxChatError("min_p must be between 0 and 1.")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise MlxChatError("max_tokens must be a positive integer.")
        if self.max_tokens < 1:
            raise MlxChatError("max_tokens must be at least 1.")
        if isinstance(self.repetition_penalty, bool) or not isinstance(
            self.repetition_penalty, (int, float)
        ):
            raise MlxChatError("repetition_penalty must be a non-negative number.")
        if self.repetition_penalty < 0:
            raise MlxChatError("repetition_penalty cannot be negative.")
        if (
            isinstance(self.startup_timeout, bool)
            or not isinstance(self.startup_timeout, int)
            or isinstance(self.request_timeout, bool)
            or not isinstance(self.request_timeout, int)
        ):
            raise MlxChatError("Timeouts must be positive integers.")
        if self.startup_timeout < 1 or self.request_timeout < 1:
            raise MlxChatError("Timeouts must be positive.")
        if isinstance(self.web_results, bool) or not isinstance(self.web_results, int):
            raise MlxChatError("web_results must be an integer between 1 and 10.")
        if not 1 <= self.web_results <= 10:
            raise MlxChatError("web_results must be between 1 and 10.")
        if isinstance(self.web_timeout, bool) or not isinstance(self.web_timeout, int):
            raise MlxChatError("web_timeout must be an integer between 1 and 60.")
        if not 1 <= self.web_timeout <= 60:
            raise MlxChatError("web_timeout must be between 1 and 60 seconds.")
        if not isinstance(self.web_region, str) or not self.web_region.strip():
            raise MlxChatError("web_region cannot be empty.")
        if self.web_safesearch not in {"on", "moderate", "off"}:
            raise MlxChatError("web_safesearch must be on, moderate, or off.")
        if not isinstance(self.system_prompt, str):
            raise MlxChatError("system_prompt must be text.")
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
