"""Generation profiles."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import Settings, _atomic_json_write, app_home, normalize_session_name
from .context import DEFAULT_SYSTEM_PROMPT
from .errors import MlxChatError
from .sessions import SESSION_SETTINGS

PROFILE_FIELDS = ("system_prompt", *SESSION_SETTINGS)
BUILTIN_PROFILES = {
    "default": {},
    "coding": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT + " Focus on practical coding help. "
        "Explain the relevant code, suggest focused changes, "
        "and include useful verification steps.",
        "temperature": 0.2,
    },
    "writing": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT + " Help with writing and editing. "
        "Preserve the author's intent and voice. Use clear prose and concrete wording.",
        "temperature": 0.7,
    },
}


def profile_path(name: str) -> Path:
    return app_home() / "profiles" / f"{normalize_session_name(name, label='Profile')}.json"


def list_profiles() -> list[str]:
    return sorted(
        set(BUILTIN_PROFILES) | {path.stem for path in (app_home() / "profiles").glob("*.json")}
    )


def apply_profile(name: str, settings: Settings) -> Settings:
    path = profile_path(name)
    name = path.stem
    if name in BUILTIN_PROFILES:
        values = BUILTIN_PROFILES[name]
    else:
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise MlxChatError(
                f"No profile named {name!r}. Use /profile to list profiles."
            ) from exc
        except (OSError, ValueError) as exc:
            raise MlxChatError(f"Could not read profile {name!r}: {exc}") from exc
    if (
        not isinstance(values, dict)
        or set(values) - set(PROFILE_FIELDS)
        or any(value is None for value in values.values())
    ):
        raise MlxChatError("Profiles must contain only generation settings and a system prompt.")
    defaults = asdict(Settings())
    generation = {key: defaults[key] for key in PROFILE_FIELDS}
    generation.update(values)
    return settings.merged(**generation)


def save_profile(name: str, settings: Settings) -> Path:
    path = profile_path(name)
    if path.stem in BUILTIN_PROFILES:
        raise MlxChatError("Choose a custom name; built-in profiles cannot be overwritten.")
    settings.validate()
    _atomic_json_write(path, {key: getattr(settings, key) for key in PROFILE_FIELDS})
    return path


def delete_profile(name: str) -> None:
    path = profile_path(name)
    if path.stem in BUILTIN_PROFILES:
        raise MlxChatError("Built-in profiles cannot be deleted.")
    try:
        path.unlink()
    except FileNotFoundError as exc:
        raise MlxChatError(f"No saved profile named {path.stem!r}.") from exc
