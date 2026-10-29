"""Saved conversations and response metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .client import GenerationResult
from .config import Settings
from .errors import MlxChatError

SESSION_SETTINGS = (
    "think",
    "concise",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "context_tokens",
    "repetition_penalty",
)


def _duration(value: Any, field: str, *, optional=False) -> float | None:
    if value is None and optional:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise MlxChatError(f"Saved response {field} must be a finite non-negative number.")
    try:
        return float(value)
    except OverflowError as exc:
        raise MlxChatError(f"Saved response {field} is too large.") from exc


def result_from_message(message: dict[str, Any]) -> GenerationResult:
    usage = message.get("usage", {})
    metrics = message.get("metrics", {})
    finish_reason = message.get("finish_reason")
    if not isinstance(usage, dict) or not isinstance(metrics, dict):
        raise MlxChatError("Saved response usage and metrics must be objects.")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise MlxChatError("Saved response finish_reason must be text.")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if key in usage and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise MlxChatError(f"Saved response {key} must be a non-negative integer.")
    return GenerationResult(
        answer=message["content"],
        reasoning=message.get("reasoning", ""),
        usage=dict(usage),
        finish_reason=finish_reason,
        elapsed=_duration(metrics.get("elapsed", 0), "elapsed"),
        first_token_seconds=_duration(
            metrics.get("first_token_seconds"), "first_token_seconds", optional=True
        ),
        tool_calls=list(message.get("tool_calls") or []),
    )


def latest_result(messages: list[dict[str, Any]]) -> GenerationResult | None:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return result_from_message(message)
    return None


def _validate_tool_message(message: dict[str, Any]) -> None:
    if (
        any(not isinstance(message.get(key), str) for key in ("content", "tool_call_id", "name"))
        or any(
            key in message and not isinstance(message[key], str)
            for key in ("summary", "arguments", "diff", "approval")
        )
        or ("ok" in message and not isinstance(message["ok"], bool))
    ):
        raise MlxChatError("Saved session is not valid.")


@dataclass(frozen=True)
class RestoredSession:
    messages: list[dict[str, Any]]
    settings: Settings
    last_result: GenerationResult | None
    model: str | None


def restore_session(payload: dict[str, Any], settings: Settings) -> RestoredSession:
    messages = payload.get("messages")
    saved = payload.get("settings", {})
    system_prompt = payload.get("system_prompt", "")
    model = payload.get("model")
    if (
        payload.get("schema_version", 1) != 1
        or not isinstance(messages, list)
        or not isinstance(saved, dict)
        or not isinstance(system_prompt, str)
        or (model is not None and not isinstance(model, str))
    ):
        raise MlxChatError("Saved session is not valid.")
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            _validate_tool_message(message)
            continue
        if isinstance(message, dict) and message.get("role") == "summary":
            if not isinstance(message.get("content"), str):
                raise MlxChatError("Saved session is not valid.")
            continue
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("user", "assistant")
            or not isinstance(message.get("content"), str)
            or (
                "attachments" in message
                and (
                    not isinstance(message["attachments"], list)
                    or not all(isinstance(path, str) for path in message["attachments"])
                )
            )
            or any(
                key in message and not isinstance(message[key], str)
                for key in ("api_content", "reasoning")
            )
        ):
            raise MlxChatError("Saved session is not valid.")
        if message["role"] == "assistant":
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list) or not all(
                isinstance(call, dict)
                and isinstance(call.get("id"), str)
                and isinstance(call.get("function"), dict)
                and isinstance(call["function"].get("name"), str)
                and isinstance(call["function"].get("arguments"), str)
                for call in calls
            ):
                raise MlxChatError("Saved session is not valid.")
            result_from_message(message)
    overrides = {key: saved[key] for key in SESSION_SETTINGS if key in saved}
    # None must not silently bypass merged() validation for saved values.
    if any(value is None for value in overrides.values()):
        raise MlxChatError("Saved session settings are not valid.")
    candidate = settings.merged(system_prompt=system_prompt, **overrides)
    return RestoredSession(messages, candidate, latest_result(messages), model)
