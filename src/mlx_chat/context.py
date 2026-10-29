"""System prompts and request context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .errors import MlxChatError

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant running locally on an Apple Silicon Mac running macOS. "
    "Give clear, direct answers. Tailor commands, paths, shortcuts, and setup instructions "
    "to macOS and its default zsh shell unless the user asks for another platform. "
    "Use fenced Markdown code blocks with a language label for code and commands. "
    "Be honest about uncertainty and never claim to have run commands or accessed files "
    "unless you actually have."
)

CONCISE_SYSTEM_PROMPT = (
    "Answer directly and briefly. Prefer the minimum explanation needed to be useful; "
    "do not reveal chain-of-thought."
)


def system_messages(settings: Settings) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": settings.system_prompt or DEFAULT_SYSTEM_PROMPT}]
    if settings.concise:
        messages.append({"role": "system", "content": CONCISE_SYSTEM_PROMPT})
    return messages


TEMPLATE_RESERVE = 256


def summary_message(text: str) -> dict[str, str]:
    return {
        "role": "system",
        "content": "Summary of the earlier conversation (older messages are not shown):\n" + text,
    }


def estimated_tokens(message: dict[str, Any], bytes_per_token: int = 1) -> int:
    # Estimate bytes instead of loading the model's tokenizer.
    size = len((message.get("content") or "").encode("utf-8"))
    if message.get("tool_calls"):
        size += len(json.dumps(message["tool_calls"], ensure_ascii=False).encode("utf-8"))
    return -(-size // bytes_per_token) + 16


@dataclass(frozen=True)
class ContextSelection:
    messages: list[dict[str, Any]]
    omitted_messages: int
    estimated_tokens: int


def select_context(
    messages: list[dict[str, Any]], settings: Settings, *, bytes_per_token: int = 1
) -> ContextSelection:
    """Keep system instructions and a contiguous suffix of complete recent turns.

    One byte per token never underestimates. Agent turns pass 3, which still
    overestimates code and prose for Qwen and Llama tokenizers.
    """

    def estimate(message: dict[str, Any]) -> int:
        return estimated_tokens(message, bytes_per_token)

    systems = [message for message in messages if message["role"] == "system"]
    history = [message for message in messages if message["role"] != "system"]
    budget = settings.context_tokens - settings.max_tokens - TEMPLATE_RESERVE
    cost = sum(estimate(message) for message in systems)
    if cost > budget:
        raise MlxChatError(
            "System instructions exceed the context budget. Shorten /system, increase "
            "--context-tokens, or reduce --max-tokens."
        )
    turns: list[list[dict[str, Any]]] = []
    for message in history:
        if message["role"] == "user" or not turns:
            turns.append([])
        turns[-1].append(message)
    selected = []
    for turn in reversed(turns):
        turn_cost = sum(estimate(message) for message in turn)
        if cost + turn_cost > budget:
            if not selected:
                raise MlxChatError(
                    "The latest turn exceeds the context budget. Shorten the prompt, increase "
                    "--context-tokens, or reduce --max-tokens."
                )
            break
        selected.append(turn)
        cost += turn_cost
    recent = [message for turn in reversed(selected) for message in turn]
    return ContextSelection(systems + recent, len(history) - len(recent), cost)
