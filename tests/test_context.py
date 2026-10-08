from __future__ import annotations

import pytest

from mlx_chat.config import Settings
from mlx_chat.context import select_context
from mlx_chat.errors import MlxChatError


def message(role, text):
    return {"role": role, "content": text}


def test_budget_keeps_system_and_recent_complete_turns_without_mutating_history():
    messages = [
        message("system", "Keep these instructions"),
        message("user", "old " * 80),
        message("assistant", "old answer " * 20),
        message("user", "recent question"),
        message("assistant", "recent answer"),
        message("user", "latest question"),
    ]
    original = list(messages)
    settings = Settings(context_tokens=600, max_tokens=100)
    selected = select_context(messages, settings)
    assert selected.messages == [messages[0], *messages[3:]]
    assert selected.omitted_messages == 2
    assert selected.estimated_tokens + settings.max_tokens + 256 <= settings.context_tokens
    assert messages == original


def test_budget_keeps_a_contiguous_suffix_without_skipping_oversized_turns():
    messages = [
        message("user", "small old question"),
        message("assistant", "small old answer"),
        message("user", "oversized " * 100),
        message("assistant", "middle answer"),
        message("user", "latest"),
    ]
    selected = select_context(messages, Settings(context_tokens=600, max_tokens=100))
    assert selected.messages == [messages[-1]]
    assert selected.omitted_messages == 4


@pytest.mark.parametrize("role", ["system", "user"])
def test_oversized_required_messages_are_rejected_instead_of_cut(role):
    with pytest.raises(MlxChatError, match="context budget"):
        select_context([message(role, "x" * 1000)], Settings(context_tokens=600, max_tokens=100))


def test_budget_counts_multibyte_text_and_web_context():
    messages = [message("user", "界" * 100)]
    with pytest.raises(MlxChatError, match="latest turn"):
        select_context(messages, Settings(context_tokens=600, max_tokens=100))


def test_empty_history_retains_all_system_instructions():
    messages = [message("system", "default"), message("system", "concise")]
    selected = select_context(messages, Settings())
    assert selected.messages == messages
    assert selected.omitted_messages == 0
