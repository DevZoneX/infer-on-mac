from __future__ import annotations

import copy

import pytest

from mlx_chat.config import Settings
from mlx_chat.errors import MlxChatError
from mlx_chat.sessions import latest_result, restore_session
from mlx_chat.ui import TerminalUI


def conversation():
    return [
        {"role": "user", "content": "question", "api_content": "private enriched question"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "reason",
            "usage": {"completion_tokens": 10},
            "metrics": {"elapsed": 2.0, "first_token_seconds": 0.5},
            "finish_reason": "stop",
        },
    ]


def test_restore_reconstructs_result_and_preserves_runtime_settings():
    current = Settings(port=8123, connect_timeout=7, preload_voice=True)
    restored = restore_session(
        {"messages": conversation(), "settings": {"concise": True}, "model": "saved/model"},
        current,
    )
    assert restored.last_result.answer == "answer"
    assert restored.last_result.reasoning == "reason"
    assert restored.last_result.elapsed == 2
    assert restored.last_result.first_token_seconds == 0.5
    assert restored.last_result.tokens_per_second == 5
    assert restored.settings.concise is True
    assert restored.settings.port == 8123
    assert restored.settings.preload_voice is True
    assert current.concise is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("metrics", []),
        ("usage", []),
        ("finish_reason", []),
        ("metrics", {"elapsed": float("nan")}),
        ("metrics", {"elapsed": -1}),
        ("metrics", {"first_token_seconds": True}),
        ("usage", {"completion_tokens": -1}),
        ("usage", {"completion_tokens": True}),
    ],
)
def test_restore_rejects_corrupt_response_metadata_before_mutation(field, value):
    messages = conversation()
    messages[-1][field] = value
    settings = Settings()
    previous = copy.deepcopy(settings)
    with pytest.raises(MlxChatError, match="Saved response"):
        restore_session({"messages": messages}, settings)
    assert settings == previous


def test_restore_rejects_null_settings_and_unknown_schema():
    with pytest.raises(MlxChatError, match="not valid"):
        restore_session({"messages": [], "settings": {"think": None}}, Settings())
    with pytest.raises(MlxChatError, match="not valid"):
        restore_session({"messages": [], "schema_version": 2}, Settings())


def test_legacy_assistant_without_metrics_restores_with_safe_defaults():
    result = latest_result([{"role": "assistant", "content": "old answer"}])
    assert result.answer == "old answer"
    assert result.elapsed == 0
    assert result.first_token_seconds is None
    assert result.usage == {}
    assert latest_result([]) is None


def test_transcript_rendering_displays_visible_content_and_historical_metrics():
    from types import SimpleNamespace

    rendered = []
    ui = SimpleNamespace(
        user_message=lambda value: rendered.append(("user", value)),
        final_generation=lambda result, **kw: rendered.append(("assistant", result, kw)),
    )
    ui.clear = lambda: None
    TerminalUI.restore_transcript(ui, conversation())
    assert rendered[0] == ("user", "question")
    assert rendered[1][1].answer == "answer"
    assert rendered[1][2] == {"think": True, "historical": True}


@pytest.mark.parametrize("attachments", ["file.py", [None], {}])
def test_restoration_rejects_invalid_attachment_labels(attachments):
    with pytest.raises(MlxChatError, match="not valid"):
        restore_session(
            {"messages": [{"role": "user", "content": "question", "attachments": attachments}]},
            Settings(),
        )
