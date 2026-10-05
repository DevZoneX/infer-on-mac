from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx_chat import chat
from mlx_chat.client import GenerationResult
from mlx_chat.config import Settings
from mlx_chat.errors import MlxChatError
from mlx_chat.web import WebSearchResult, WebSource


def bare_app() -> chat.ChatApp:
    app = object.__new__(chat.ChatApp)
    app.settings = Settings()
    app.transcript = [{"role": "user", "content": "keep me"}]
    app.session_name = "current"
    app.last_result = GenerationResult(answer="old")
    app.ui = SimpleNamespace(
        success=lambda _message: None,
        warning=lambda _message: None,
    )
    app._autosave = lambda: None
    return app


def test_new_session_validation_does_not_clear_current_conversation():
    app = bare_app()
    original = list(app.transcript)

    with pytest.raises(MlxChatError, match="Session names"):
        app._cmd_new(["bad name"])

    assert app.transcript == original
    assert app.session_name == "current"


def test_load_session_is_transactional_when_saved_settings_are_invalid(monkeypatch):
    app = bare_app()
    original = list(app.transcript)
    monkeypatch.setattr(
        chat,
        "load_session",
        lambda _name: {
            "messages": [{"role": "user", "content": "replacement"}],
            "system_prompt": "replacement",
            "settings": {"temperature": 99},
        },
    )

    with pytest.raises(MlxChatError, match="Temperature"):
        app._load("broken")

    assert app.transcript == original
    assert app.session_name == "current"
    assert app.settings.system_prompt == ""
    assert app.settings.temperature == 0.7


def test_failed_retry_restores_previous_assistant_and_metrics():
    app = bare_app()
    assistant = {"role": "assistant", "content": "old"}
    app.transcript.append(assistant)
    previous = app.last_result
    app._generate = lambda _prompt=None: False

    app._cmd_retry()

    assert app.transcript[-1] is assistant
    assert app.last_result is previous


def test_clear_only_wipes_the_terminal_and_preserves_state():
    app = bare_app()
    original_transcript = list(app.transcript)
    original_result = app.last_result
    calls = []
    app.ui = SimpleNamespace(clear=lambda: calls.append(("clear",)))
    app._autosave = lambda: calls.append(("autosave",))

    app._cmd_clear([])

    assert calls == [("clear",)]
    assert app.transcript == original_transcript
    assert app.last_result is original_result


def test_clear_rejects_arguments_without_changing_the_terminal():
    app = bare_app()
    calls = []
    app.ui = SimpleNamespace(clear=lambda: calls.append("clear"))

    with pytest.raises(MlxChatError, match="Usage: /clear"):
        app._cmd_clear(["unexpected"])

    assert calls == []


def test_api_messages_uses_private_model_context_but_keeps_display_content():
    app = bare_app()
    app.transcript = [
        {"role": "user", "content": "visible query", "api_content": "enriched context"},
        {"role": "assistant", "content": "answer"},
    ]

    assert app.api_messages() == [
        {"role": "system", "content": chat.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "enriched context"},
        {"role": "assistant", "content": "answer"},
    ]
    assert app.transcript[0]["content"] == "visible query"


def test_web_command_searches_and_generates_with_bounded_context():
    app = bare_app()
    result = WebSearchResult(
        query="latest MLX release",
        sources=[WebSource("MLX", "https://example.com/mlx", "Release details")],
    )
    calls = []

    class Status:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    app.web = SimpleNamespace(search=lambda query: calls.append(("search", query)) or result)
    app.ui = SimpleNamespace(
        console=SimpleNamespace(status=lambda _message: Status()),
        info=lambda message: calls.append(("info", message)),
    )
    app._generate = lambda prompt, **kwargs: calls.append(("generate", prompt, kwargs)) or True

    app._cmd_web(["latest", "MLX", "release"])

    assert calls[0] == ("search", "latest MLX release")
    assert calls[1][0] == "info"
    _, prompt, kwargs = calls[2]
    assert prompt == "latest MLX release"
    assert "https://example.com/mlx" in kwargs["api_content"]
    assert kwargs["metadata"]["web_sources"][0]["title"] == "MLX"


def test_web_command_requires_a_query():
    app = bare_app()

    with pytest.raises(MlxChatError, match="Usage: /web QUERY"):
        app._cmd_web([])


def test_custom_system_prompt_overrides_default():
    app = bare_app()
    app.settings.system_prompt = "Custom instructions"
    assert app.api_messages()[0] == {"role": "system", "content": "Custom instructions"}


def test_system_command_shows_effective_default():
    import io

    from rich.console import Console

    app = bare_app()
    output = io.StringIO()
    app.ui = SimpleNamespace(console=Console(file=output, width=100))
    app._cmd_system([])
    assert "System prompt (default)" in output.getvalue()
    assert "Apple Silicon MacBook" in output.getvalue()


def test_voice_transcription_is_rendered_as_user_input(monkeypatch):
    from contextlib import nullcontext

    app = bare_app()
    statuses = []
    user_messages = []
    generations = []
    app.ui = SimpleNamespace(
        audio_status=lambda text: statuses.append(text) or nullcontext(),
        run_audio=lambda operation: operation(None),
        run_native=lambda operation: operation(),
        user_message=user_messages.append,
    )
    monkeypatch.setattr(chat, "record_microphone", lambda **kw: "audio")
    monkeypatch.setattr(chat, "transcribe_audio", lambda audio, **kw: "My spoken question")
    app._generate = lambda text, **kw: generations.append(text)
    app._cmd_voice([])
    assert statuses == ["/voice Listening… F2 to finish · up to 30s", "/voice Transcribing…"]
    assert user_messages == ["My spoken question"]
    assert generations == ["My spoken question"]
