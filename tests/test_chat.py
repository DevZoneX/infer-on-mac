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
    app.pending_attachments = []
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
    assert "Apple Silicon Mac" in output.getvalue()


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
    assert statuses == ["/voice Listening… F3 to finish · up to 30s", "/voice Transcribing…"]
    assert user_messages == ["My spoken question"]
    assert generations == ["My spoken question"]


def test_redirect_includes_cancelled_answer_and_new_message_in_context():
    app = bare_app()
    requests = []
    results = iter(
        [
            GenerationResult(answer="Partial answer", finish_reason="cancelled"),
            GenerationResult(answer="Answer to the new direction", finish_reason="stop"),
        ]
    )
    app.client = SimpleNamespace(
        collect=lambda messages, **kw: (requests.append(messages.copy()), next(results))[1]
    )
    app.ui.render_generation = lambda generate, **kw: generate(lambda *args: None)
    assert app._generate("Original question")
    assert app.last_result.answer == "Partial answer"
    assert app.transcript[-1]["finish_reason"] == "cancelled"
    assert app._generate("New direction")
    assert requests[-1][-2]["content"] == "Partial answer"
    assert requests[-1][-1]["content"] == "New direction"


@pytest.mark.parametrize("command", ["/stop", "/exit", "/quit"])
def test_quit_commands_do_not_open_a_second_popup(command):
    app = bare_app()
    app.ui.command_popup = lambda title: pytest.fail("Quitting must not require dismissing a popup")
    dispatched = []
    app._dispatch_command = dispatched.append
    app._handle_command(command)
    assert dispatched == [command]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "99"])
def test_invalid_numeric_command_preserves_active_settings(value):
    app = bare_app()
    original = app.settings.temperature

    with pytest.raises(MlxChatError):
        app._set_number("temperature", [value], float, 0, 5)

    assert app.settings.temperature == original
    assert app.api_messages()[-1]["content"] == "keep me"


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": ["invalid"]},
        {"messages": [{"role": [], "content": "invalid"}]},
        {"messages": [{"role": "user", "content": 123}]},
        {"messages": [{"role": "user", "content": "hi", "api_content": []}]},
        {"messages": [], "settings": []},
        {"messages": [], "system_prompt": []},
    ],
)
def test_invalid_session_preserves_current_conversation(monkeypatch, payload):
    app = bare_app()
    original = app.transcript
    previous_result = app.last_result
    monkeypatch.setattr(chat, "load_session", lambda _name: payload)

    with pytest.raises(MlxChatError, match="not valid"):
        app._load("broken")

    assert app.transcript is original
    assert app.last_result is previous_result
    assert app.session_name == "current"


@pytest.mark.parametrize("healthy", [True, False])
def test_chat_status_reports_endpoint_health(monkeypatch, healthy):
    import io

    from rich.console import Console

    app = bare_app()
    app.manager = SimpleNamespace(matching_record=lambda: None)
    output = io.StringIO()
    app.ui.console = Console(file=output, width=100)
    monkeypatch.setattr(chat, "health", lambda host, port: healthy)

    app._cmd_status()

    assert ("running" if healthy else "unavailable") in output.getvalue()


@pytest.mark.parametrize("preload", [False, True])
def test_voice_preload_is_optional_at_chat_startup(monkeypatch, preload):
    from mlx_chat.server import StartResult

    loaded = []
    monkeypatch.setattr(chat, "TerminalUI", lambda *_args, **_kw: SimpleNamespace())
    monkeypatch.setattr(chat.ChatApp, "_preload_voice", lambda _self: loaded.append("voice"))
    settings = Settings(preload_voice=preload)
    start = StartResult(settings.model, settings.host, settings.port, None, None, True, False)
    app = chat.ChatApp(settings, start)
    assert loaded == (["voice"] if preload else [])
    assert app.transcript == []


def test_load_restores_latest_result_and_visible_transcript(monkeypatch):
    app = bare_app()
    rendered = []
    messages = [
        {"role": "user", "content": "Saved question"},
        {
            "role": "assistant",
            "content": "Saved answer",
            "usage": {"completion_tokens": 20},
            "metrics": {"elapsed": 4, "first_token_seconds": 1},
        },
    ]
    monkeypatch.setattr(chat, "load_session", lambda _name: {"messages": messages})
    app.ui.set_context = lambda name, count: rendered.append((name, count))
    app.ui.restore_transcript = lambda value: rendered.append(value)
    app._load("saved")
    assert app.last_result.answer == "Saved answer"
    assert app.last_result.tokens_per_second == 5
    assert rendered == [("saved", 2), messages]
    assert app.session_name == "saved"


def test_undo_restores_previous_result_and_speech_target():
    app = bare_app()
    app.transcript = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer", "reasoning": "First reasoning"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]
    restored = []
    app.ui.restore_transcript = restored.append
    app._cmd_undo()
    assert app.last_result.answer == "First answer"
    assert app.last_result.reasoning == "First reasoning"
    assert restored == [app.transcript]
    assert len(app.transcript) == 2


def test_generation_warns_about_omitted_context_but_preserves_full_transcript():
    app = bare_app()
    app.settings = Settings(system_prompt="Short instructions", context_tokens=600, max_tokens=100)
    app.transcript = [
        {"role": "user", "content": "Old question " * 100},
        {"role": "assistant", "content": "Old answer"},
    ]
    warnings = []
    requests = []
    app.ui.warning = warnings.append
    app.ui.render_generation = lambda generate, **kw: generate(lambda *args: None)
    app.client = SimpleNamespace(
        collect=lambda messages, **kw: requests.append(messages) or GenerationResult(answer="new")
    )
    assert app._generate("Latest question")
    assert requests[-1] == [
        {"role": "system", "content": "Short instructions"},
        {"role": "user", "content": "Latest question"},
    ]
    assert len(app.transcript) == 4
    assert "2 older messages omitted" in warnings[0]


def test_invalid_saved_metrics_do_not_replace_current_session(monkeypatch):
    app = bare_app()
    original = app.transcript
    previous = app.last_result
    monkeypatch.setattr(
        chat,
        "load_session",
        lambda _name: {
            "messages": [{"role": "assistant", "content": "bad", "metrics": {"elapsed": -1}}]
        },
    )
    with pytest.raises(MlxChatError, match="elapsed"):
        app._load("broken")
    assert app.transcript is original
    assert app.last_result is previous


def test_speak_and_stats_work_immediately_after_loading(monkeypatch):
    import io
    from contextlib import nullcontext

    from rich.console import Console

    app = bare_app()
    spoken = []
    output = io.StringIO()
    monkeypatch.setattr(
        chat,
        "load_session",
        lambda _name: {
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "Restored answer",
                    "usage": {"completion_tokens": 6},
                    "metrics": {"elapsed": 2, "first_token_seconds": 0.5},
                },
            ]
        },
    )
    monkeypatch.setattr(chat, "speak", lambda text, **kw: spoken.append(text) or False)
    app.ui.set_context = lambda *_args: None
    app.ui.restore_transcript = lambda _messages: None
    app.ui.console = Console(file=output, width=100)
    app.ui.audio_status = lambda _message: nullcontext()
    app.ui.run_audio = lambda operation, **kw: operation(None)
    app.ui.info = lambda _message: None
    app._load("saved")
    app._cmd_speak([])
    app._cmd_stats()
    assert spoken == ["Restored answer"]
    assert "2.000 s" in output.getvalue()
    assert "3.00 token/s" in output.getvalue()


def test_attached_context_is_sent_once_and_reused_on_retry(tmp_path):
    app = bare_app()
    path = tmp_path / "code.py"
    path.write_text("print('snapshot')")
    notices = []
    app.ui.info = notices.append
    app._cmd_attach([str(path)])
    assert "Selected file excerpts" in notices[-1]
    requests = []
    app.client = SimpleNamespace(
        collect=lambda messages, **kw: (
            requests.append(messages),
            GenerationResult(answer="explanation"),
        )[1]
    )
    app.ui.render_generation = lambda generate, **kw: generate(lambda *args: None)
    assert app._generate("Explain my code")
    assert "print('snapshot')" in requests[-1][-1]["content"]
    assert app.transcript[-2]["content"] == "Explain my code"
    assert app.pending_attachments == []
    path.write_text("changed since first prompt")
    app._cmd_retry()
    assert "print('snapshot')" in requests[-1][-1]["content"]
    assert "changed since" not in requests[-1][-1]["content"]
    assert app._generate("Another question")
    assert requests[-1][-1]["content"] == "Another question"


def test_failed_generation_keeps_pending_attachments_for_next_attempt(tmp_path):
    app = bare_app()
    path = tmp_path / "note.txt"
    path.write_text("context")
    app.ui.info = lambda message: None
    app._cmd_attach([str(path)])
    queued = app.pending_attachments
    app.ui.error = lambda message: None
    app.ui.render_generation = lambda *args, **kw: (_ for _ in ()).throw(MlxChatError("offline"))
    assert not app._generate("Question")
    assert app.pending_attachments is queued
    assert app.transcript == [{"role": "user", "content": "keep me"}]


def test_failed_attachment_budget_check_preserves_queue(tmp_path):
    app = bare_app()
    app.settings = Settings(system_prompt="small", context_tokens=600, max_tokens=100)
    path = tmp_path / "note.txt"
    path.write_text("x" * 300)
    with pytest.raises(MlxChatError, match="context budget"):
        app._cmd_attach([str(path)])
    assert app.pending_attachments == []


def test_profile_command_applies_and_saves_generation_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "home"))
    app = bare_app()
    app.ui.info = lambda message: None
    app._cmd_profile(["coding"])
    assert app.settings.temperature == 0.2
    app.settings.temperature = 0.35
    app._cmd_profile(["save", "mine"])
    app._cmd_profile(["writing"])
    assert app.settings.temperature == 0.7
    app._cmd_profile(["mine"])
    assert app.settings.temperature == 0.35
    assert "practical coding help" in app.settings.system_prompt
    assert app._session_payload()["settings"]["repetition_penalty"] == 1.0


def test_profile_with_too_small_context_preserves_settings_and_attachments(tmp_path, monkeypatch):
    from mlx_chat.attachments import Attachment
    from mlx_chat.profiles import save_profile

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "home"))
    app = bare_app()
    app.pending_attachments = [Attachment("file", "x" * 1000)]
    save_profile("small", Settings(system_prompt="small", context_tokens=600, max_tokens=100))
    with pytest.raises(MlxChatError, match="context budget"):
        app._cmd_profile(["small"])
    assert app.settings.context_tokens == 32768
    assert len(app.pending_attachments) == 1


def test_new_and_reset_clear_pending_file_context():
    from mlx_chat.attachments import Attachment

    app = bare_app()
    app.pending_attachments = [Attachment("file", "context")]
    app._cmd_new([])
    assert app.pending_attachments == []
    app.pending_attachments = [Attachment("file", "context")]
    app._dispatch_command("/reset")
    assert app.pending_attachments == []


def test_new_rejects_existing_session_without_overwriting(tmp_path, monkeypatch):
    from mlx_chat.config import load_session, save_session

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "home"))
    saved = {"messages": [{"role": "user", "content": "keep saved text"}]}
    save_session("existing", saved)
    app = bare_app()
    original = app.transcript
    with pytest.raises(MlxChatError, match="already exists"):
        app._cmd_new(["existing"])
    assert app.transcript is original
    assert app.session_name == "current"
    assert load_session("existing") == saved


def test_startup_cli_generation_overrides_win_over_saved_session(tmp_path, monkeypatch):
    from mlx_chat.attachments import Attachment
    from mlx_chat.config import save_session
    from mlx_chat.server import StartResult

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "home"))
    save_session(
        "saved",
        {
            "messages": [],
            "system_prompt": "Saved instructions",
            "settings": {"temperature": 0.9, "concise": False},
        },
    )
    monkeypatch.setattr(chat, "TerminalUI", lambda *args, **kw: SimpleNamespace())
    settings = Settings()
    start = StartResult(settings.model, settings.host, settings.port, None, None, True, False)
    app = chat.ChatApp(
        settings,
        start,
        session_name="saved",
        generation_overrides={"temperature": 0.2, "concise": True},
    )
    assert app.settings.temperature == 0.2
    assert app.settings.concise is True
    assert app.settings.system_prompt == "Saved instructions"
    assert app.pending_attachments == []
    loaded = chat.ChatApp(
        Settings(), start, session_name="saved", attachments=[Attachment("file", "context")]
    )
    assert len(loaded.pending_attachments) == 1


def test_load_clears_pending_files_but_keeps_saved_snapshots(tmp_path, monkeypatch):
    from mlx_chat.attachments import Attachment
    from mlx_chat.config import save_session

    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "home"))
    save_session(
        "saved",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "question",
                    "api_content": "snapshot",
                    "attachments": ["file.py"],
                }
            ]
        },
    )
    app = bare_app()
    app.pending_attachments = [Attachment("new-file", "new content")]
    app._load("saved", announce=False)
    assert app.pending_attachments == []
    assert app.transcript[0]["api_content"] == "snapshot"


def test_save_uses_a_canonical_session_name(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "home"))
    app = bare_app()
    path = app._save("  notes  ")
    assert app.session_name == "notes"
    assert path.name == "notes.json"
