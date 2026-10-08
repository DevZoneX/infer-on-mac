from __future__ import annotations

import argparse

import pytest

from mlx_chat import DEFAULT_MODEL, MODEL_PRESETS, cli
from mlx_chat.config import Settings
from mlx_chat.errors import MlxChatError


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ["chat"]),
        (["other/model"], ["chat", "other/model"]),
        (["--think"], ["chat", "--think"]),
        (["chat", "other/model"], ["chat", "other/model"]),
        (["run", "hello"], ["run", "hello"]),
        (["start"], ["start"]),
        (["--help"], ["--help"]),
        (["--version"], ["--version"]),
    ],
)
def test_normalize_argv(argv, expected):
    assert cli.normalize_argv(argv) == expected


def parse_runtime(*argv: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(cli.normalize_argv(argv))


def test_parser_supports_default_short_form_and_explicit_model_flag():
    default = parse_runtime()
    positional = parse_runtime("org/positional", "--port", "8123")
    flagged = parse_runtime("org/positional", "--model-name", "org/flagged")

    assert default.command == "chat"
    assert default.model is None
    assert positional.model == "org/positional"
    assert positional.port == 8123
    assert flagged.model_flag == "org/flagged"


def test_settings_resolution_precedence_and_cli_overrides(monkeypatch):
    persisted = Settings(model="persisted", port=7000, temperature=0.2, think=False)
    monkeypatch.setattr(cli, "load_settings", lambda: persisted)
    monkeypatch.setenv("MLX_CHAT_MODEL", "from-env")

    env_settings = cli.settings_from_args(parse_runtime("--top-p", "0.42"))
    positional = cli.settings_from_args(parse_runtime("positional"))
    flagged = cli.settings_from_args(parse_runtime("positional", "-m", "flagged", "--think"))

    assert env_settings.model == "from-env"
    assert env_settings.port == 7000
    assert env_settings.temperature == 0.2
    assert env_settings.top_p == 0.42
    assert positional.model == "positional"
    assert flagged.model == "flagged"
    assert flagged.think is True


def test_settings_resolution_uses_built_in_default(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", Settings)
    monkeypatch.delenv("MLX_CHAT_MODEL", raising=False)

    assert cli.settings_from_args(parse_runtime()).model == DEFAULT_MODEL


@pytest.mark.parametrize("preset", ["tiny", "small", "medium", "large"])
def test_model_size_presets_resolve_at_launch(monkeypatch, preset):
    monkeypatch.setattr(cli, "load_settings", Settings)

    assert cli.settings_from_args(parse_runtime(preset)).model == MODEL_PRESETS[preset]


@pytest.mark.parametrize("argument", ["--model=x", "--host=localhost", "--port=9999"])
def test_mlx_args_cannot_override_lifecycle_fields(monkeypatch, argument):
    monkeypatch.setattr(cli, "load_settings", Settings)
    args = parse_runtime(f"--mlx-arg={argument}")

    with pytest.raises(MlxChatError, match="own option"):
        cli.settings_from_args(args)


def test_other_mlx_args_are_forwarded(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", Settings)
    args = parse_runtime("--mlx-arg=--trust-remote-code", "--mlx-arg=--adapter-path=adapter")

    settings = cli.settings_from_args(args)

    assert settings.server_args == ["--trust-remote-code", "--adapter-path=adapter"]


def test_non_loopback_bind_requires_explicit_acknowledgement(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", Settings)

    with pytest.raises(MlxChatError, match="Refusing to bind"):
        cli.settings_from_args(parse_runtime("--host", "0.0.0.0"))

    settings = cli.settings_from_args(parse_runtime("--host", "0.0.0.0", "--allow-network"))
    assert settings.host == "0.0.0.0"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_bind_does_not_need_network_ack(monkeypatch, host):
    monkeypatch.setattr(cli, "load_settings", Settings)

    assert cli.settings_from_args(parse_runtime("--host", host)).host == host


def test_persisted_non_loopback_setting_is_also_guarded(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(host="192.168.1.20"))

    with pytest.raises(MlxChatError, match="--allow-network"):
        cli.settings_from_args(parse_runtime())


@pytest.mark.parametrize("key", ["base-url", "merged", "validate"])
def test_config_set_rejects_attributes_that_are_not_settings(monkeypatch, key):
    monkeypatch.setattr(cli, "load_settings", Settings)
    monkeypatch.setattr(cli, "save_settings", lambda _settings: pytest.fail("Must not save"))
    args = parse_runtime("config", "set", key, "invalid")

    with pytest.raises(MlxChatError, match="Unknown setting"):
        cli.run_config(args)


@pytest.mark.parametrize("system_prompt", ["", "Custom instructions"])
def test_run_once_honors_concise_mode_and_system_prompt(monkeypatch, system_prompt):
    from mlx_chat.client import GenerationResult
    from mlx_chat.context import CONCISE_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT

    settings = Settings(think=True, concise=True, system_prompt=system_prompt)
    requests = []
    monkeypatch.setattr(cli, "settings_from_args", lambda _args: settings)
    monkeypatch.setattr(cli, "ensure_server", lambda *_args, **_kwargs: None)

    class FakeClient:
        def __init__(self, _settings):
            pass

        def collect(self, messages, *, think):
            requests.append((messages, think))
            return GenerationResult(answer="Brief answer")

    monkeypatch.setattr(cli, "ChatClient", FakeClient)

    assert cli.run_once(parse_runtime("run", "hello", "--concise")) == 0
    assert requests == [
        (
            [
                {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
                {"role": "system", "content": CONCISE_SYSTEM_PROMPT},
                {"role": "user", "content": "hello"},
            ],
            False,
        )
    ]


def test_new_runtime_options_override_saved_configuration(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(preload_voice=True))
    settings = cli.settings_from_args(
        parse_runtime(
            "--context-tokens",
            "8192",
            "--max-tokens",
            "512",
            "--connect-timeout",
            "3",
            "--no-preload-voice",
        )
    )
    assert settings.context_tokens == 8192
    assert settings.connect_timeout == 3
    assert settings.max_tokens == 512
    assert settings.preload_voice is False


def test_oversized_one_off_prompt_is_rejected_before_server_start(monkeypatch):
    monkeypatch.setattr(
        cli,
        "settings_from_args",
        lambda _args: Settings(
            system_prompt="small",
            context_tokens=600,
            max_tokens=100,
        ),
    )
    monkeypatch.setattr(cli, "ensure_server", lambda *a, **kw: pytest.fail("Must not start"))
    with pytest.raises(MlxChatError, match="context budget"):
        cli.run_once(parse_runtime("run", "x" * 1000))


def test_cli_import_does_not_load_optional_backends():
    import json
    import subprocess
    import sys

    program = (
        "import sys,json; import mlx_chat.cli; "
        "print(json.dumps([name for name in "
        "('ddgs', 'mlx_whisper', 'kokoro_mlx', 'mlx.core', 'torch', 'mlx_chat.ui', "
        "'prompt_toolkit') if name in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
