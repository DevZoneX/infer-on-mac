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
