from __future__ import annotations

import json

import pytest

from mlx_chat import DEFAULT_MODEL
from mlx_chat.config import (
    Settings,
    config_path,
    history_path,
    list_sessions,
    load_session,
    load_settings,
    normalize_session_name,
    save_session,
    save_settings,
)
from mlx_chat.errors import MlxChatError


@pytest.fixture(autouse=True)
def isolated_app_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "app-home"))


def test_settings_defaults_and_base_urls():
    settings = Settings()

    assert settings.model == DEFAULT_MODEL
    assert settings.server_args == []
    assert settings.base_url == "http://127.0.0.1:8080"
    assert Settings(host="0.0.0.0", port=9000).base_url == "http://127.0.0.1:9000"
    assert Settings(host="::", port=9001).base_url == "http://127.0.0.1:9001"
    assert Settings(host="::1", port=9002).base_url == "http://[::1]:9002"


def test_merged_ignores_none_and_does_not_mutate_original():
    original = Settings(model="original", port=8080, server_args=["--foo"])

    merged = original.merged(model="replacement", port=None)
    merged.server_args.append("--bar")

    assert merged.model == "replacement"
    assert merged.port == 8080
    assert original.model == "original"
    assert original.server_args == ["--foo"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "  "}, "Model name"),
        ({"host": ""}, "Host"),
        ({"port": 0}, "Port"),
        ({"port": 65536}, "Port"),
        ({"temperature": -0.01}, "Temperature"),
        ({"temperature": 5.01}, "Temperature"),
        ({"top_p": 1.01}, "top_p"),
        ({"top_k": -1}, "top_k"),
        ({"min_p": -0.01}, "min_p"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"repetition_penalty": -0.01}, "repetition_penalty"),
        ({"startup_timeout": 0}, "Timeouts"),
        ({"request_timeout": 0}, "Timeouts"),
        ({"web_results": 0}, "web_results"),
        ({"web_results": 11}, "web_results"),
        ({"web_timeout": 0}, "web_timeout"),
        ({"web_timeout": 61}, "web_timeout"),
        ({"web_region": ""}, "web_region"),
        ({"web_safesearch": "maximum"}, "web_safesearch"),
    ],
)
def test_settings_validation_rejects_out_of_range_values(overrides, message):
    with pytest.raises(MlxChatError, match=message):
        Settings(**overrides)


def test_settings_boundary_values_are_accepted():
    settings = Settings(
        port=65535,
        temperature=5,
        top_p=1,
        top_k=0,
        min_p=1,
        max_tokens=1,
        repetition_penalty=0,
        startup_timeout=1,
        request_timeout=1,
    )

    assert settings.port == 65535


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": 7},
        {"host": []},
        {"port": "8080"},
        {"think": "false"},
        {"temperature": True},
        {"top_p": "0.5"},
        {"top_k": 2.5},
        {"max_tokens": False},
        {"system_prompt": 3},
        {"server_args": "--foo"},
        {"server_args": ["ok", 2]},
        {"web_results": True},
        {"web_timeout": "10"},
    ],
)
def test_settings_validation_rejects_wrong_types(overrides):
    with pytest.raises(MlxChatError):
        Settings(**overrides)


@pytest.mark.parametrize("flag", ["--model", "--mod", "--host=x", "--hos", "--por"])
def test_server_args_cannot_override_managed_lifecycle(flag):
    with pytest.raises(MlxChatError, match="own option"):
        Settings(server_args=[flag])


def test_settings_round_trip_is_atomic_and_unknown_keys_are_ignored():
    settings = Settings(model="local/model", port=8123, think=True, server_args=["--draft-model=x"])
    save_settings(settings)

    saved = json.loads(config_path().read_text(encoding="utf-8"))
    saved["future_option"] = "ignored"
    config_path().write_text(json.dumps(saved), encoding="utf-8")

    assert load_settings() == settings
    assert not list(config_path().parent.glob(".config.json.*"))


def test_missing_config_returns_defaults():
    assert load_settings() == Settings()


@pytest.mark.parametrize("contents", ["not-json", "[]", '{"port": "not-an-int"}'])
def test_invalid_config_is_reported_with_its_path(contents):
    config_path().parent.mkdir(parents=True)
    config_path().write_text(contents, encoding="utf-8")

    with pytest.raises(MlxChatError) as caught:
        load_settings()

    assert str(config_path()) in str(caught.value)


@pytest.mark.parametrize("name", ["chat", "A-1_b.c", "9", "x" * 64])
def test_session_name_accepts_safe_names(name):
    assert normalize_session_name(f"  {name}  ") == name


@pytest.mark.parametrize(
    "name",
    ["", " ", ".hidden", "-leading", "../escape", "a/b", "a b", "x" * 65, "💬"],
)
def test_session_name_rejects_traversal_and_unsafe_names(name):
    with pytest.raises(MlxChatError, match="Session names"):
        normalize_session_name(name)


def test_session_round_trip_and_sorted_listing():
    payload = {"messages": [{"role": "user", "content": "hello"}], "model": "m"}
    second_path = save_session("z-last", payload)
    first_path = save_session("a-first", {"messages": []})

    assert load_session("z-last") == payload
    assert list_sessions() == [("a-first", first_path), ("z-last", second_path)]
    assert second_path.parent.name == "sessions"


def test_load_session_rejects_missing_malformed_and_wrong_schema():
    with pytest.raises(MlxChatError, match="No saved session"):
        load_session("missing")

    session_dir = config_path().parent / "sessions"
    session_dir.mkdir(parents=True)
    (session_dir / "broken.json").write_text("{", encoding="utf-8")
    (session_dir / "wrong.json").write_text('{"messages": {}}', encoding="utf-8")

    with pytest.raises(MlxChatError, match="Could not load session"):
        load_session("broken")
    with pytest.raises(MlxChatError, match="is not valid"):
        load_session("wrong")


def test_history_path_creates_parent_directory():
    path = history_path()

    assert path.name == "prompt_history"
    assert path.parent.is_dir()
