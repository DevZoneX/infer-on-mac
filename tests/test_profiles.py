from __future__ import annotations

import json

import pytest

from mlx_chat.config import Settings
from mlx_chat.errors import MlxChatError
from mlx_chat.profiles import (
    apply_profile,
    delete_profile,
    list_profiles,
    profile_path,
    save_profile,
)


@pytest.fixture(autouse=True)
def isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "home"))


def test_builtins_apply_generation_settings_without_changing_server_configuration():
    original = Settings(model="active/model", port=8123, server_args=["--trust-remote-code"])
    coding = apply_profile("coding", original)
    assert coding.temperature == 0.2
    assert "practical coding help" in coding.system_prompt
    assert (coding.model, coding.port, coding.server_args) == (
        original.model,
        original.port,
        original.server_args,
    )
    writing = apply_profile("writing", coding)
    assert "writing and editing" in writing.system_prompt
    assert writing.temperature == 0.7
    default = apply_profile("default", writing)
    assert default.system_prompt == ""
    assert original.system_prompt == ""


def test_custom_profile_round_trip_and_delete():
    saved = Settings(
        system_prompt="Custom instructions",
        concise=True,
        max_tokens=512,
        temperature=0.3,
        repetition_penalty=1.1,
    )
    path = save_profile("my-coding", saved)
    restored = apply_profile("my-coding", Settings(model="keep-model"))
    assert restored.model == "keep-model"
    assert restored.system_prompt == saved.system_prompt
    assert restored.max_tokens == 512
    assert restored.concise
    assert restored.repetition_penalty == 1.1
    assert "model" not in json.loads(path.read_text())
    assert list_profiles() == ["coding", "default", "my-coding", "writing"]
    delete_profile("my-coding")
    assert not path.exists()


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        "{broken",
        '{"model":"other"}',
        '{"temperature":99}',
        '{"think":null}',
        '{"think":"on"}',
        '{"system_prompt":[]}',
    ],
)
def test_invalid_profiles_do_not_mutate_active_settings(contents):
    path = profile_path("broken")
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    settings = Settings(system_prompt="keep", temperature=0.4)
    with pytest.raises(MlxChatError):
        apply_profile("broken", settings)
    assert settings.system_prompt == "keep"
    assert settings.temperature == 0.4


@pytest.mark.parametrize("name", ["../escape", "bad name", ""])
def test_profile_names_cannot_escape_storage(name):
    with pytest.raises(MlxChatError, match="Profile names"):
        save_profile(name, Settings())


def test_builtin_profiles_are_read_only_and_missing_profiles_are_reported():
    for name in ("coding", "writing", "default"):
        with pytest.raises(MlxChatError, match="built-in"):
            save_profile(name, Settings())
        with pytest.raises(MlxChatError, match="Built-in"):
            delete_profile(name)
    with pytest.raises(MlxChatError, match="No profile"):
        apply_profile("missing", Settings())
    with pytest.raises(MlxChatError, match="No saved profile"):
        delete_profile("missing")
