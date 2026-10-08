from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from mlx_chat import cli, server
from mlx_chat.config import Settings
from mlx_chat.errors import MlxChatError, ServerUnavailableError
from mlx_chat.server import ServerRecord, StartResult, StopResult


def make_record(pid: int, model: str, *, port: int = 8080) -> ServerRecord:
    return ServerRecord(
        pid=pid,
        pgid=pid,
        model=model,
        host="127.0.0.1",
        port=port,
        started_at="2026-10-02T00:00:00+00:00",
        log_path=f"/tmp/{pid}.log",
        command=[
            "mlx_lm.server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        process_identity=f"start-{pid}",
        instance_id=f"instance-{pid}",
    )


def test_cleanup_never_stops_a_replacement_server(monkeypatch):
    settings = Settings(model="new/model")
    replacement = make_record(901, "new/model")
    replacement.instance_id = "replacement-instance"
    stopped = []

    class FakeManager:
        def __init__(self, _settings):
            pass

        def matching_record(self):
            return replacement

    monkeypatch.setattr(cli, "ServerManager", FakeManager)
    monkeypatch.setattr(cli, "stop_record", lambda *args, **kwargs: stopped.append(args[0]))

    cli._stop_new_server(
        settings,
        StartResult(
            model="new/model",
            host="127.0.0.1",
            port=8080,
            pid=901,
            log_path=None,
            reused=False,
            managed=True,
            instance_id="instance-901",
        ),
    )

    assert stopped == []


@pytest.mark.parametrize("failure", [False, True])
def test_ephemeral_chat_cleans_current_server_after_model_switch(monkeypatch, failure):
    settings = Settings(model="old/model")
    original = StartResult("old/model", settings.host, settings.port, 100, None, False, True, "old")
    switched = StartResult("new/model", settings.host, settings.port, 200, None, False, True, "new")
    cleanup = []
    monkeypatch.setattr(cli, "settings_from_args", lambda _args: settings)
    monkeypatch.setattr(cli, "ensure_server", lambda *_args, **_kwargs: original)
    monkeypatch.setattr(cli, "_stop_new_server", lambda _settings, result: cleanup.append(result))

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            self.start_result = original

        def run(self):
            settings.model = "new/model"
            self.start_result = switched
            if failure:
                raise MlxChatError("chat failed")
            return 0

    monkeypatch.setattr("mlx_chat.chat.ChatApp", FakeApp)
    args = cli.build_parser().parse_args(["chat", "--ephemeral", "--plain"])
    if failure:
        with pytest.raises(MlxChatError, match="chat failed"):
            cli.run_chat(args)
    else:
        assert cli.run_chat(args) == 0

    assert cleanup == [switched]


def test_interrupted_chat_initialization_cleans_new_server(monkeypatch):
    settings = Settings()
    result = StartResult(
        settings.model, settings.host, settings.port, 100, None, False, True, "new"
    )
    cleanup = []
    monkeypatch.setattr(cli, "settings_from_args", lambda _args: settings)
    monkeypatch.setattr(cli, "ensure_server", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        cli, "_stop_new_server", lambda _settings, result, **_kwargs: cleanup.append(result)
    )

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("mlx_chat.chat.ChatApp", interrupt)
    args = cli.build_parser().parse_args(["chat", "--plain"])

    with pytest.raises(KeyboardInterrupt):
        cli.run_chat(args)

    assert cleanup == [result]


def test_failed_model_switch_restores_previous_model(monkeypatch):
    settings = Settings(
        model="new/model",
        startup_timeout=3,
        server_args=["--new-setting"],
    )
    old_record = make_record(100, "old/model")
    old_record.server_args = ["--old-setting"]
    new_record = make_record(200, "new/model")
    restored_record = make_record(300, "old/model")
    current = {"record": old_record}
    events: list[tuple[str, object]] = []

    class FakeManager:
        def __init__(self, managed_settings):
            self.settings = managed_settings

        def matching_record(self):
            return current["record"]

        def start(self, *, progress, ready_check=None):
            events.append(("start", (self.settings.model, tuple(self.settings.server_args))))
            progress(f"starting {self.settings.model}")
            selected = new_record if self.settings.model == "new/model" else restored_record
            current["record"] = selected
            result = StartResult(
                model=selected.model,
                host=selected.host,
                port=selected.port,
                pid=selected.pid,
                log_path=None,
                reused=False,
                managed=True,
                instance_id=selected.instance_id,
            )
            try:
                if ready_check is not None:
                    ready_check(result)
            except BaseException:
                fake_stop(selected, grace=2)
                raise
            return result

    def fake_stop(item, *, grace=8.0):
        events.append(("stop", item.pid))
        if current["record"] and current["record"].pid == item.pid:
            current["record"] = None
        return StopResult(item, True, "stopped")

    class FakeClient:
        def __init__(self, client_settings):
            self.settings = client_settings

        def warmup(self, *, timeout):
            events.append(("warmup", self.settings.model))
            assert timeout == 3
            if self.settings.model == "new/model":
                raise MlxChatError("new model warmup failed")

    monkeypatch.setattr(cli, "ServerManager", FakeManager)
    monkeypatch.setattr(cli, "stop_record", fake_stop)
    monkeypatch.setattr(cli, "ChatClient", FakeClient)
    output = io.StringIO()

    with pytest.raises(MlxChatError, match="new model warmup failed"):
        cli.ensure_server(
            settings,
            warmup=True,
            console=Console(file=output, force_terminal=False),
        )

    assert events == [
        ("stop", 100),
        ("start", ("new/model", ("--new-setting",))),
        ("warmup", "new/model"),
        ("stop", 200),
        ("start", ("old/model", ("--old-setting",))),
        ("warmup", "old/model"),
    ]
    assert current["record"] == restored_record
    assert "Restored previous model: old/model" in output.getvalue()


def test_changed_server_arguments_restart_the_same_model(monkeypatch):
    settings = Settings(model="same/model", server_args=["--new-setting"])
    old_record = make_record(105, "same/model")
    old_record.server_args = ["--old-setting"]
    new_record = make_record(205, "same/model")
    new_record.server_args = ["--new-setting"]
    current = {"record": old_record}
    stopped = []

    class FakeManager:
        def __init__(self, _settings):
            pass

        def matching_record(self):
            return current["record"]

        def start(self, *, progress, ready_check=None):
            progress("starting replacement")
            current["record"] = new_record
            result = StartResult(
                model=new_record.model,
                host=new_record.host,
                port=new_record.port,
                pid=new_record.pid,
                log_path=None,
                reused=False,
                managed=True,
                instance_id=new_record.instance_id,
            )
            if ready_check is not None:
                ready_check(result)
            return result

    def fake_stop(item, *, grace=8.0):
        stopped.append(item)
        current["record"] = None
        return StopResult(item, True, "stopped")

    monkeypatch.setattr(cli, "ServerManager", FakeManager)
    monkeypatch.setattr(cli, "stop_record", fake_stop)

    result = cli.ensure_server(
        settings,
        warmup=False,
        console=Console(file=io.StringIO(), force_terminal=False),
    )

    assert stopped == [old_record]
    assert result.pid == new_record.pid


def test_interrupted_switch_stop_still_restores_previous_model(monkeypatch):
    settings = Settings(model="new/model", server_args=["--new-setting"])
    old_record = make_record(110, "old/model")
    old_record.server_args = ["--old-setting"]
    restored_record = make_record(310, "old/model")
    current = {"record": old_record}
    started_with = []

    class FakeManager:
        def __init__(self, managed_settings):
            self.settings = managed_settings

        def matching_record(self):
            return current["record"]

        def start(self, *, progress, ready_check=None):
            progress("restoring")
            started_with.append((self.settings.model, self.settings.server_args))
            current["record"] = restored_record
            result = StartResult(
                model=restored_record.model,
                host=restored_record.host,
                port=restored_record.port,
                pid=restored_record.pid,
                log_path=None,
                reused=False,
                managed=True,
                instance_id=restored_record.instance_id,
            )
            if ready_check is not None:
                ready_check(result)
            return result

    def interrupted_stop(_record, *, grace=8.0):
        current["record"] = None
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "ServerManager", FakeManager)
    monkeypatch.setattr(cli, "stop_record", interrupted_stop)

    with pytest.raises(KeyboardInterrupt):
        cli.ensure_server(
            settings,
            warmup=False,
            console=Console(file=io.StringIO(), force_terminal=False),
        )

    assert started_with == [("old/model", ["--old-setting"])]
    assert current["record"] == restored_record


def test_attach_failure_never_falls_back_to_start(monkeypatch):
    settings = Settings(model="requested/model")
    calls = []

    class FakeManager:
        def __init__(self, _settings):
            pass

        def matching_record(self):
            return None

        def attach(self, *, progress, ready_check=None):
            calls.append("attach")
            raise ServerUnavailableError("server disappeared")

        def start(self, *, progress):
            calls.append("start")
            raise AssertionError("--attach must not start a server")

    monkeypatch.setattr(cli, "ServerManager", FakeManager)

    with pytest.raises(ServerUnavailableError, match="disappeared"):
        cli.ensure_server(
            settings,
            warmup=False,
            attach_only=True,
            console=Console(file=io.StringIO(), force_terminal=False),
        )

    assert calls == ["attach"]


def test_restart_failure_restores_previous_server_arguments(monkeypatch):
    requested = Settings(model="new/model", server_args=["--new-setting"])
    old_record = make_record(120, "old/model")
    old_record.server_args = ["--old-setting"]
    restored = []

    class FakeManager:
        def __init__(self, _settings):
            pass

        def matching_record(self):
            return old_record

    monkeypatch.setattr(cli, "settings_from_args", lambda _args: requested)
    monkeypatch.setattr(cli, "ServerManager", FakeManager)
    monkeypatch.setattr(cli, "stop_record", lambda item: StopResult(item, True, "stopped"))
    monkeypatch.setattr(
        cli,
        "ensure_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(MlxChatError("restart failed")),
    )
    monkeypatch.setattr(
        cli,
        "_restore_server",
        lambda restored_settings, **_kwargs: restored.append(restored_settings),
    )

    with pytest.raises(MlxChatError, match="restart failed"):
        cli.run_restart(SimpleNamespace(no_warmup=True))

    assert len(restored) == 1
    assert restored[0].model == "old/model"
    assert restored[0].server_args == ["--old-setting"]


def test_interrupted_start_cleans_the_exact_spawned_child(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "app"))
    settings = Settings(model="new/model", port=8123, startup_timeout=1)
    manager = server.ServerManager(settings)
    stopped = []

    class FakeProcess:
        pid = 444

        def poll(self):
            raise KeyboardInterrupt

    child = FakeProcess()
    monkeypatch.setattr(manager, "matching_record", lambda: None)
    monkeypatch.setattr(server, "health", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(server, "port_is_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(server, "resolve_server_command", lambda: ["mlx_lm.server"])
    monkeypatch.setattr(server.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(server, "process_start_identity", lambda _pid: "exact-start")
    monkeypatch.setattr(server, "_replace_record", lambda _record: None)
    monkeypatch.setattr(
        server,
        "_stop_spawned_child",
        lambda process, record: stopped.append((process, record.pid, record.process_identity)),
    )

    with pytest.raises(KeyboardInterrupt):
        manager._start_locked(lambda _message: None)

    assert stopped == [(child, 444, "exact-start")]


def test_failed_warmup_is_cleaned_before_endpoint_lock_is_released(monkeypatch):
    settings = Settings(model="new/model")
    manager = server.ServerManager(settings)
    started = StartResult(
        model=settings.model,
        host=settings.host,
        port=settings.port,
        pid=555,
        log_path=None,
        reused=False,
        managed=True,
        instance_id="instance-555",
    )
    events = []

    @contextlib.contextmanager
    def fake_endpoint_lock(_port, _progress):
        events.append("lock-enter")
        try:
            yield
        finally:
            events.append("lock-exit")

    def failed_ready_check(_result):
        events.append("warmup")
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "_endpoint_lock", fake_endpoint_lock)
    monkeypatch.setattr(
        manager,
        "_start_locked",
        lambda _progress: (events.append("healthy") or started),
    )
    monkeypatch.setattr(
        manager,
        "_cleanup_failed_start",
        lambda result: events.append(("cleanup", result.instance_id)),
    )

    with pytest.raises(KeyboardInterrupt):
        manager.start(ready_check=failed_ready_check)

    assert events == [
        "lock-enter",
        "healthy",
        "warmup",
        ("cleanup", "instance-555"),
        "lock-exit",
    ]
