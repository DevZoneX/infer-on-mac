from __future__ import annotations

import json
import signal
from dataclasses import asdict

import pytest

from mlx_chat import server
from mlx_chat.server import ServerRecord


@pytest.fixture(autouse=True)
def isolated_app_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "app-home"))


def record(pid: int, *, port: int = 8080, command: list[str] | None = None) -> ServerRecord:
    return ServerRecord(
        pid=pid,
        pgid=pid,
        model=f"model-{pid}",
        host="127.0.0.1",
        port=port,
        started_at="2026-01-02T03:04:05+00:00",
        log_path=f"/tmp/server-{pid}.log",
        command=command or ["mlx_lm.server"],
        process_identity=f"start-{pid}",
        instance_id=f"instance-{pid}",
    )


def write_registry(*records: ServerRecord, extra=None):
    server.registry_path().parent.mkdir(parents=True, exist_ok=True)
    values = [asdict(item) for item in records]
    if extra is not None:
        values.append(extra)
    server.registry_path().write_text(
        json.dumps({"version": 1, "servers": values}),
        encoding="utf-8",
    )


def test_server_record_from_dict_coerces_serialized_fields():
    item = ServerRecord.from_dict(
        {
            "pid": "42",
            "model": 123,
            "host": "localhost",
            "port": "9000",
            "started_at": 456,
            "log_path": "/tmp/log",
            "command": ["python", 7],
        }
    )

    assert item.pid == 42
    assert item.pgid == 42
    assert item.model == "123"
    assert item.port == 9000
    assert item.command == ["python", "7"]


def test_registry_reader_tolerates_corruption_and_skips_bad_records():
    server.registry_path().parent.mkdir(parents=True)
    server.registry_path().write_text("not json", encoding="utf-8")
    assert server._read_registry_unlocked() == []

    good = record(20)
    write_registry(good, extra={"pid": "missing required fields"})
    assert server._read_registry_unlocked() == [good]


def test_list_records_prunes_stale_or_recycled_pids(monkeypatch):
    live = record(20)
    stale = record(21, port=8081)
    write_registry(live, stale)
    monkeypatch.setattr(server, "record_matches_process", lambda item: item.pid == live.pid)

    assert server.list_records() == [live]
    persisted = json.loads(server.registry_path().read_text(encoding="utf-8"))
    assert [item["pid"] for item in persisted["servers"]] == [live.pid]


def test_list_records_can_report_stale_snapshot_while_cleaning_registry(monkeypatch):
    live = record(30)
    stale = record(31, port=8081)
    write_registry(live, stale)
    monkeypatch.setattr(server, "record_matches_process", lambda item: item.pid == live.pid)

    assert server.list_records(include_stale=True) == [live, stale]
    assert server._read_registry_unlocked() == [live]


def test_process_alive_never_probes_init_or_invalid_pids(monkeypatch):
    calls = []
    monkeypatch.setattr(server.os, "kill", lambda *args: calls.append(args))

    assert server.process_alive(-3) is False
    assert server.process_alive(0) is False
    assert server.process_alive(1) is False
    assert calls == []


def test_record_matching_requires_a_recognizable_mlx_server_command(monkeypatch):
    item = record(40)
    monkeypatch.setattr(server.os, "getpgid", lambda _pid: 40)
    monkeypatch.setattr(server, "process_start_identity", lambda _pid: "start-40")
    monkeypatch.setattr(server, "process_command", lambda _pid: "python -m unrelated.service")
    assert server.record_matches_process(item) is False

    monkeypatch.setattr(
        server,
        "process_command",
        lambda _pid: ("python -m mlx_lm server --model model-40 --host 127.0.0.1 --port 8080"),
    )
    assert server.record_matches_process(item) is True

    monkeypatch.setattr(
        server,
        "process_command",
        lambda _pid: ("/venv/bin/mlx_lm.server --model model-40 --host 127.0.0.1 --port 8080"),
    )
    assert server.record_matches_process(item) is True


def test_record_matching_rejects_identityless_and_prefix_option_matches(monkeypatch):
    item = record(42, port=808)
    item.model = "model-4"
    command = "mlx_lm.server --model model-40 --host 127.0.0.1 --port 8080"
    monkeypatch.setattr(server.os, "getpgid", lambda _pid: 42)
    monkeypatch.setattr(server, "process_command", lambda _pid: command)
    monkeypatch.setattr(server, "process_start_identity", lambda _pid: "start-42")

    item.process_identity = ""
    assert server.record_matches_process(item) is False

    item.process_identity = "start-42"
    assert server.record_matches_process(item) is False


def test_record_matching_rejects_changed_process_group_or_start_identity(monkeypatch):
    item = record(41)
    item.process_identity = "original-start"
    command = "mlx_lm.server --model model-41 --host 127.0.0.1 --port 8080"
    monkeypatch.setattr(server, "process_command", lambda _pid: command)
    monkeypatch.setattr(server.os, "getpgid", lambda _pid: 99)
    assert server.record_matches_process(item) is False

    monkeypatch.setattr(server.os, "getpgid", lambda _pid: 41)
    monkeypatch.setattr(server, "process_start_identity", lambda _pid: "different-start")
    assert server.record_matches_process(item) is False

    monkeypatch.setattr(server, "process_start_identity", lambda _pid: "original-start")
    assert server.record_matches_process(item) is True


def test_registry_removal_uses_instance_token_not_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "app-home"))
    original = record(43)
    replacement = record(43)
    replacement.instance_id = "replacement-instance"
    replacement.started_at = "2026-01-02T03:04:06+00:00"
    write_registry(replacement)

    server._remove_record(original)

    assert server._read_registry_unlocked() == [replacement]


def test_stop_record_does_not_signal_a_recycled_pid(monkeypatch):
    item = record(50)
    removed = []
    signals = []
    monkeypatch.setattr(server, "process_alive", lambda _pid: True)
    monkeypatch.setattr(server, "record_matches_process", lambda _record: False)
    monkeypatch.setattr(server, "_remove_record", lambda item: removed.append(item.pid))
    monkeypatch.setattr(server.os, "killpg", lambda *args: signals.append(("group", args)))
    monkeypatch.setattr(server.os, "kill", lambda *args: signals.append(("pid", args)))

    result = server.stop_record(item)

    assert result.stopped is False
    assert result.detail == "PID was reused; left the unrelated process untouched"
    assert removed == [50]
    assert signals == []


def test_stop_record_cleans_up_an_already_stopped_process(monkeypatch):
    item = record(60)
    removed = []
    monkeypatch.setattr(server, "process_alive", lambda _pid: False)
    monkeypatch.setattr(server, "_remove_record", lambda item: removed.append(item.pid))

    result = server.stop_record(item)

    assert result.detail == "already stopped"
    assert result.stopped is False
    assert removed == [60]


def test_stop_record_prefers_process_group_sigint_and_cleans_registry(monkeypatch):
    item = record(70)
    alive = iter([True, False, False, False])
    removed = []
    signals = []
    times = iter([0.0, 2.0])
    monkeypatch.setattr(server, "process_alive", lambda _pid: next(alive))
    monkeypatch.setattr(server, "record_matches_process", lambda _record: True)
    monkeypatch.setattr(server, "_remove_record", lambda item: removed.append(item.pid))
    monkeypatch.setattr(server.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(server.time, "monotonic", lambda: next(times))

    result = server.stop_record(item, grace=1)

    assert result.stopped is True
    assert result.detail == "stopped"
    assert signals == [(70, signal.SIGINT)]
    assert removed == [70]


def test_stop_record_escalates_to_sigterm_and_sigkill(monkeypatch):
    item = record(80)
    signals = []
    times = iter([0.0, 1.0, 2.0, 3.0, 4.0, 7.0])
    monkeypatch.setattr(server, "process_alive", lambda _pid: True)
    monkeypatch.setattr(server, "record_matches_process", lambda _record: True)
    monkeypatch.setattr(server.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(server.time, "monotonic", lambda: next(times))

    result = server.stop_record(item, grace=0)

    assert result.stopped is False
    assert result.detail == "could not stop process"
    assert signals == [
        (80, signal.SIGINT),
        (80, signal.SIGTERM),
        (80, signal.SIGKILL),
    ]


def test_stop_record_falls_back_to_pid_when_group_signal_is_denied(monkeypatch):
    item = record(90)
    alive = iter([True, False, False, False])
    pid_signals = []
    times = iter([0.0, 2.0])
    monkeypatch.setattr(server, "process_alive", lambda _pid: next(alive))
    monkeypatch.setattr(server, "record_matches_process", lambda _record: True)
    monkeypatch.setattr(server, "_remove_record", lambda _record: None)

    def denied_group_signal(*_args):
        raise PermissionError

    monkeypatch.setattr(server.os, "killpg", denied_group_signal)
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: pid_signals.append((pid, sig)))
    monkeypatch.setattr(server.time, "monotonic", lambda: next(times))

    result = server.stop_record(item, grace=1)

    assert result.stopped is True
    assert pid_signals == [(90, signal.SIGINT)]


def test_stop_record_does_not_raise_when_all_signals_are_denied(monkeypatch):
    item = record(91)
    times = iter([0.0, 1.0, 2.0, 3.0, 4.0, 7.0])
    monkeypatch.setattr(server, "process_alive", lambda _pid: True)
    monkeypatch.setattr(server, "record_matches_process", lambda _record: True)
    monkeypatch.setattr(
        server.os, "killpg", lambda *_args: (_ for _ in ()).throw(PermissionError())
    )
    monkeypatch.setattr(server.os, "kill", lambda *_args: (_ for _ in ()).throw(PermissionError()))
    monkeypatch.setattr(server.time, "monotonic", lambda: next(times))

    result = server.stop_record(item, grace=0)

    assert result.stopped is False
    assert result.detail == "could not stop process"


def test_spawn_cleanup_keeps_a_stubborn_child_registered(monkeypatch):
    item = record(92)
    removed = []

    class StubbornProcess:
        def poll(self):
            return None

        def wait(self, *, timeout):
            raise server.subprocess.TimeoutExpired("mlx_lm.server", timeout)

    monkeypatch.setattr(server.os, "killpg", lambda *_args: None)
    monkeypatch.setattr(server, "_remove_record", lambda target: removed.append(target))

    server._stop_spawned_child(StubbornProcess(), item, grace=0)

    assert removed == []


def test_stop_all_filters_by_port_and_forwards_grace(monkeypatch):
    first = record(100, port=8080)
    second = record(101, port=9000)
    calls = []
    monkeypatch.setattr(server, "list_records", lambda: [first, second])

    def fake_stop(item, *, grace):
        calls.append((item.pid, grace))
        return server.StopResult(item, True, "stopped")

    monkeypatch.setattr(server, "stop_record", fake_stop)

    results = server.stop_all(port=9000, grace=1.5)

    assert [result.record.pid for result in results] == [101]
    assert calls == [(101, 1.5)]


def test_resolve_server_command_honors_an_explicit_executable(tmp_path, monkeypatch):
    executable = tmp_path / "mlx-server"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("MLX_LM_SERVER_BIN", str(executable))

    assert server.resolve_server_command() == [str(executable)]
