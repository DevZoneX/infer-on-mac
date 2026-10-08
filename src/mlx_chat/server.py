"""Lifecycle management for MLX-LM server processes owned by mlx-chat."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Settings, _atomic_json_write, app_home
from .errors import MlxChatError, ServerUnavailableError


@dataclass
class ServerRecord:
    pid: int
    pgid: int
    model: str
    host: str
    port: int
    started_at: str
    log_path: str
    command: list[str]
    process_identity: str = ""
    instance_id: str = ""
    server_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ServerRecord:
        return cls(
            pid=int(value["pid"]),
            pgid=int(value.get("pgid", value["pid"])),
            model=str(value["model"]),
            host=str(value["host"]),
            port=int(value["port"]),
            started_at=str(value["started_at"]),
            log_path=str(value["log_path"]),
            command=[str(item) for item in value.get("command", [])],
            process_identity=str(value.get("process_identity", "")),
            instance_id=str(value.get("instance_id", "")),
            server_args=[str(item) for item in value.get("server_args", [])],
        )


@dataclass
class StartResult:
    model: str
    host: str
    port: int
    pid: int | None
    log_path: Path | None
    reused: bool
    managed: bool
    instance_id: str = ""


@dataclass
class StopResult:
    record: ServerRecord
    stopped: bool
    detail: str


def registry_path() -> Path:
    return app_home() / "servers.json"


def registry_lock_path() -> Path:
    return app_home() / "servers.lock"


@contextlib.contextmanager
def _endpoint_lock(port: int, progress: Callable[[str], None]) -> Iterator[None]:
    """Serialize check/spawn/register for one listening port across CLI processes."""
    path = app_home() / "locks" / f"port-{port}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        announced = False
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not announced:
                    progress(f"Another launch is preparing port {port}; waiting")
                    announced = True
                time.sleep(0.25)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _registry_lock() -> Iterator[None]:
    path = registry_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_registry_unlocked() -> list[ServerRecord]:
    path = registry_path()
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        raw_records = value.get("servers", []) if isinstance(value, dict) else []
        if not isinstance(raw_records, list):
            return []
        records = []
        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            try:
                records.append(ServerRecord.from_dict(raw))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return records
    except (OSError, json.JSONDecodeError):
        return []


def _write_registry_unlocked(records: list[ServerRecord]) -> None:
    _atomic_json_write(
        registry_path(),
        {"version": 1, "servers": [asdict(record) for record in records]},
    )


def process_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    # Reap the process when it is our own child. Without this, a server stopped
    # from the same launcher can remain a zombie and `kill(pid, 0)` reports it
    # as alive forever.
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return False
    except ChildProcessError:
        # A later `mlx-chat stop` process is not the server's parent.
        pass
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # `kill(pid, 0)` also succeeds for a zombie owned by another parent.
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip().startswith("Z"):
            return False
    except (OSError, subprocess.SubprocessError):
        pass
    return True


def process_command(pid: int) -> str:
    if not process_alive(pid):
        return ""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def process_start_identity(pid: int) -> str:
    """Return the OS process start stamp used to detect recycled PIDs."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def record_matches_process(record: ServerRecord) -> bool:
    """Guard against signaling a PID that the OS has recycled."""
    # Old registry entries did not carry an OS start stamp. Treat them as
    # untrusted instead of risking a signal to a recycled PID.
    if not record.process_identity:
        return False
    command = process_command(record.pid)
    if not command:
        return False
    try:
        if os.getpgid(record.pid) != record.pgid:
            return False
    except (ProcessLookupError, PermissionError, OSError):
        return False
    current_identity = process_start_identity(record.pid)
    if not current_identity or current_identity != record.process_identity:
        return False
    try:
        arguments = shlex.split(command)
    except ValueError:
        return False

    official_command = any(Path(item).name == "mlx_lm.server" for item in arguments) or any(
        arguments[index : index + 3] == ["-m", "mlx_lm", "server"]
        for index in range(max(0, len(arguments) - 2))
    )
    configured_executable = bool(
        record.command
        and any(
            item == record.command[0] or Path(item).name == Path(record.command[0]).name
            for item in arguments
        )
    )
    if not (official_command or configured_executable):
        return False
    expected_arguments = (
        ("--model", record.model),
        ("--host", record.host),
        ("--port", str(record.port)),
    )

    def has_exact_option(flag: str, value: str) -> bool:
        return any(
            item == f"{flag}={value}"
            or (item == flag and index + 1 < len(arguments) and arguments[index + 1] == value)
            for index, item in enumerate(arguments)
        )

    return all(has_exact_option(flag, value) for flag, value in expected_arguments)


def list_records(*, include_stale: bool = False) -> list[ServerRecord]:
    with _registry_lock():
        records = _read_registry_unlocked()
        live = [record for record in records if record_matches_process(record)]
        if len(live) != len(records):
            _write_registry_unlocked(live)
    return records if include_stale else live


def _replace_record(record: ServerRecord) -> None:
    with _registry_lock():
        records = _read_registry_unlocked()
        records = [
            existing
            for existing in records
            if not (existing.host == record.host and existing.port == record.port)
            and existing.pid != record.pid
            and record_matches_process(existing)
        ]
        records.append(record)
        _write_registry_unlocked(records)


def _same_server_instance(current: ServerRecord, target: ServerRecord) -> bool:
    if current.pid != target.pid:
        return False
    if target.instance_id:
        return current.instance_id == target.instance_id
    if target.process_identity:
        return current.process_identity == target.process_identity
    # Compatibility for registry records written before ownership tokens were
    # introduced. Avoid deleting a different record merely because its PID was
    # recycled.
    return (
        current.started_at == target.started_at
        and current.host == target.host
        and current.port == target.port
        and current.command == target.command
    )


def _remove_record(target: ServerRecord) -> None:
    with _registry_lock():
        records = [
            record
            for record in _read_registry_unlocked()
            if not _same_server_instance(record, target)
        ]
        _write_registry_unlocked(records)


def resolve_server_command() -> list[str]:
    override = os.environ.get("MLX_LM_SERVER_BIN")
    if override:
        resolved = shutil.which(override) if os.sep not in override else override
        if resolved and Path(resolved).exists():
            return [str(resolved)]
        raise MlxChatError(f"MLX_LM_SERVER_BIN does not exist: {override}")

    executable = shutil.which("mlx_lm.server")
    if executable:
        return [executable]

    # The supported module fallback is `python -m mlx_lm server`.
    try:
        import importlib.util

        if importlib.util.find_spec("mlx_lm") is not None:
            return [sys.executable, "-m", "mlx_lm", "server"]
    except (ImportError, ValueError):
        pass

    raise MlxChatError(
        "mlx_lm.server was not found. Run ./setup.sh, or activate an environment "
        "that contains mlx-lm."
    )


def endpoint(host: str, port: int) -> str:
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in client_host and not client_host.startswith("["):
        client_host = f"[{client_host}]"
    return f"http://{client_host}:{port}"


def health(host: str, port: int, timeout: float = 0.75) -> bool:
    request = urllib.request.Request(f"{endpoint(host, port)}/health")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            raw = json.loads(response.read().decode("utf-8"))
            return isinstance(raw, dict) and raw.get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def port_is_open(host: str, port: int, timeout: float = 0.25) -> bool:
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((client_host, port), timeout=timeout):
            return True
    except OSError:
        return False


def server_configuration_matches(record: ServerRecord, settings: Settings) -> bool:
    return record.model == settings.model and record.server_args == list(settings.server_args or [])


class ServerManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def matching_record(self) -> ServerRecord | None:
        for record in list_records():
            if record.host == self.settings.host and record.port == self.settings.port:
                return record
        return None

    def start(
        self,
        *,
        progress: Callable[[str], None] | None = None,
        ready_check: Callable[[StartResult], None] | None = None,
    ) -> StartResult:
        progress = progress or (lambda _message: None)
        with _endpoint_lock(self.settings.port, progress):
            result = self._start_locked(progress)
            try:
                if ready_check is not None:
                    ready_check(result)
            except BaseException:
                # Keep health + first-generation validation atomic for other
                # launchers. No waiter can reuse this server before a failed
                # owner warm-up has cleaned up its exact process.
                self._cleanup_failed_start(result)
                raise
            return result

    def attach(
        self,
        *,
        progress: Callable[[str], None] | None = None,
        ready_check: Callable[[StartResult], None] | None = None,
    ) -> StartResult:
        """Attach to a healthy endpoint without ever spawning a process."""

        progress = progress or (lambda _message: None)
        with _endpoint_lock(self.settings.port, progress):
            existing = self.matching_record()
            if existing and not server_configuration_matches(existing, self.settings):
                raise MlxChatError(
                    f"The managed server at {self.settings.base_url} uses a different "
                    "model or set of MLX server arguments. --attach never changes "
                    "server configuration."
                )
            if not health(self.settings.host, self.settings.port):
                raise ServerUnavailableError(
                    f"No healthy MLX-LM server is running at {self.settings.base_url}. "
                    "Remove --attach to start one."
                )
            result = StartResult(
                model=existing.model if existing else self.settings.model,
                host=self.settings.host,
                port=self.settings.port,
                pid=existing.pid if existing else None,
                log_path=Path(existing.log_path) if existing else None,
                reused=True,
                managed=existing is not None,
                instance_id=existing.instance_id if existing else "",
            )
            if ready_check is not None:
                ready_check(result)
            return result

    def _cleanup_failed_start(self, result: StartResult) -> None:
        if not result.managed or result.reused or result.pid is None or not result.instance_id:
            return
        record = self.matching_record()
        if record and record.pid == result.pid and record.instance_id == result.instance_id:
            with contextlib.suppress(Exception):
                stop_record(record, grace=2)

    def _start_locked(self, progress: Callable[[str], None]) -> StartResult:
        existing = self.matching_record()
        if existing and not server_configuration_matches(existing, self.settings):
            raise MlxChatError(
                f"Port {self.settings.port} is already assigned to a different managed "
                "model or set of MLX server arguments. Stop it first or choose another --port."
            )
        if health(self.settings.host, self.settings.port):
            return StartResult(
                model=existing.model if existing else self.settings.model,
                host=self.settings.host,
                port=self.settings.port,
                pid=existing.pid if existing else None,
                log_path=Path(existing.log_path) if existing else None,
                reused=True,
                managed=existing is not None,
                instance_id=existing.instance_id if existing else "",
            )

        if existing:
            progress("Waiting for the existing managed server to become healthy")
            deadline = time.monotonic() + self.settings.startup_timeout
            while time.monotonic() < deadline:
                if health(self.settings.host, self.settings.port):
                    return StartResult(
                        model=existing.model,
                        host=existing.host,
                        port=existing.port,
                        pid=existing.pid,
                        log_path=Path(existing.log_path),
                        reused=True,
                        managed=True,
                        instance_id=existing.instance_id,
                    )
                if not record_matches_process(existing):
                    _remove_record(existing)
                    existing = None
                    break
                time.sleep(0.25)
            if existing is not None:
                raise ServerUnavailableError(
                    f"The managed server on port {self.settings.port} is still running but "
                    f"did not become healthy within {self.settings.startup_timeout} seconds. "
                    f"Inspect {existing.log_path} or run `mlx-chat stop --port "
                    f"{self.settings.port}`."
                )

        if port_is_open(self.settings.host, self.settings.port):
            raise MlxChatError(
                f"Port {self.settings.port} is occupied by a service that is not a healthy "
                "MLX-LM server. Choose another --port or stop that service."
            )

        command = resolve_server_command()
        command.extend(
            [
                "--model",
                self.settings.model,
                "--host",
                self.settings.host,
                "--port",
                str(self.settings.port),
            ]
        )
        command.extend(self.settings.server_args or [])

        log_dir = app_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"server-{self.settings.port}-{stamp}.log"
        progress("Starting mlx_lm.server")
        try:
            with log_path.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    cwd=str(Path.cwd()),
                )
        except OSError as exc:
            raise MlxChatError(f"Could not start mlx_lm.server: {exc}") from exc

        record = ServerRecord(
            pid=process.pid,
            pgid=process.pid,
            model=self.settings.model,
            host=self.settings.host,
            port=self.settings.port,
            started_at=datetime.now(UTC).isoformat(),
            log_path=str(log_path),
            command=command,
            process_identity="",
            instance_id=uuid4().hex,
            server_args=list(self.settings.server_args or []),
        )
        try:
            record.process_identity = process_start_identity(process.pid)
            if not record.process_identity:
                raise ServerUnavailableError(
                    "Could not establish a safe process identity for mlx_lm.server; "
                    "the child was stopped instead of leaving an unsafe managed process."
                )
            _replace_record(record)

            deadline = time.monotonic() + self.settings.startup_timeout
            last_progress = 0.0
            while time.monotonic() < deadline:
                code = process.poll()
                if code is not None:
                    tail = tail_log(log_path, lines=30)
                    detail = f"\n\nLast server output:\n{tail}" if tail else ""
                    raise ServerUnavailableError(
                        f"mlx_lm.server exited with status {code}.{detail}"
                    )
                if health(self.settings.host, self.settings.port):
                    return StartResult(
                        model=self.settings.model,
                        host=self.settings.host,
                        port=self.settings.port,
                        pid=process.pid,
                        log_path=log_path,
                        reused=False,
                        managed=True,
                        instance_id=record.instance_id,
                    )
                now = time.monotonic()
                if now - last_progress >= 2:
                    progress(
                        "Waiting for the local server (the first model download may take a while)"
                    )
                    last_progress = now
                time.sleep(0.2)

            raise ServerUnavailableError(
                f"Server did not become ready within {self.settings.startup_timeout} seconds. "
                f"See {log_path}."
            )
        except BaseException:
            # This Popen object is proof of exact ownership. Cleaning the child
            # here avoids both orphaning it on Ctrl-C and ever guessing from a
            # registry snapshot that may have been replaced by another launcher.
            _stop_spawned_child(process, record)
            raise


def _stop_spawned_child(
    process: subprocess.Popen[bytes],
    record: ServerRecord,
    *,
    grace: float = 2.0,
) -> None:
    """Stop the exact child created by this process, even during early startup."""

    def wait_for_exit(timeout: float) -> bool:
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False
        except (ChildProcessError, OSError):
            return process.poll() is not None

    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(record.pgid, signal.SIGINT)
        if not wait_for_exit(grace):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(record.pgid, signal.SIGTERM)
            if not wait_for_exit(min(grace, 2.0)):
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(record.pgid, signal.SIGKILL)
                wait_for_exit(2.0)
    # Keep a stubborn child registered so `mlx-chat stop` can retry. Registry
    # removal is ownership-aware and only happens after the exact Popen child
    # is confirmed gone.
    if process.poll() is not None:
        _remove_record(record)


def stop_record(record: ServerRecord, *, grace: float = 8.0) -> StopResult:
    if not process_alive(record.pid):
        _remove_record(record)
        return StopResult(record, False, "already stopped")
    if not record_matches_process(record):
        _remove_record(record)
        return StopResult(record, False, "PID was reused; left the unrelated process untouched")

    def send(sig: signal.Signals) -> None:
        if not record_matches_process(record):
            return
        try:
            os.killpg(record.pgid, sig)
        except ProcessLookupError:
            return
        except (PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(record.pid, sig)

    # SIGINT lets MLX-LM shut down its HTTP and generation threads cleanly.
    send(signal.SIGINT)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and process_alive(record.pid):
        time.sleep(0.1)
    if process_alive(record.pid):
        send(signal.SIGTERM)
        deadline = time.monotonic() + min(grace, 4)
        while time.monotonic() < deadline and process_alive(record.pid):
            time.sleep(0.1)
    if process_alive(record.pid):
        send(signal.SIGKILL)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and process_alive(record.pid):
            time.sleep(0.05)

    stopped = not process_alive(record.pid)
    if stopped:
        _remove_record(record)
    return StopResult(record, stopped, "stopped" if stopped else "could not stop process")


def stop_all(*, port: int | None = None, grace: float = 8.0) -> list[StopResult]:
    records = list_records()
    if port is not None:
        records = [record for record in records if record.port == port]
    results = [stop_record(record, grace=grace) for record in records]
    return results


def tail_log(path: Path, *, lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            block = 4096
            data = b""
            position = size
            while position > 0 and data.count(b"\n") <= lines:
                take = min(block, position)
                position -= take
                handle.seek(position)
                data = handle.read(take) + data
        return b"\n".join(data.splitlines()[-lines:]).decode("utf-8", errors="replace")
    except OSError:
        return ""


def latest_log() -> Path | None:
    records = list_records()
    if records:
        return Path(max(records, key=lambda item: item.started_at).log_path)
    directory = app_home() / "logs"
    logs = list(directory.glob("server-*.log")) if directory.exists() else []
    return max(logs, key=lambda item: item.stat().st_mtime) if logs else None
