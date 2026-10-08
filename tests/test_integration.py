"""Real local HTTP, CLI subprocess, and harmless child-process lifecycle checks."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_chat import server
from mlx_chat.client import ChatClient
from mlx_chat.config import Settings
from mlx_chat.errors import ApiError, ServerUnavailableError
from mlx_chat.server import ServerRecord

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_api(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "home"))
    state = SimpleNamespace(
        mode="normal", release=threading.Event(), started=threading.Event(), requests=[]
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            body = json.dumps(
                {"status": "ok"} if self.path == "/health" else {"data": [{"id": "test/model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(payload)
            try:
                if state.mode == "headers":
                    state.started.set()
                    state.release.wait(5)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if state.mode == "malformed":
                    self.wfile.write(b"data: {invalid}\n\n")
                elif state.mode == "half-line":
                    self.wfile.write(b'data: {"choices":')
                else:
                    packet = {"choices": [{"delta": {"content": "Hello"}}]}
                    self.wfile.write(f"data: {json.dumps(packet)}\n\n".encode())
                self.wfile.flush()
                state.started.set()
                if state.mode in {"idle", "half-line"}:
                    state.release.wait(5)
                if state.mode != "disconnect":
                    packet = {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 1},
                    }
                    self.wfile.write(f"data: {json.dumps(packet)}\n\ndata: [DONE]\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    state.port = httpd.server_port
    try:
        yield state
    finally:
        state.release.set()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


def client(api, **kw):
    return ChatClient(Settings(model="test/model", port=api.port, **kw))


def test_real_stream_and_request_payload(local_api):
    result = client(local_api).collect([{"role": "user", "content": "hello"}], think=False)
    assert result.answer == "Hello"
    assert result.finish_reason == "stop"
    assert result.completion_tokens == 1
    assert local_api.requests[0]["messages"][-1]["content"] == "hello"
    assert local_api.requests[0]["stream_options"] == {"include_usage": True}
    assert client(local_api).list_models() == ["test/model"]


@pytest.mark.parametrize("mode", ["headers", "idle", "half-line"])
def test_cancellation_interrupts_stalled_http_without_waiting_for_tokens(local_api, mode):
    local_api.mode = mode
    cancelled = threading.Event()

    def cancel():
        assert local_api.started.wait(2)
        time.sleep(0.1)
        cancelled.set()

    timer = threading.Thread(target=cancel)
    timer.start()
    started = time.monotonic()
    result = client(local_api).collect([], think=False, cancel_event=cancelled)
    elapsed = time.monotonic() - started
    timer.join(timeout=2)
    assert result.finish_reason == "cancelled"
    assert result.answer == ("Hello" if mode == "idle" else "")
    assert elapsed < 1.5


@pytest.mark.parametrize("mode", ["headers", "idle"])
def test_read_timeout_reports_connection_error(local_api, mode):
    local_api.mode = mode
    with pytest.raises(ServerUnavailableError, match="connection"):
        client(local_api, request_timeout=1).collect([], think=False)


@pytest.mark.parametrize(
    "mode,match", [("disconnect", "ended unexpectedly"), ("malformed", "Invalid streaming JSON")]
)
def test_disconnects_and_invalid_sse_have_friendly_errors(local_api, mode, match):
    local_api.mode = mode
    with pytest.raises(ApiError, match=match):
        client(local_api).collect([], think=False)


def run_cli(api, tmp_path, *args):
    env = dict(os.environ, MLX_CHAT_HOME=str(tmp_path / "cli-home"), PYTHONPATH=str(ROOT / "src"))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_chat",
            "run",
            "hello",
            "--attach",
            "--no-warmup",
            "--port",
            str(api.port),
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_cli_json_stdout_contains_only_result(local_api, tmp_path):
    outcome = run_cli(local_api, tmp_path, "--json", "--concise", "--think")
    assert outcome.returncode == 0, outcome.stderr
    packet = json.loads(outcome.stdout)
    assert packet["answer"] == "Hello"
    assert local_api.requests[-1]["chat_template_kwargs"]["enable_thinking"] is False
    assert "briefly" in local_api.requests[-1]["messages"][1]["content"]


def test_cli_inference_error_exit_code_and_stderr(local_api, tmp_path):
    local_api.mode = "malformed"
    outcome = run_cli(local_api, tmp_path, "--json")
    assert outcome.returncode == 5
    assert outcome.stdout == ""
    assert "Invalid streaming JSON" in outcome.stderr
    assert "Traceback" not in outcome.stderr


@pytest.fixture
def harmless_child(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_CHAT_HOME", str(tmp_path / "home"))
    script = (
        "import signal,sys,time; "
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
        "print('ready', flush=True); time.sleep(30)"
    )
    command = [
        sys.executable,
        "-c",
        script,
        "--model",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "54321",
    ]
    process = subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 5)[0], "Dummy child failed to start"
        assert process.stdout.readline().strip() == "ready"
        record = ServerRecord(
            process.pid,
            process.pid,
            "dummy",
            "127.0.0.1",
            54321,
            "test",
            str(tmp_path / "dummy.log"),
            command,
            process_identity=server.process_start_identity(process.pid),
            instance_id="dummy",
        )
        assert record.process_identity
        yield process, record
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        process.stdout.close()


def test_real_owned_child_is_registered_and_stopped(harmless_child):
    process, record = harmless_child
    assert server.record_matches_process(record)
    server._replace_record(record)
    assert server.list_records() == [record]
    stopped = server.stop_record(record, grace=1)
    assert stopped.stopped
    assert not server.process_alive(process.pid)
    assert server.list_records() == []


def test_real_child_with_wrong_identity_is_left_running(harmless_child):
    process, record = harmless_child
    record.process_identity = "wrong identity"
    stopped = server.stop_record(record, grace=0)
    assert not stopped.stopped
    assert process.poll() is None


def test_built_wheel_imports_and_cli_work_outside_checkout(tmp_path):
    wheels = sorted(
        (ROOT / "dist").glob("mlx_chat_terminal-*.whl"), key=lambda path: path.stat().st_mtime
    )
    if not wheels:
        pytest.skip("Build a wheel with uv build before running packaging smoke tests")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheels[-1]) as archive:
        metadata_path = next(name for name in archive.namelist() if name.endswith("/METADATA"))
        dependencies = archive.read(metadata_path).decode().splitlines()
        search_dependency = next(
            line for line in dependencies if line.startswith("Requires-Dist: ddgs")
        )
        assert "extra ==" in search_dependency and "web" in search_dependency
        archive.extractall(installed)
    program = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import mlx_chat,mlx_chat.transport,mlx_chat.sessions,mlx_chat.context; "
        "assert mlx_chat.__file__.startswith(sys.argv[1]); "
        "from mlx_chat.cli import main; main(['config', 'show'])"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(installed)],
        cwd=tmp_path,
        env=dict(os.environ, MLX_CHAT_HOME=str(tmp_path / "wheel-home")),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["context_tokens"] == 32768


@pytest.mark.parametrize("mode", ["headers", "idle"])
def test_terminal_ctrl_c_cancels_real_stalled_generation(local_api, monkeypatch, mode):
    import io

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from rich.console import Console

    from mlx_chat.ui import TerminalUI

    local_api.mode = mode
    monkeypatch.setattr(
        "mlx_chat.ui.Console",
        lambda **kw: Console(
            file=io.StringIO(),
            force_terminal=True,
            width=80,
            height=40,
        ),
    )
    monkeypatch.setattr("mlx_chat.rendering.resource_snapshot", lambda _port: None)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        ui = TerminalUI(Settings(port=local_api.port))
        ui.enable_conversation()

        def cancel():
            assert local_api.started.wait(2)
            time.sleep(0.1)
            pipe.send_text("\x03")

        timer = threading.Thread(target=cancel)
        timer.start()
        started = time.monotonic()
        try:
            result = ui.render_generation(
                lambda callback: client(local_api).collect(
                    [],
                    think=False,
                    on_event=callback,
                    cancel_event=ui.cancellation_event,
                ),
                think=False,
            )
            assert result.finish_reason == "cancelled"
            assert time.monotonic() - started < 2
            assert ui._conversation.mode == "input"
        finally:
            timer.join(timeout=2)
            ui.close_conversation()


def test_already_cancelled_request_never_contacts_server(local_api):
    cancelled = threading.Event()
    cancelled.set()
    result = client(local_api).collect([], think=False, cancel_event=cancelled)
    assert result.finish_reason == "cancelled"
    assert local_api.requests == []
