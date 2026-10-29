from __future__ import annotations

import json
import threading

import pytest

from mlx_chat.errors import MlxChatError
from mlx_chat.tools import ToolCall, Workspace


def call(name, **arguments):
    return ToolCall.from_api(
        {"id": "c1", "function": {"name": name, "arguments": json.dumps(arguments)}}, 0
    )


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def greet():\n    return 'hello'\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret")
    return Workspace(tmp_path)


def test_paths_are_confined_to_the_workspace(workspace, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope")
    for path in ("../outside.txt", str(outside)):
        outcome = workspace.execute(call("read_file", path=path))
        assert not outcome.ok
        assert "outside the workspace" in outcome.content


def test_symlink_escape_is_refused(workspace, tmp_path):
    target = tmp_path.parent / "elsewhere"
    target.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(target)
    outcome = workspace.execute(call("write_file", path="link/x.txt", content="x"))
    assert not outcome.ok
    assert not (target / "x.txt").exists()


def test_read_file_numbers_lines_and_pages(workspace, tmp_path):
    (tmp_path / "long.txt").write_text("\n".join(f"line {n}" for n in range(1, 1001)))
    first = workspace.execute(call("read_file", path="long.txt"))
    assert first.ok
    assert "  1│line 1" in first.content
    assert "start_line=401" in first.content
    page = workspace.execute(call("read_file", path="long.txt", start_line=990))
    assert "1000│line 1000" in page.content
    assert page.summary == "lines 990-1000 of 1000"


def test_read_file_rejects_binary(workspace, tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    assert "binary" in workspace.execute(call("read_file", path="blob.bin")).content


def test_list_files_skips_ignored_folders(workspace):
    outcome = workspace.execute(call("list_files"))
    assert "src/" in outcome.content
    assert "app.py" in outcome.content
    assert ".git" not in outcome.content


def test_search_finds_matches_with_python_fallback(workspace, monkeypatch):
    monkeypatch.setattr("mlx_chat.tools.shutil.which", lambda _name: None)
    outcome = workspace.execute(call("search", pattern="ret(urn)?", glob="*.py"))
    assert outcome.content == "src/app.py:2:    return 'hello'"
    assert workspace.execute(call("search", pattern="secret")).summary == "no matches"


def test_edit_requires_a_unique_exact_match(workspace, tmp_path):
    path = tmp_path / "src" / "app.py"
    missing = workspace.execute(call("edit_file", path="src/app.py", old_text="bye", new_text="x"))
    assert not missing.ok and "not found" in missing.content
    path.write_text("a = 1\na = 1\n")
    repeated = workspace.execute(
        call("edit_file", path="src/app.py", old_text="a = 1", new_text="")
    )
    assert not repeated.ok and "matches 2 places" in repeated.content
    everywhere = workspace.execute(
        call("edit_file", path="src/app.py", old_text="a = 1", new_text="b = 2", replace_all=True)
    )
    assert everywhere.ok
    assert path.read_text() == "b = 2\nb = 2\n"
    assert everywhere.diff.count("+b = 2") == 2


def test_edit_preview_does_not_change_the_file(workspace, tmp_path):
    preview = workspace.preview(
        call("edit_file", path="src/app.py", old_text="'hello'", new_text="'hi'")
    )
    assert "+    return 'hi'" in preview.diff
    assert "'hello'" in (tmp_path / "src" / "app.py").read_text()


def test_write_file_creates_parents_and_keeps_mode(workspace, tmp_path):
    script = tmp_path / "run.sh"
    script.write_text("echo old\n")
    script.chmod(0o755)
    assert workspace.execute(call("write_file", path="run.sh", content="echo new\n")).ok
    assert script.stat().st_mode & 0o777 == 0o755
    created = workspace.execute(call("write_file", path="docs/new/notes.md", content="hi\n"))
    assert created.ok and created.summary.startswith("created")
    assert (tmp_path / "docs" / "new" / "notes.md").read_text() == "hi\n"


def test_run_command_reports_exit_code_and_output(workspace):
    outcome = workspace.execute(call("run_command", command="echo out; echo err >&2; exit 3"))
    assert not outcome.ok
    assert "out" in outcome.content and "err" in outcome.content
    assert "[exit 3" in outcome.content


def test_run_command_times_out_and_can_be_cancelled(workspace):
    timed = workspace.execute(call("run_command", command="sleep 5", timeout=1))
    assert "timed out after 1s" in timed.content
    event = threading.Event()
    threading.Timer(0.2, event.set).start()
    cancelled = workspace.execute(call("run_command", command="sleep 5"), cancel_event=event)
    assert cancelled.cancelled


def test_bad_arguments_become_tool_errors_for_the_model(workspace):
    broken = ToolCall.from_api({"id": "x", "function": {"name": "read_file", "arguments": "{"}}, 0)
    assert "not valid JSON" in workspace.execute(broken).content
    assert "Unknown tool" in workspace.execute(call("delete_everything")).content
    assert "Invalid arguments" in workspace.execute(call("read_file", nope=1)).content


def test_output_is_clipped_to_the_configured_size(tmp_path):
    workspace = Workspace(tmp_path, output_chars=1000)
    (tmp_path / "big.txt").write_text("x" * 50 + "\n" + "y" * 5000)
    outcome = workspace.execute(call("read_file", path="big.txt"))
    assert len(outcome.content) < 1100
    assert "characters omitted" in outcome.content


def test_approval_modes(tmp_path):
    workspace = Workspace(tmp_path)
    assert not workspace.needs_approval("read_file", "ask", set())
    assert workspace.needs_approval("edit_file", "ask", set())
    assert not workspace.needs_approval("edit_file", "edits", set())
    assert workspace.needs_approval("run_command", "edits", set())
    assert not workspace.needs_approval("run_command", "ask", {"run_command"})
    assert not workspace.needs_approval("run_command", "all", set())


def test_missing_workspace_is_an_error(tmp_path):
    with pytest.raises(MlxChatError):
        Workspace(tmp_path / "missing")


def test_server_gets_a_bounded_prompt_cache_unless_configured():
    from mlx_chat.hardware import recommended_preset, tuned_server_args

    assert tuned_server_args([], 16) == ["--prompt-cache-bytes", "2000000000"]
    assert tuned_server_args([], 128) == ["--prompt-cache-bytes", "8000000000"]
    assert tuned_server_args(["--prompt-cache-size=2"], 16) == []
    assert [recommended_preset(value) for value in (8, 16, 24, 36, 96)] == [
        "tiny",
        "small",
        "small",
        "medium",
        "medium",
    ]


def test_find_files_matches_names_and_nested_globs(workspace, tmp_path):
    (tmp_path / "test_top.py").write_text("")
    (tmp_path / "src" / "test_inner.py").write_text("")
    assert workspace.execute(call("find_files", pattern="*.py")).summary == "3 files"
    nested = workspace.execute(call("find_files", pattern="**/test_*.py"))
    assert nested.content.splitlines() == ["test_top.py", "src/test_inner.py"]
    assert workspace.execute(call("find_files", pattern="*.rs")).summary == "no files"


def test_outline_lists_python_and_other_definitions(workspace, tmp_path):
    (tmp_path / "src" / "view.swift").write_text(
        "import SwiftUI\n\nstruct ContentView: View {\n    private func load() {}\n}\n"
    )
    (tmp_path / "src" / "app.py").write_text(
        "class Greeter(Base):\n    def hello(self, name):\n        pass\n\n"
        "async def main():\n    pass\n"
    )
    outcome = workspace.execute(call("outline", path="src"))
    assert "1  class Greeter(Base)" in outcome.content
    assert "2    def hello(self, name)" in outcome.content
    assert "5  async def main()" in outcome.content
    assert "3  struct ContentView: View" in outcome.content
    assert "4  private func load() {}" in outcome.content
    assert outcome.summary == "5 definitions in 2 files"


def test_update_plan_validates_steps(workspace):
    outcome = workspace.execute(
        call(
            "update_plan",
            steps=[{"step": "Read", "status": "done"}, {"step": "Edit", "status": "in_progress"}],
        )
    )
    assert outcome.summary == "1/2 done"
    assert workspace.plan[1] == {"step": "Edit", "status": "in_progress"}
    assert not workspace.execute(call("update_plan", steps=[{"step": "x", "status": "later"}])).ok
    assert not workspace.needs_approval("update_plan", "ask", set())


def test_fetch_url_returns_readable_text_and_needs_approval(tmp_path):
    import http.server
    import threading

    from mlx_chat.config import Settings

    page = (
        b"<html><head><title>Docs</title><style>.x{}</style></head><body><h1>Install</h1>"
        b"<p>Run <code>uv sync</code>.</p><script>alert(1)</script></body></html>"
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        workspace = Workspace(tmp_path, web=Settings())
        url = f"http://127.0.0.1:{server.server_port}/docs"
        outcome = workspace.execute(call("fetch_url", url=url))
    finally:
        server.shutdown()
    assert outcome.ok
    assert "# Install" in outcome.content and "Run uv sync." in outcome.content
    assert "alert" not in outcome.content and ".x{}" not in outcome.content
    assert outcome.summary.endswith("Docs")
    assert workspace.needs_approval("fetch_url", "edits", set())
    assert not workspace.execute(call("fetch_url", url="file:///etc/passwd")).ok


def test_web_tools_are_only_offered_with_web_settings(tmp_path):
    assert "fetch_url" not in Workspace(tmp_path).tools
