from __future__ import annotations

import json

import pytest

from mlx_chat.agent import AgentState, run_agent
from mlx_chat.client import GenerationResult, merge_tool_call_deltas
from mlx_chat.config import Settings
from mlx_chat.errors import ApiError, MlxChatError
from mlx_chat.sessions import restore_session
from mlx_chat.tools import Workspace


def tool_call(call_id, name, **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class FakeHost:
    def __init__(self, results, decisions=()):
        self.results = list(results)
        self.decisions = list(decisions)
        self.requests = []
        self.finished = []
        self.notices = []

    def generate(self, messages, tools):
        self.requests.append((messages, tools))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def approve(self, call, preview):
        return self.decisions.pop(0)

    def execute(self, call, run):
        return run(None)

    def tool_finished(self, item):
        self.finished.append(item)

    def notice(self, message):
        self.notices.append(message)


def history(transcript):
    return lambda: [{"role": "system", "content": "s"}, *transcript]


def test_loop_runs_tools_and_returns_the_final_answer(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n")
    host = FakeHost(
        [
            GenerationResult(tool_calls=[tool_call("1", "read_file", path="a.txt")]),
            GenerationResult(
                answer="It says alpha.",
                usage={"prompt_tokens": 50, "prompt_tokens_details": {"cached_tokens": 40}},
            ),
        ]
    )
    transcript = [{"role": "user", "content": "What is in a.txt?"}]
    run = run_agent(AgentState(Workspace(tmp_path)), host, transcript, history(transcript))

    assert run.result.answer == "It says alpha."
    assert (run.steps, run.tool_calls, run.cached_tokens) == (2, 1, 40)
    assert [item["role"] for item in transcript] == ["user", "assistant", "tool", "assistant"]
    assert transcript[2]["tool_call_id"] == "1"
    assert "1│alpha" in transcript[2]["content"]
    # The second request carries the call and its result after the original prefix.
    messages, tools = host.requests[1]
    assert messages[:2] == host.requests[0][0][:2]
    assert tools and {tool["function"]["name"] for tool in tools} >= {"read_file", "run_command"}


def test_declined_call_is_reported_to_the_model_with_feedback(tmp_path):
    host = FakeHost(
        [
            GenerationResult(tool_calls=[tool_call("1", "write_file", path="x.txt", content="x")]),
            GenerationResult(answer="OK, I will not write it."),
        ],
        decisions=[("deny", "use notes.md instead")],
    )
    transcript = [{"role": "user", "content": "write"}]
    run_agent(AgentState(Workspace(tmp_path)), host, transcript, history(transcript))

    assert not (tmp_path / "x.txt").exists()
    assert transcript[2]["approval"] == "denied"
    assert "use notes.md instead" in transcript[2]["content"]


def test_always_allows_later_calls_without_asking(tmp_path):
    state = AgentState(Workspace(tmp_path))
    host = FakeHost(
        [
            GenerationResult(tool_calls=[tool_call("1", "run_command", command="true")]),
            GenerationResult(tool_calls=[tool_call("2", "run_command", command="true")]),
            GenerationResult(answer="done"),
        ],
        decisions=[("always", "")],
    )
    transcript = [{"role": "user", "content": "go"}]
    run_agent(state, host, transcript, history(transcript))
    assert [item.get("approval") for item in transcript if item["role"] == "tool"] == [
        "always",
        "auto",
    ]
    assert state.always_allowed == {"run_command"}


def test_step_limit_forces_a_final_answer_without_tools(tmp_path):
    looping = GenerationResult(tool_calls=[tool_call("1", "list_files")])
    host = FakeHost([looping, looping, GenerationResult(answer="summary")])
    transcript = [{"role": "user", "content": "go"}]
    run = run_agent(
        AgentState(Workspace(tmp_path), max_steps=2), host, transcript, history(transcript)
    )
    assert run.result.answer == "summary"
    assert host.requests[-1][1] is None
    assert host.notices


def test_models_without_tool_support_fall_back_to_chat(tmp_path):
    state = AgentState(Workspace(tmp_path))
    host = FakeHost(
        [
            ApiError(
                "MLX-LM returned HTTP 400: Received tools but model does not support tool calling."
            ),
            GenerationResult(answer="plain"),
        ]
    )
    transcript = [{"role": "user", "content": "hi"}]
    run = run_agent(state, host, transcript, history(transcript))
    assert run.result.answer == "plain"
    assert not state.tools_supported
    assert host.requests[1][1] is None


def test_streamed_tool_call_deltas_merge_complete_and_incremental_forms():
    calls = []
    merge_tool_call_deltas(calls, [tool_call("a", "read_file", path="x") | {"index": 0}])
    merge_tool_call_deltas(
        calls, [{"index": 1, "id": "b", "function": {"name": "search", "arguments": '{"pat'}}]
    )
    merge_tool_call_deltas(calls, [{"index": 1, "function": {"arguments": 'tern": "y"}'}}])
    assert [call["id"] for call in calls] == ["a", "b"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"pattern": "y"}


def test_sessions_round_trip_tool_steps_and_reject_malformed_ones():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("1", "list_files")]},
        {"role": "tool", "tool_call_id": "1", "name": "list_files", "content": "./", "ok": True},
        {"role": "assistant", "content": "done"},
    ]
    restored = restore_session({"messages": messages}, Settings())
    assert restored.messages == messages
    for broken in (
        {"role": "tool", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": 1}]},
    ):
        with pytest.raises(MlxChatError):
            restore_session({"messages": [messages[0], broken]}, Settings())


def test_chat_agent_context_shortens_old_tool_output_only_when_needed(tmp_path):
    from types import SimpleNamespace

    from mlx_chat import chat

    app = object.__new__(chat.ChatApp)
    app.settings = Settings(context_tokens=3000, max_tokens=500)
    app.agent = AgentState(Workspace(tmp_path))
    app.ui = SimpleNamespace()
    big = "x" * 4000
    app.transcript = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("1", "list_files")]},
        {"role": "tool", "tool_call_id": "1", "name": "list_files", "content": big},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("2", "list_files")]},
        {"role": "tool", "tool_call_id": "2", "name": "list_files", "content": big},
    ]
    messages = app.api_messages()
    tools = [message for message in messages if message["role"] == "tool"]
    assert "shortened" in tools[0]["content"]
    assert tools[1]["content"] == big
    assert app.transcript[2]["content"] == big
    assert messages[1]["content"].startswith("You are a coding and automation agent")


def test_project_memory_is_loaded_into_the_agent_prompt(tmp_path):
    from mlx_chat.agent import agent_system_message, load_project_memory

    (tmp_path / "AGENTS.md").write_text("Run tests with make check.\n")
    (tmp_path / "CLAUDE.md").write_text("x" * 20000)
    memory = load_project_memory(tmp_path)
    assert [name for name, _ in memory] == ["AGENTS.md", "CLAUDE.md"]
    assert sum(len(text) for _, text in memory) < 12100
    content = agent_system_message(Workspace(tmp_path), memory)["content"]
    assert "Project instructions from AGENTS.md" in content
    assert "make check" in content


def test_planning_blocks_edits_and_commands_but_allows_reading(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n")
    host = FakeHost(
        [
            GenerationResult(
                tool_calls=[
                    tool_call("1", "read_file", path="a.txt"),
                    tool_call("2", "write_file", path="a.txt", content="changed"),
                ]
            ),
            GenerationResult(answer="1. Change a.txt"),
        ]
    )
    transcript = [{"role": "user", "content": "plan"}]
    state = AgentState(Workspace(tmp_path), read_only=True)
    run_agent(state, host, transcript, history(transcript))
    assert (tmp_path / "a.txt").read_text() == "alpha\n"
    tools = [item for item in transcript if item["role"] == "tool"]
    assert tools[0]["ok"] and not tools[1]["ok"]
    assert "planning" in tools[1]["content"]
    # Planning sends the same tool list as execution so the cached prefix stays valid.
    assert len(host.requests[0][1]) == len(Workspace(tmp_path).schemas())


def test_compaction_summarizes_older_turns_and_sends_only_the_summary(tmp_path):
    from types import SimpleNamespace

    from mlx_chat import chat

    app = object.__new__(chat.ChatApp)
    app.settings = Settings()
    app.agent = None
    app.session_name = None
    requests = []
    notices = []

    def collect(messages, *, think):
        requests.append(messages)
        return GenerationResult(answer="- earlier work")

    app.client = SimpleNamespace(collect=collect)
    app.ui = SimpleNamespace(
        run_task=lambda operation, label=None: operation(),
        summary_notice=lambda item, automatic=False: notices.append(item),
    )
    app.transcript = []
    for number in range(3):
        app.transcript += [
            {"role": "user", "content": f"question {number}"},
            {"role": "assistant", "content": f"answer {number}"},
        ]
    app._compact(keep_turns=1)

    assert "question 0" in requests[0][1]["content"]
    assert [item["role"] for item in app.transcript].count("summary") == 1
    assert app.transcript[4]["role"] == "summary" and app.transcript[4]["covers"] == 4
    sent = app.api_messages()
    assert "- earlier work" in sent[1]["content"]
    assert [message["content"] for message in sent[2:]] == ["question 2", "answer 2"]
    assert notices
    restore_session({"messages": app.transcript}, Settings())


def test_automatic_compaction_waits_for_a_full_context(tmp_path):
    from types import SimpleNamespace

    from mlx_chat import chat

    app = object.__new__(chat.ChatApp)
    app.settings = Settings(context_tokens=2000, max_tokens=500)
    app.agent = None
    calls = []
    app.client = SimpleNamespace()
    app._compact = lambda automatic=False: calls.append(automatic)
    app.transcript = [{"role": "user", "content": "short"}] * 3
    app._maybe_compact()
    assert calls == []
    app.transcript = [
        item
        for number in range(3)
        for item in ({"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y"})
    ]
    app._maybe_compact()
    assert calls == [True]
