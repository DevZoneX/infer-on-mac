from __future__ import annotations

import json
from collections.abc import Iterable

import pytest

from mlx_chat.client import ChatClient, StreamEvent, ThinkTagSplitter
from mlx_chat.config import Settings
from mlx_chat.errors import ApiError


class FakeResponse:
    def __init__(
        self,
        lines: Iterable[bytes] = (),
        *,
        status: int = 200,
        content_type: str = "text/event-stream; charset=utf-8",
        body: bytes = b"",
    ):
        self.lines = list(lines)
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self) -> bytes:
        return self.body


def data_line(value) -> bytes:
    return f"data: {json.dumps(value)}\n".encode()


def test_thinking_payload_uses_qwen_profile_and_separation_flag():
    client = ChatClient(Settings())
    messages = [{"role": "user", "content": "Why?"}]

    payload = client.payload(messages, think=True)

    assert payload == {
        "model": "mlx-community/Qwen3-4B-4bit",
        "messages": messages,
        "stream": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 4096,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream_options": {"include_usage": True},
    }


def test_direct_payload_and_non_streaming_override():
    client = ChatClient(Settings(model="m", max_tokens=100))

    payload = client.payload([], think=False, stream=False, max_tokens=7)

    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["max_tokens"] == 7
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "stream_options" not in payload


def test_custom_sampling_profile_is_not_replaced_when_thinking():
    payload = ChatClient(Settings(temperature=0.25, top_p=0.5)).payload([], think=True)

    assert payload["temperature"] == 0.25
    assert payload["top_p"] == 0.5


def test_sse_stream_parses_keepalive_usage_reasoning_content_and_finish(monkeypatch):
    packets = [
        b": keepalive\n",
        b"\n",
        b"event: completion\n",
        data_line({"usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
        data_line({"choices": [{"delta": {"reasoning": "plan "}}]}),
        data_line({"choices": [{"delta": {"reasoning_content": [{"text": "carefully"}]}}]}),
        data_line(
            {
                "choices": [
                    {
                        "delta": {"content": [{"text": "Hello"}, " world"]},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        b"data: [DONE]\n",
    ]
    response = FakeResponse(packets)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return response

    monkeypatch.setattr("mlx_chat.client.urllib.request.urlopen", fake_urlopen)
    client = ChatClient(Settings(model="test-model", request_timeout=123))

    events = list(client.stream_chat([{"role": "user", "content": "hi"}], think=True))

    assert [(event.kind, event.text) for event in events] == [
        ("usage", ""),
        ("reasoning", "plan "),
        ("reasoning", "carefully"),
        ("content", "Hello world"),
        ("finish", ""),
        ("done", ""),
    ]
    assert events[0].usage == {"prompt_tokens": 3, "completion_tokens": 2}
    assert events[-2].finish_reason == "stop"
    assert captured["request"].full_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert captured["request"].get_header("Content-type") == "application/json"
    assert captured["timeout"] == 123
    sent = json.loads(captured["request"].data)
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}


def test_sse_stream_splits_legacy_think_tags_across_packets(monkeypatch):
    response = FakeResponse(
        [
            data_line({"choices": [{"delta": {"content": "Before<th"}}]}),
            data_line({"choices": [{"delta": {"content": "ink>step 1</thi"}}]}),
            data_line({"choices": [{"delta": {"content": "nk>After"}}]}),
            b"data: [DONE]\n",
        ]
    )
    monkeypatch.setattr("mlx_chat.client.urllib.request.urlopen", lambda *_args, **_kw: response)

    events = list(ChatClient(Settings()).stream_chat([], think=True))

    assert [(event.kind, event.text) for event in events] == [
        ("content", "Before"),
        ("reasoning", "step 1"),
        ("content", "After"),
        ("done", ""),
    ]


def test_sse_stream_accepts_message_shape_and_thinking_alias(monkeypatch):
    response = FakeResponse(
        [
            data_line({"choices": [{"message": {"thinking": "idea", "content": "answer"}}]}),
            b"data: [DONE]\n",
        ]
    )
    monkeypatch.setattr("mlx_chat.client.urllib.request.urlopen", lambda *_args, **_kw: response)

    events = list(ChatClient(Settings()).stream_chat([], think=True))

    assert [(event.kind, event.text) for event in events] == [
        ("reasoning", "idea"),
        ("content", "answer"),
        ("done", ""),
    ]


def test_sse_rejects_non_event_stream_response(monkeypatch):
    response = FakeResponse(content_type="application/json", body=b'{"error":"wrong endpoint"}')
    monkeypatch.setattr("mlx_chat.client.urllib.request.urlopen", lambda *_args, **_kw: response)

    with pytest.raises(ApiError, match="Expected an event stream"):
        list(ChatClient(Settings()).stream_chat([], think=False))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (b"data: {not-json}\n", "Invalid streaming JSON"),
        (data_line({"error": "out of memory"}), "generation error"),
        (data_line({"error": {"message": "bad request"}}), "generation error"),
    ],
)
def test_sse_reports_malformed_or_error_packets(monkeypatch, line, message):
    response = FakeResponse([line])
    monkeypatch.setattr("mlx_chat.client.urllib.request.urlopen", lambda *_args, **_kw: response)

    with pytest.raises(ApiError, match=message):
        list(ChatClient(Settings()).stream_chat([], think=False))


def test_sse_rejects_non_object_packets_and_unexpected_eof(monkeypatch):
    non_object = FakeResponse([b"data: []\n", b"data: [DONE]\n"])
    monkeypatch.setattr("mlx_chat.client.urllib.request.urlopen", lambda *_args, **_kw: non_object)
    with pytest.raises(ApiError, match="expected a JSON object"):
        list(ChatClient(Settings()).stream_chat([], think=False))

    truncated = FakeResponse([data_line({"choices": [{"delta": {"content": "partial"}}]})])
    monkeypatch.setattr("mlx_chat.client.urllib.request.urlopen", lambda *_args, **_kw: truncated)
    with pytest.raises(ApiError, match="ended unexpectedly"):
        list(ChatClient(Settings()).stream_chat([], think=False))


def test_collect_accumulates_channels_usage_finish_and_metrics(monkeypatch):
    client = ChatClient(Settings())
    events = iter(
        [
            StreamEvent("reasoning", text="why "),
            StreamEvent("reasoning", text="so"),
            StreamEvent("content", text="answer"),
            StreamEvent("usage", usage={"completion_tokens": 12}),
            StreamEvent("finish", finish_reason="stop"),
            StreamEvent("done"),
        ]
    )
    monkeypatch.setattr(client, "stream_chat", lambda *_args, **_kwargs: events)
    times = iter([10.0, 10.4, 10.9])
    monkeypatch.setattr("mlx_chat.client.time.monotonic", lambda: next(times))
    observed = []

    result = client.collect(
        [], think=True, on_event=lambda event, current: observed.append(event.kind)
    )

    assert result.reasoning == "why so"
    assert result.answer == "answer"
    assert result.usage == {"completion_tokens": 12}
    assert result.finish_reason == "stop"
    assert result.elapsed == pytest.approx(0.9)
    assert result.first_token_seconds == pytest.approx(0.4)
    assert result.tokens_per_second == pytest.approx(12 / 0.9)
    assert observed == ["reasoning", "reasoning", "content", "usage", "finish", "done"]


def test_think_tag_splitter_handles_every_tag_boundary():
    source = "intro<think>reasoning</think>answer"
    for boundary in range(len(source) + 1):
        splitter = ThinkTagSplitter()
        pieces = (
            splitter.feed(source[:boundary]) + splitter.feed(source[boundary:]) + splitter.finish()
        )
        content = "".join(text for kind, text in pieces if kind == "content")
        reasoning = "".join(text for kind, text in pieces if kind == "reasoning")
        assert content == "introanswer"
        assert reasoning == "reasoning"


def test_think_tag_splitter_flushes_incomplete_marker_as_text():
    splitter = ThinkTagSplitter()

    assert splitter.feed("answer<th") == [("content", "answer")]
    assert splitter.finish() == [("content", "<th")]
    assert splitter.finish() == []


def test_cancel_keeps_partial_answer_and_reasoning_and_closes_stream(monkeypatch):
    client = ChatClient(Settings())
    closed = []

    def stream(*args, **kwargs):
        try:
            yield StreamEvent("reasoning", text="Partial reasoning")
            yield StreamEvent("content", text="Partial answer")
            yield StreamEvent("content", text="Should not be reached")
        finally:
            closed.append(True)

    monkeypatch.setattr(client, "stream_chat", stream)

    def cancel(event, result):
        if event.kind == "content":
            raise KeyboardInterrupt

    result = client.collect([], think=True, on_event=cancel)
    assert result.answer == "Partial answer"
    assert result.reasoning == "Partial reasoning"
    assert result.finish_reason == "cancelled"
    assert result.first_token_seconds is not None
    assert closed == [True]
