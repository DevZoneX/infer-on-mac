"""Small dependency-free client for the MLX-LM OpenAI-compatible API."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .errors import ApiError, ServerUnavailableError


@dataclass
class StreamEvent:
    kind: str
    text: str = ""
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class GenerationResult:
    answer: str = ""
    reasoning: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    elapsed: float = 0.0
    first_token_seconds: float | None = None

    @property
    def completion_tokens(self) -> int | None:
        value = self.usage.get("completion_tokens")
        return int(value) if isinstance(value, int) else None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or self.elapsed <= 0:
            return None
        return self.completion_tokens / self.elapsed


class ThinkTagSplitter:
    """Split legacy `<think>` streams even when tags cross chunk boundaries."""

    START = "<think>"
    END = "</think>"

    def __init__(self) -> None:
        self._state = "content"
        self._pending = ""

    @staticmethod
    def _partial_suffix(value: str, marker: str) -> int:
        maximum = min(len(value), len(marker) - 1)
        for size in range(maximum, 0, -1):
            if marker.startswith(value[-size:]):
                return size
        return 0

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        if not chunk:
            return []
        self._pending += chunk
        output: list[tuple[str, str]] = []
        while self._pending:
            marker = self.START if self._state == "content" else self.END
            next_state = "reasoning" if self._state == "content" else "content"
            index = self._pending.find(marker)
            if index >= 0:
                if index:
                    output.append((self._state, self._pending[:index]))
                self._pending = self._pending[index + len(marker) :]
                self._state = next_state
                continue
            keep = self._partial_suffix(self._pending, marker)
            ready = self._pending[:-keep] if keep else self._pending
            if ready:
                output.append((self._state, ready))
            self._pending = self._pending[-keep:] if keep else ""
            break
        return output

    def finish(self) -> list[tuple[str, str]]:
        if not self._pending:
            return []
        output = [(self._state, self._pending)]
        self._pending = ""
        return output


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


class ChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def base_url(self) -> str:
        return self.settings.base_url

    def _request(self, path: str, payload: dict[str, Any], *, timeout: float | None = None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(
                request,
                timeout=timeout if timeout is not None else self.settings.request_timeout,
            )
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
                parsed = json.loads(detail)
                detail = parsed.get("error", detail) if isinstance(parsed, dict) else detail
            except (OSError, ValueError):
                detail = str(exc)
            raise ApiError(f"MLX-LM returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ServerUnavailableError(f"Cannot reach {self.base_url}: {reason}") from exc
        except TimeoutError as exc:
            raise ServerUnavailableError(
                f"The request to {self.base_url} timed out after "
                f"{timeout or self.settings.request_timeout} seconds."
            ) from exc

    def payload(
        self,
        messages: list[dict[str, str]],
        *,
        think: bool,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        # Qwen publishes distinct recommended profiles for thinking and direct replies.
        temperature = self.settings.temperature
        top_p = self.settings.top_p
        if think and temperature == 0.7 and top_p == 0.8:
            temperature = 0.6
            top_p = 0.95
        value: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": self.settings.top_k,
            "min_p": self.settings.min_p,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "repetition_penalty": self.settings.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": think},
        }
        if stream:
            value["stream_options"] = {"include_usage": True}
        return value

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        think: bool,
    ) -> Iterator[StreamEvent]:
        splitter = ThinkTagSplitter()
        native_reasoning = False
        saw_done = False
        payload = self.payload(messages, think=think, stream=True)
        with self._request("/v1/chat/completions", payload) as response:
            content_type = response.headers.get("Content-Type", "")
            if response.status != 200:
                raise ApiError(f"MLX-LM returned HTTP {response.status}.")
            if "text/event-stream" not in content_type:
                raw = response.read().decode("utf-8", errors="replace")
                raise ApiError(f"Expected an event stream but received: {raw[:500]}")

            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    packet = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ApiError(f"Invalid streaming JSON from MLX-LM: {data[:300]}") from exc
                if not isinstance(packet, dict):
                    raise ApiError("Invalid streaming event from MLX-LM: expected a JSON object.")
                if isinstance(packet.get("error"), (str, dict)):
                    raise ApiError(f"MLX-LM generation error: {packet['error']}")
                usage = packet.get("usage")
                if isinstance(usage, dict):
                    yield StreamEvent("usage", usage=usage, raw=packet)
                choices = packet.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                finish_reason = choice.get("finish_reason")
                delta = choice.get("delta") or choice.get("message") or {}
                if not isinstance(delta, dict):
                    delta = {}

                reasoning = ""
                for key in ("reasoning", "reasoning_content", "thinking"):
                    reasoning = _extract_text(delta.get(key))
                    if reasoning:
                        break
                if reasoning:
                    native_reasoning = True
                    yield StreamEvent("reasoning", text=reasoning, raw=packet)

                content = _extract_text(delta.get("content"))
                if content:
                    if native_reasoning:
                        yield StreamEvent("content", text=content, raw=packet)
                    else:
                        for kind, text in splitter.feed(content):
                            if text:
                                yield StreamEvent(kind, text=text, raw=packet)
                if finish_reason:
                    yield StreamEvent("finish", finish_reason=str(finish_reason), raw=packet)

        if not saw_done:
            raise ApiError(
                "The MLX-LM stream ended unexpectedly before its completion marker. "
                "The partial response was discarded; check `mlx-chat logs`."
            )
        if not native_reasoning:
            for kind, text in splitter.finish():
                if text:
                    yield StreamEvent(kind, text=text)
        yield StreamEvent("done")

    def collect(
        self,
        messages: list[dict[str, str]],
        *,
        think: bool,
        on_event: Any = None,
    ) -> GenerationResult:
        result = GenerationResult()
        started = time.monotonic()
        first = None
        try:
            for event in self.stream_chat(messages, think=think):
                if event.kind in {"reasoning", "content"} and event.text:
                    if first is None:
                        first = time.monotonic()
                    if event.kind == "reasoning":
                        result.reasoning += event.text
                    else:
                        result.answer += event.text
                elif event.kind == "usage" and event.usage:
                    result.usage.update(event.usage)
                elif event.kind == "finish":
                    result.finish_reason = event.finish_reason
                if on_event is not None:
                    on_event(event, result)
        except (ApiError, ServerUnavailableError):
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise ServerUnavailableError(
                f"The connection to {self.base_url} ended during generation: {exc}"
            ) from exc
        result.elapsed = time.monotonic() - started
        result.first_token_seconds = (first - started) if first is not None else None
        return result

    def warmup(self, *, timeout: float | None = None) -> None:
        payload = self.payload(
            [{"role": "user", "content": "Reply OK."}],
            think=False,
            stream=False,
            max_tokens=1,
        )
        with self._request(
            "/v1/chat/completions",
            payload,
            timeout=timeout or self.settings.startup_timeout,
        ) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                packet = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ApiError(f"Invalid warm-up response from MLX-LM: {raw[:300]}") from exc
            if not isinstance(packet, dict):
                raise ApiError("Invalid warm-up response from MLX-LM: expected a JSON object.")
            if isinstance(packet.get("error"), (str, dict)):
                raise ApiError(f"MLX-LM warm-up failed: {packet['error']}")

    def list_models(self) -> list[str]:
        request = urllib.request.Request(f"{self.base_url}/v1/models")
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                packet = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise ServerUnavailableError(f"Cannot list models from {self.base_url}: {exc}") from exc
        data: Iterable[Any] = packet.get("data", []) if isinstance(packet, dict) else []
        return sorted(
            str(item["id"])
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
