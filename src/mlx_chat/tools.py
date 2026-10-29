"""Workspace tools for the agent loop."""

from __future__ import annotations

import ast
import contextlib
import difflib
import fnmatch
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from . import __version__
from .errors import MlxChatError

APPROVAL_MODES = ("ask", "edits", "all")
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".build",
    "DerivedData",
    "build",
    "dist",
    "target",
}
READ_LINES = 400
MAX_READ_BYTES = 512 * 1024
MAX_WRITE_BYTES = 1024 * 1024
LIST_ENTRIES = 300
SEARCH_MATCHES = 120
SEARCH_FILES = 5000
COMMAND_TIMEOUT = 120
MAX_COMMAND_TIMEOUT = 600
COMMAND_OUTPUT_BYTES = 256 * 1024
FETCH_BYTES = 2 * 1024 * 1024
OUTLINE_FILES = 40
OUTLINE_SYMBOLS = 60
PLAN_MARKS = {"pending": "○", "in_progress": "◐", "done": "●"}
OUTLINE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".swift",
    ".go",
    ".rs",
    ".rb",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".m",
    ".mm",
    ".cs",
    ".php",
    ".md",
    ".sh",
}
DEFINITION_PATTERNS = {
    "js": re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"(?:function\*?\s+\w+|class\s+\w+|interface\s+\w+|type\s+\w+\s*=|enum\s+\w+|"
        r"(?:const|let)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>)"
    ),
    "swift": re.compile(
        r"^\s*(?:@\w+\s+)*(?:(?:public|private|internal|fileprivate|open|static|final|"
        r"override|mutating)\s+)*(?:func|class|struct|enum|protocol|extension|actor)\s+\w+"
    ),
    "go": re.compile(r"^(?:func|type)\s+"),
    "rs": re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl|mod)\b"
    ),
    "rb": re.compile(r"^\s*(?:def|class|module)\s+"),
    "c": re.compile(
        r"^(?:[A-Za-z_][\w\s\*&:<>,]*\s+\**)?[A-Za-z_][\w:~]*\s*\([^;]*\)\s*(?:const\s*)?\{?\s*$"
        r"|^\s*(?:class|struct|enum|namespace|@interface|@implementation)\s+\w+"
    ),
    "sh": re.compile(r"^\s*(?:function\s+)?[\w-]+\s*\(\)\s*\{"),
    "md": re.compile(r"^#{1,4}\s+\S"),
}
SUFFIX_FAMILIES = {
    ".js": "js",
    ".jsx": "js",
    ".ts": "js",
    ".tsx": "js",
    ".mjs": "js",
    ".swift": "swift",
    ".go": "go",
    ".rs": "rs",
    ".rb": "rb",
    ".java": "c",
    ".kt": "c",
    ".c": "c",
    ".h": "c",
    ".cc": "c",
    ".cpp": "c",
    ".hpp": "c",
    ".m": "c",
    ".mm": "c",
    ".cs": "c",
    ".php": "c",
    ".md": "md",
    ".sh": "sh",
}


class ToolError(MlxChatError):
    """A tool call that the model can correct and retry."""


@dataclass
class ToolOutcome:
    """The text returned to the model and a short line for the transcript."""

    content: str
    summary: str
    ok: bool = True
    diff: str = ""
    cancelled: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    error: str = ""

    @classmethod
    def from_api(cls, value: dict[str, Any], index: int) -> ToolCall:
        function = value.get("function") if isinstance(value.get("function"), dict) else {}
        name = str(function.get("name") or "")
        raw = function.get("arguments", "")
        call_id = str(value.get("id") or f"call_{index}")
        if isinstance(raw, dict):
            return cls(call_id, name, raw, json.dumps(raw, ensure_ascii=False))
        raw = raw if isinstance(raw, str) else ""
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            return cls(call_id, name, {}, raw, f"Arguments are not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            return cls(call_id, name, {}, raw, "Arguments must be a JSON object.")
        return cls(call_id, name, parsed, raw)

    def api_value(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments or json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., ToolOutcome]
    kind: str = "read"  # read, edit, or command

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ToolError(f"{name} must be a non-empty string.")
    return value


def _integer(value: Any, name: str, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ToolError(f"{name} must be an integer.")
    return max(minimum, min(maximum, int(value)))


def _is_binary(raw: bytes) -> bool:
    return b"\x00" in raw[:8192]


def _shell() -> list[str]:
    for candidate in ("/bin/zsh", "/bin/bash", "/bin/sh"):
        if Path(candidate).exists():
            return [candidate, "-c"]
    return ["sh", "-c"]


class Workspace:
    """Tools confined to one project directory."""

    def __init__(self, root: Path, *, output_chars: int = 12000, web: Any = None):
        try:
            self.root = Path(root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise MlxChatError(f"Workspace {root} is not available: {exc}") from exc
        if not self.root.is_dir():
            raise MlxChatError(f"Workspace {root} is not a directory.")
        self.output_chars = output_chars
        self.web = web
        self.plan: list[dict[str, str]] = []
        self.tools: dict[str, Tool] = {item.name: item for item in self._definitions()}

    # Paths

    def resolve(self, value: Any, *, must_exist: bool = True) -> Path:
        name = _text(value if value not in (None, "") else ".", "path")
        candidate = Path(name).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise ToolError(f"{name} does not exist.") from exc
        except (OSError, RuntimeError) as exc:
            raise ToolError(f"Cannot resolve {name}: {exc}") from exc
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError(f"{name} is outside the workspace {self.root}.")
        return resolved

    def relative(self, path: Path) -> str:
        if path == self.root:
            return "."
        return str(path.relative_to(self.root))

    def clip(self, text: str) -> str:
        if len(text) <= self.output_chars:
            return text
        head = self.output_chars * 2 // 3
        tail = self.output_chars - head
        omitted = len(text) - head - tail
        return f"{text[:head]}\n… [{omitted} characters omitted] …\n{text[-tail:]}"

    # Registry

    def schemas(self) -> list[dict[str, Any]]:
        return [item.schema() for item in self.tools.values()]

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def needs_approval(self, name: str, mode: str, allowed: set[str]) -> bool:
        tool = self.tools.get(name)
        if tool is None or tool.kind in {"read", "plan"} or name in allowed or mode == "all":
            return False
        return not (mode == "edits" and tool.kind == "edit")

    def execute(self, call: ToolCall, *, cancel_event: Any = None) -> ToolOutcome:
        if call.error:
            return ToolOutcome(f"Error: {call.error}", call.error, ok=False)
        tool = self.tools.get(call.name)
        if tool is None:
            names = ", ".join(self.tools)
            message = f"Unknown tool {call.name!r}. Available tools: {names}."
            return ToolOutcome(f"Error: {message}", message, ok=False)
        try:
            arguments = dict(call.arguments)
            if tool.kind in {"command", "web"}:
                arguments["cancel_event"] = cancel_event
            outcome = tool.run(**arguments)
        except TypeError as exc:
            message = f"Invalid arguments for {call.name}: {exc}"
            return ToolOutcome(f"Error: {message}", message, ok=False)
        except (ToolError, MlxChatError, OSError, UnicodeError, re.error) as exc:
            return ToolOutcome(f"Error: {exc}", str(exc), ok=False)
        outcome.content = self.clip(outcome.content)
        return outcome

    def preview(self, call: ToolCall) -> ToolOutcome | None:
        """Describe a change before it is approved; None when no preview is available."""
        try:
            if call.name == "write_file":
                path = self.resolve(call.arguments.get("path"), must_exist=False)
                content = _text(call.arguments.get("content", ""), "content", allow_empty=True)
                before = self._read_text(path) if path.exists() else ""
                diff = self._diff(path, before, content)
                return ToolOutcome("", self.relative(path), diff=diff)
            if call.name == "edit_file":
                path, before, after, _count = self._planned_edit(**call.arguments)
                return ToolOutcome("", self.relative(path), diff=self._diff(path, before, after))
        except (ToolError, TypeError, OSError, UnicodeError):
            return None
        return None

    def _definitions(self) -> list[Tool]:
        path = {"type": "string", "description": "Path relative to the workspace root."}
        return [
            Tool(
                "list_files",
                "List files and folders as a tree. Skips .git, virtualenvs, caches and builds.",
                _object(
                    {
                        "path": {**path, "description": "Folder to list (default: root)."},
                        "depth": {"type": "integer", "description": "Levels deep, 1-5 (2)."},
                    },
                    [],
                ),
                self.list_files,
            ),
            Tool(
                "read_file",
                "Read a UTF-8 text file with line numbers. Long files are returned in pages; "
                "use start_line to continue.",
                _object(
                    {
                        "path": path,
                        "start_line": {"type": "integer", "description": "First line (1)."},
                        "end_line": {"type": "integer", "description": "Last line, inclusive."},
                    },
                    ["path"],
                ),
                self.read_file,
            ),
            Tool(
                "search",
                "Search file contents with a regular expression (ripgrep syntax). Returns "
                "path:line: text matches.",
                _object(
                    {
                        "pattern": {"type": "string", "description": "Regular expression."},
                        "path": {**path, "description": "Folder or file to search (root)."},
                        "glob": {"type": "string", "description": "File filter, e.g. *.py"},
                        "ignore_case": {"type": "boolean"},
                    },
                    ["pattern"],
                ),
                self.search,
            ),
            Tool(
                "edit_file",
                "Replace an exact text snippet in a file. old_text must match once, including "
                "whitespace; read the file first. Set replace_all to change every match.",
                _object(
                    {
                        "path": path,
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    ["path", "old_text", "new_text"],
                ),
                self.edit_file,
                "edit",
            ),
            Tool(
                "write_file",
                "Create a file or replace its whole content. Prefer edit_file for small changes.",
                _object({"path": path, "content": {"type": "string"}}, ["path", "content"]),
                self.write_file,
                "edit",
            ),
            Tool(
                "run_command",
                "Run a shell command (zsh) in the workspace root and return its exit code and "
                "output. Non-interactive: stdin is closed. Use for tests, builds, git and "
                "inspection.",
                _object(
                    {
                        "command": {"type": "string"},
                        "timeout": {
                            "type": "integer",
                            "description": f"Seconds before it is stopped ({COMMAND_TIMEOUT}).",
                        },
                    },
                    ["command"],
                ),
                self.run_command,
                "command",
            ),
            Tool(
                "find_files",
                "Find files by name with a glob pattern, e.g. *.swift, src/**/*.py, or "
                "**/test_*. Patterns without a slash match file names anywhere.",
                _object(
                    {
                        "pattern": {"type": "string"},
                        "path": {**path, "description": "Folder to search (default: root)."},
                    },
                    ["pattern"],
                ),
                self.find_files,
            ),
            Tool(
                "outline",
                "List the classes, functions and other definitions in a file or folder with "
                "line numbers. Use it to find code before reading whole files.",
                _object({"path": path}, ["path"]),
                self.outline,
            ),
            Tool(
                "update_plan",
                "Record the plan for a multi-step task and mark progress. Send every step each "
                "time, with status pending, in_progress, or done. Keep one step in_progress.",
                _object(
                    {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "done"],
                                    },
                                },
                                "required": ["step", "status"],
                            },
                        }
                    },
                    ["steps"],
                ),
                self.update_plan,
                "plan",
            ),
            *self._web_definitions(),
        ]

    def _web_definitions(self) -> list[Tool]:
        if self.web is None:
            return []
        tools = [
            Tool(
                "fetch_url",
                "Download a web page or text file (http or https) and return its readable "
                "text. Use for documentation and error messages.",
                _object({"url": {"type": "string"}}, ["url"]),
                self.fetch_url,
                "web",
            )
        ]
        if importlib.util.find_spec("ddgs") is not None:
            tools.insert(
                0,
                Tool(
                    "web_search",
                    "Search the web. Returns titles, URLs and snippets; fetch a URL for "
                    "the full page.",
                    _object({"query": {"type": "string"}}, ["query"]),
                    self.web_search,
                    "web",
                ),
            )
        return tools

    # Read tools

    def _walk(self, start: Path):
        """Yield files below start in sorted order, skipping ignored folders."""
        pending = [start]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
            except OSError:
                continue
            folders = []
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in IGNORED_DIRS:
                            folders.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        yield Path(entry.path)
                except OSError:
                    continue
            pending.extend(reversed(folders))

    def list_files(self, path: str = ".", depth: int | None = None) -> ToolOutcome:
        root = self.resolve(path)
        if not root.is_dir():
            raise ToolError(f"{path} is not a folder.")
        levels = _integer(depth, "depth", 2, 1, 5)
        lines = [f"{self.relative(root)}/"]
        count = 0
        truncated = False

        def visit(directory: Path, prefix: str, level: int) -> None:
            nonlocal count, truncated
            try:
                entries = sorted(
                    (item for item in directory.iterdir() if item.name not in IGNORED_DIRS),
                    key=lambda item: (not item.is_dir(), item.name.lower()),
                )
            except OSError:
                return
            for item in entries:
                if count >= LIST_ENTRIES:
                    truncated = True
                    return
                count += 1
                if item.is_dir() and not item.is_symlink():
                    lines.append(f"{prefix}{item.name}/")
                    if level < levels:
                        visit(item, prefix + "  ", level + 1)
                else:
                    lines.append(f"{prefix}{item.name}")

        visit(root, "  ", 1)
        if truncated:
            lines.append(f"… stopped after {LIST_ENTRIES} entries; list a subfolder for more.")
        return ToolOutcome("\n".join(lines), plural(count, "entry", "entries"))

    def _read_text(self, path: Path) -> str:
        if not path.is_file():
            raise ToolError(f"{self.relative(path)} is not a regular file.")
        size = path.stat().st_size
        if size > MAX_READ_BYTES * 8:
            raise ToolError(f"{self.relative(path)} is too large to read ({size} bytes).")
        raw = path.read_bytes()
        if _is_binary(raw):
            raise ToolError(f"{self.relative(path)} looks like a binary file.")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"{self.relative(path)} is not UTF-8 text.") from exc

    def read_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> ToolOutcome:
        target = self.resolve(path)
        lines = self._read_text(target).splitlines()
        total = len(lines)
        first = _integer(start_line, "start_line", 1, 1, max(1, total))
        last_default = min(total, first + READ_LINES - 1)
        last = _integer(end_line, "end_line", last_default, first, total) if total else 0
        last = min(last, first + READ_LINES - 1)
        width = len(str(max(1, last)))
        body = "\n".join(
            f"{number:>{width}}│{lines[number - 1]}" for number in range(first, last + 1)
        )
        name = self.relative(target)
        span = f"lines {first}-{last} of {total}" if total else "empty file"
        header = f"{name} · {span}"
        footer = (
            f"\n… {total - last} more lines; call read_file with start_line={last + 1}."
            if last < total
            else ""
        )
        return ToolOutcome(f"{header}\n{body}{footer}", span)

    def search(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignore_case: bool = False,
    ) -> ToolOutcome:
        _text(pattern, "pattern")
        root = self.resolve(path)
        if glob is not None:
            _text(glob, "glob")
        ripgrep = shutil.which("rg")
        matches = (
            self._ripgrep(ripgrep, pattern, root, glob, ignore_case)
            if ripgrep
            else self._python_search(pattern, root, glob, ignore_case)
        )
        if not matches:
            return ToolOutcome(f"No matches for {pattern!r}.", "no matches")
        extra = ""
        if len(matches) > SEARCH_MATCHES:
            extra = f"\n… more than {SEARCH_MATCHES} matches; narrow the pattern or path."
            matches = matches[:SEARCH_MATCHES]
        files = len({line.split(":", 1)[0] for line in matches})
        return ToolOutcome(
            "\n".join(matches) + extra,
            f"{plural(len(matches), 'match', 'matches')} in {plural(files, 'file')}",
        )

    def _ripgrep(self, binary, pattern, root, glob, ignore_case) -> list[str]:
        command = [
            binary,
            "--line-number",
            "--no-heading",
            "--color=never",
            "--max-columns=300",
            "--max-count=20",
        ]
        for name in sorted(IGNORED_DIRS):
            command += ["--glob", f"!{name}/"]
        if glob:
            command += ["--glob", glob]
        if ignore_case:
            command.append("--ignore-case")
        command += ["--regexp", pattern, "--", str(root)]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False, cwd=self.root
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError("Search timed out after 30 seconds; narrow the path.") from exc
        if result.returncode == 2 and not result.stdout:
            raise ToolError(result.stderr.strip() or "ripgrep failed.")
        output = []
        for line in result.stdout.splitlines()[: SEARCH_MATCHES + 1]:
            if line.startswith(str(self.root) + os.sep):
                line = line[len(str(self.root)) + 1 :]
            output.append(line)
        return output

    def _python_search(self, pattern, root, glob, ignore_case) -> list[str]:
        try:
            expression = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"Invalid regular expression: {exc}") from exc
        files = [root] if root.is_file() else self._walk(root)
        output: list[str] = []
        for scanned, file in enumerate(files):
            if scanned >= SEARCH_FILES or len(output) > SEARCH_MATCHES:
                break
            if glob and not fnmatch.fnmatch(file.name, glob):
                continue
            try:
                if file.stat().st_size > MAX_READ_BYTES:
                    continue
                raw = file.read_bytes()
            except OSError:
                continue
            if _is_binary(raw):
                continue
            text = raw.decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                if expression.search(line):
                    output.append(f"{self.relative(file)}:{number}:{line[:300]}")
                    if len(output) > SEARCH_MATCHES:
                        break
        return output

    def find_files(self, pattern: str, path: str = ".") -> ToolOutcome:
        _text(pattern, "pattern")
        root = self.resolve(path)
        if not root.is_dir():
            raise ToolError(f"{path} is not a folder.")
        nested = "/" in pattern
        # "**/x" also matches x at the top level.
        alternatives = [pattern, pattern[3:]] if pattern.startswith("**/") else [pattern]
        found = []
        for file in self._walk(root):
            relative = file.relative_to(root).as_posix()
            target = relative if nested else file.name
            if any(fnmatch.fnmatch(target, item) for item in alternatives):
                found.append(self.relative(file))
                if len(found) > LIST_ENTRIES:
                    break
        if not found:
            return ToolOutcome(f"No files match {pattern!r}.", "no files")
        extra = ""
        if len(found) > LIST_ENTRIES:
            found = found[:LIST_ENTRIES]
            extra = f"\n… more than {LIST_ENTRIES} files; narrow the pattern."
        return ToolOutcome("\n".join(found) + extra, plural(len(found), "file"))

    def outline(self, path: str) -> ToolOutcome:
        target = self.resolve(path)
        files = (
            [target]
            if target.is_file()
            else [file for file in self._walk(target) if file.suffix in OUTLINE_SUFFIXES]
        )
        sections = []
        symbols = 0
        for file in files[:OUTLINE_FILES]:
            try:
                text = self._read_text(file)
            except ToolError:
                continue
            entries = outline_text(text, file.suffix)
            if entries:
                symbols += len(entries)
                listed = entries[:OUTLINE_SYMBOLS]
                more = len(entries) - len(listed)
                body = "\n".join(listed) + (f"\n  … {more} more" if more > 0 else "")
                sections.append(f"{self.relative(file)}\n{body}")
        if not sections:
            return ToolOutcome("No definitions found.", "no definitions")
        if len(files) > OUTLINE_FILES:
            sections.append(f"… {len(files) - OUTLINE_FILES} more files; outline a subfolder.")
        return ToolOutcome(
            "\n\n".join(sections),
            f"{plural(symbols, 'definition')} in {plural(min(len(files), OUTLINE_FILES), 'file')}",
        )

    def update_plan(self, steps: list[dict[str, Any]]) -> ToolOutcome:
        if not isinstance(steps, list) or not steps:
            raise ToolError("steps must be a non-empty list.")
        plan = []
        for item in steps[:30]:
            if not isinstance(item, dict):
                raise ToolError("Each step needs step and status.")
            text = _text(item.get("step"), "step").strip()
            status = item.get("status", "pending")
            if status not in {"pending", "in_progress", "done"}:
                raise ToolError("status must be pending, in_progress, or done.")
            plan.append({"step": text[:200], "status": status})
        self.plan = plan
        done = sum(1 for item in plan if item["status"] == "done")
        lines = [f"{PLAN_MARKS[item['status']]} {item['step']}" for item in plan]
        return ToolOutcome("Plan updated.\n" + "\n".join(lines), f"{done}/{len(plan)} done")

    # Web tools

    def web_search(self, query: str, cancel_event: Any = None) -> ToolOutcome:
        from .web import WebSearcher

        result = WebSearcher(self.web).search(_text(query, "query"))
        lines = [
            f"{number}. {source.title}\n   {source.url}\n   {source.snippet}"
            for number, source in enumerate(result.sources, 1)
        ]
        return ToolOutcome(
            "Web results (untrusted):\n" + "\n".join(lines), f"{len(result.sources)} results"
        )

    def fetch_url(self, url: str, cancel_event: Any = None) -> ToolOutcome:
        _text(url, "url")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ToolError("Only http and https URLs can be fetched.")
        request = urllib.request.Request(
            url, headers={"User-Agent": f"mlx-chat/{__version__} (+local agent)", "Accept": "*/*"}
        )
        timeout = getattr(self.web, "web_timeout", 10) + 5
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                kind = response.headers.get("Content-Type", "")
                raw = response.read(FETCH_BYTES + 1)
                final = response.geturl()
        except urllib.error.HTTPError as exc:
            raise ToolError(f"{url} returned HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ToolError(f"Could not fetch {url}: {getattr(exc, 'reason', exc)}") from exc
        if _is_binary(raw) and "text" not in kind and "json" not in kind:
            raise ToolError(f"{url} is not a text page ({kind or 'unknown type'}).")
        charset = "utf-8"
        match = re.search(r"charset=([\w-]+)", kind)
        if match:
            charset = match.group(1)
        try:
            text = raw[:FETCH_BYTES].decode(charset, errors="replace")
        except LookupError:
            text = raw[:FETCH_BYTES].decode("utf-8", errors="replace")
        title = ""
        if "html" in kind or text.lstrip().lower().startswith(("<!doctype html", "<html")):
            title, text = html_to_text(text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        header = f"{title}\n{final}" if title else final
        return ToolOutcome(
            f"Fetched page (untrusted):\n{header}\n\n{text}",
            f"{len(text):,} characters" + (f" · {title[:60]}" if title else ""),
        )

    # Edit tools

    def _diff(self, path: Path, before: str, after: str) -> str:
        name = self.relative(path)
        lines = difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
            n=2,
        )
        text = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
        return text[:20000]

    @staticmethod
    def _diff_counts(diff: str) -> str:
        added = sum(
            1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
        )
        removed = sum(
            1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")
        )
        return f"+{added} −{removed}"

    def _write_atomic(self, path: Path, content: str) -> None:
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ToolError(f"Content is too large ({len(data)} bytes).")
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = path.stat().st_mode & 0o7777 if path.exists() else None
        temporary = path.with_name(f".{path.name}.mlx-chat-tmp")
        try:
            temporary.write_bytes(data)
            if mode is not None:
                os.chmod(temporary, mode)
            os.replace(temporary, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    def write_file(self, path: str, content: str) -> ToolOutcome:
        target = self.resolve(path, must_exist=False)
        _text(content, "content", allow_empty=True)
        if target.exists() and not target.is_file():
            raise ToolError(f"{path} exists and is not a regular file.")
        existed = target.exists()
        before = self._read_text(target) if existed else ""
        self._write_atomic(target, content)
        diff = self._diff(target, before, content)
        verb = "Updated" if existed else "Created"
        name = self.relative(target)
        lines = len(content.splitlines())
        return ToolOutcome(
            f"{verb} {name} ({lines} lines).",
            f"{verb.lower()} · {lines} lines · {self._diff_counts(diff)}",
            diff=diff,
        )

    def _planned_edit(
        self, path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> tuple[Path, str, str, int]:
        target = self.resolve(path)
        _text(old_text, "old_text")
        _text(new_text, "new_text", allow_empty=True)
        before = self._read_text(target)
        count = before.count(old_text)
        if count == 0:
            stripped = old_text.strip()
            hint = (
                " The text exists with different surrounding whitespace."
                if stripped and stripped in before
                else ""
            )
            raise ToolError(
                f"old_text was not found in {self.relative(target)}.{hint} "
                "Read the file and copy the exact text."
            )
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_text matches {count} places in {self.relative(target)}. Include more "
                "surrounding lines to make it unique, or set replace_all."
            )
        after = (
            before.replace(old_text, new_text)
            if replace_all
            else before.replace(old_text, new_text, 1)
        )
        return target, before, after, count if replace_all else 1

    def edit_file(
        self, path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> ToolOutcome:
        target, before, after, count = self._planned_edit(path, old_text, new_text, replace_all)
        self._write_atomic(target, after)
        diff = self._diff(target, before, after)
        name = self.relative(target)
        places = f" in {count} places" if count > 1 else ""
        return ToolOutcome(
            f"Edited {name}{places}.",
            self._diff_counts(diff) + (f" · {count} places" if count > 1 else ""),
            diff=diff,
        )

    # Commands

    def run_command(
        self, command: str, timeout: int | None = None, cancel_event: Any = None
    ) -> ToolOutcome:
        _text(command.strip() if isinstance(command, str) else command, "command")
        limit = _integer(timeout, "timeout", COMMAND_TIMEOUT, 1, MAX_COMMAND_TIMEOUT)
        environment = {
            **os.environ,
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "TERM": "dumb",
            "NO_COLOR": "1",
            "CLICOLOR": "0",
        }
        started = time.monotonic()
        process = subprocess.Popen(
            [*_shell(), command],
            cwd=self.root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        chunks: list[bytes] = []
        size = 0
        dropped = 0

        def read() -> None:
            nonlocal size, dropped
            assert process.stdout is not None
            for chunk in iter(lambda: process.stdout.read1(65536), b""):
                if size < COMMAND_OUTPUT_BYTES:
                    chunks.append(chunk)
                    size += len(chunk)
                else:
                    dropped += len(chunk)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reason = ""
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                reason = "cancelled"
            elif time.monotonic() - started > limit:
                reason = f"timed out after {limit}s"
            if reason:
                self._stop(process)
                break
            time.sleep(0.05)
        process.wait()
        reader.join(timeout=2)
        elapsed = time.monotonic() - started
        output = b"".join(chunks).decode("utf-8", errors="replace").rstrip()
        if dropped:
            output += f"\n… [{dropped} more bytes of output were dropped]"
        code = process.returncode
        status = reason or f"exit {code}"
        content = f"$ {command}\n{output or '(no output)'}\n[{status} · {elapsed:.1f}s]"
        return ToolOutcome(
            content,
            f"{status} · {elapsed:.1f}s",
            ok=not reason and code == 0,
            cancelled=reason == "cancelled",
        )

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        for sent in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, sent)
            try:
                process.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                continue


def plural(count: int, singular: str, many: str | None = None) -> str:
    return f"{count} {singular if count == 1 else many or singular + 's'}"


def outline_text(text: str, suffix: str) -> list[str]:
    """Definitions with line numbers: exact for Python, pattern-based elsewhere."""
    if suffix == ".py":
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            return list(_python_outline(tree.body, 0))
    family = SUFFIX_FAMILIES.get(suffix)
    pattern = DEFINITION_PATTERNS.get(family or "")
    if pattern is None:
        return []
    entries = []
    fenced = False
    for number, line in enumerate(text.splitlines(), 1):
        if family == "md" and line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced:
            continue
        if pattern.search(line) and len(line) < 300:
            entries.append(f"  {number:>5}  {line.strip().rstrip('{').strip()}")
    return entries


def _python_outline(nodes, depth: int):
    for node in nodes:
        indent = "  " * depth
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            yield f"  {node.lineno:>5}  {indent}class {node.name}" + (f"({bases})" if bases else "")
            yield from _python_outline(node.body, depth + 1)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            arguments = ast.unparse(node.args)
            if len(arguments) > 80:
                arguments = arguments[:79] + "…"
            yield f"  {node.lineno:>5}  {indent}{prefix} {node.name}({arguments})"


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "head"}
    BLOCKS = {
        "p",
        "div",
        "section",
        "article",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "blockquote",
        "header",
        "footer",
        "table",
        "ul",
        "ol",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3"}:
            self.parts.append("# ")
        elif tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> tuple[str, str]:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except (AssertionError, ValueError):
        pass
    lines = [
        re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in "".join(parser.parts).split("\n")
    ]
    return parser.title.strip(), "\n".join(lines)


def describe_call(call: ToolCall) -> str:
    """A compact one-line description of a tool call for the transcript."""
    arguments = call.arguments
    if call.name == "run_command":
        lines = str(arguments.get("command", "")).strip().splitlines()
        return lines[0][:160] + (" …" if len(lines) > 1 else "") if lines else ""
    if call.name == "search":
        where = arguments.get("path")
        return f"{arguments.get('pattern', '')!r}" + (f" in {where}" if where else "")
    if call.name == "read_file":
        span = ""
        if arguments.get("start_line") or arguments.get("end_line"):
            span = f" · {arguments.get('start_line', 1)}-{arguments.get('end_line', '…')}"
        return f"{arguments.get('path', '')}{span}"
    if call.name == "find_files":
        return str(arguments.get("pattern", ""))
    if call.name in {"web_search"}:
        return str(arguments.get("query", ""))[:160]
    if call.name == "fetch_url":
        return str(arguments.get("url", ""))[:160]
    if call.name == "update_plan":
        return ""
    if "path" in arguments or call.name == "list_files":
        return str(arguments.get("path") or ".")
    return call.raw_arguments[:160]
