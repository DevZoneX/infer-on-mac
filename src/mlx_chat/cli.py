"""Command-line entry point for mlx-chat."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import DEFAULT_MODEL, MODEL_PRESETS, __version__, resolve_model_preset
from .client import ChatClient
from .config import Settings, config_path, load_settings, save_settings
from .context import select_context, system_messages
from .errors import ApiError, MlxChatError, ServerUnavailableError
from .server import (
    ServerManager,
    StartResult,
    health,
    latest_log,
    list_records,
    resolve_server_command,
    server_configuration_matches,
    stop_all,
    stop_record,
    tail_log,
)
from .startup import StartupStatus
from .terminal import is_plain_terminal, sanitize_terminal_text, short_model

COMMANDS = {
    "chat",
    "run",
    "serve",
    "start",
    "restart",
    "stop",
    "status",
    "logs",
    "models",
    "doctor",
    "config",
}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class HelpfulParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"mlx-chat: error: {message}\n")


def add_runtime_options(parser: argparse.ArgumentParser, *, positional_model: bool = True) -> None:
    if positional_model:
        parser.add_argument("model", nargs="?", help=f"MLX model (default: {DEFAULT_MODEL})")
    parser.add_argument("-m", "--model-name", dest="model_flag", help="MLX model name or path")
    parser.add_argument("--host", help="Server bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Server port (default: 8080)")
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable separated Qwen reasoning",
    )
    parser.add_argument(
        "--concise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefer a brief direct answer and skip thinking",
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, help="Top-k sampling threshold")
    parser.add_argument("--min-p", type=float, help="Minimum probability sampling threshold")
    parser.add_argument("--max-tokens", type=int, help="Maximum generated tokens")
    parser.add_argument("--system", dest="system_prompt", help="System prompt")
    parser.add_argument("--startup-timeout", type=int, help="Server startup timeout in seconds")
    parser.add_argument("--request-timeout", type=int, help="Generation timeout in seconds")
    parser.add_argument("--connect-timeout", type=int, help="Connection timeout in seconds")
    parser.add_argument("--context-tokens", type=int, help="Model context budget (default: 32768)")
    parser.add_argument(
        "--mlx-arg",
        action="append",
        default=None,
        metavar="ARG",
        help="Extra mlx_lm.server argument (repeat; use --mlx-arg=--flag)",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Acknowledge the risk of binding beyond localhost",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not validate generation before reporting ready",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = HelpfulParser(
        prog="mlx-chat",
        description="Beautiful local MLX inference on your Mac.",
        epilog=(
            "Short form: `mlx-chat [MODEL]` starts the server and opens chat. "
            "Run `mlx-chat stop` to stop every server managed by this app."
        ),
    )
    parser.add_argument("--version", action="version", version=f"mlx-chat {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="Start/reuse a server and open interactive chat")
    add_runtime_options(chat)
    chat.add_argument("--session", help="Load or create an auto-saved session")
    chat.add_argument("--ephemeral", action="store_true", help="Stop the server when chat exits")
    chat.add_argument("--attach", action="store_true", help="Require an already-running server")
    chat.add_argument(
        "--plain", action="store_true", help="Show completed responses without a live UI"
    )
    chat.add_argument(
        "--preload-voice",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load Whisper at startup instead of on first voice input",
    )

    run = subparsers.add_parser("run", help="Run one prompt and exit")
    run.add_argument("prompt", help="Prompt text, or - to read standard input")
    add_runtime_options(run)
    run.add_argument("--json", action="store_true", help="Print structured JSON")
    run.add_argument(
        "--show-reasoning", action="store_true", help="Include reasoning in human output"
    )
    run.add_argument("--ephemeral", action="store_true", help="Stop the server afterward")
    run.add_argument("--attach", action="store_true", help="Require an already-running server")

    serve = subparsers.add_parser(
        "serve", aliases=["start"], help="Start a managed background server"
    )
    add_runtime_options(serve)
    serve.add_argument("--json", action="store_true", help="Print structured JSON")

    restart = subparsers.add_parser("restart", help="Restart a managed server")
    add_runtime_options(restart)

    stop = subparsers.add_parser("stop", help="Stop all managed MLX-LM servers")
    stop.add_argument(
        "--all", action="store_true", help="Accepted for clarity; this is the default"
    )
    stop.add_argument("--port", type=int, help="Only stop the managed server on this port")
    stop.add_argument("--grace", type=float, default=8.0, help="Graceful shutdown seconds")
    stop.add_argument("--json", action="store_true", help="Print structured JSON")

    status = subparsers.add_parser("status", help="Show managed server status")
    status.add_argument("--json", action="store_true", help="Print structured JSON")

    logs = subparsers.add_parser("logs", help="Show the latest server log")
    logs.add_argument("-n", "--lines", type=int, default=80, help="Number of lines")
    logs.add_argument("-f", "--follow", action="store_true", help="Follow new output")

    models = subparsers.add_parser("models", help="List locally cached MLX models")
    models.add_argument("--json", action="store_true", help="Print structured JSON")

    doctor = subparsers.add_parser("doctor", help="Check this Mac and installation")
    doctor.add_argument("--json", action="store_true", help="Print structured JSON")

    config = subparsers.add_parser("config", help="View or change persistent defaults")
    config_sub = config.add_subparsers(dest="config_command")
    config_sub.add_parser("show", help="Print effective configuration")
    config_sub.add_parser("path", help="Print the configuration path")
    set_parser = config_sub.add_parser("set", help="Set one configuration value")
    set_parser.add_argument("key")
    set_parser.add_argument("value")
    config_sub.add_parser("reset", help="Reset persistent defaults")

    return parser


def normalize_argv(argv: Sequence[str]) -> list[str]:
    values = list(argv)
    if not values:
        return ["chat"]
    first = values[0]
    if first in {"-h", "--help", "--version"}:
        return values
    if first not in COMMANDS:
        return ["chat", *values]
    return values


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    positional = getattr(args, "model", None)
    flagged = getattr(args, "model_flag", None)
    model = flagged or positional or os.environ.get("MLX_CHAT_MODEL")
    if model:
        model = resolve_model_preset(model)
    server_args = getattr(args, "mlx_arg", None)
    settings = settings.merged(
        model=model,
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
        think=getattr(args, "think", None),
        concise=getattr(args, "concise", None),
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
        top_k=getattr(args, "top_k", None),
        min_p=getattr(args, "min_p", None),
        max_tokens=getattr(args, "max_tokens", None),
        system_prompt=getattr(args, "system_prompt", None),
        startup_timeout=getattr(args, "startup_timeout", None),
        request_timeout=getattr(args, "request_timeout", None),
        connect_timeout=getattr(args, "connect_timeout", None),
        context_tokens=getattr(args, "context_tokens", None),
        preload_voice=getattr(args, "preload_voice", None),
        server_args=server_args,
    )
    if settings.host not in LOOPBACK_HOSTS and not getattr(args, "allow_network", False):
        raise MlxChatError(
            f"Refusing to bind to {settings.host!r} without --allow-network. MLX-LM's server "
            "is intended for trusted local use and has only basic security checks."
        )
    return settings


def _stop_new_server(
    settings: Settings,
    result: StartResult | None,
    *,
    grace: float = 8.0,
) -> None:
    """Stop only the exact server returned by this invocation."""

    if (
        not result
        or not result.managed
        or result.reused
        or result.pid is None
        or not result.instance_id
    ):
        return
    record = ServerManager(settings).matching_record()
    if record and record.pid == result.pid and record.instance_id == result.instance_id:
        stop_record(record, grace=grace)


def _restore_server(
    settings: Settings,
    *,
    warmup: bool,
    console: Console,
) -> StartResult:
    """Restore a previous model and clean up only a failed new restore."""

    result: StartResult | None = None
    status = console.status(Text(f"Restoring {short_model(settings.model)}…", style="yellow"))
    try:

        def ready_check(_result: StartResult) -> None:
            status.update(Text("Validating the restored model…", style="yellow"))
            ChatClient(settings).warmup(timeout=settings.startup_timeout)

        with status:
            manager = ServerManager(settings)
            result = manager.start(
                progress=lambda message: status.update(Text(f"{message}…", style="yellow")),
                ready_check=ready_check if warmup else None,
            )
    except (Exception, KeyboardInterrupt):
        _stop_new_server(settings, result, grace=2)
        raise
    restored = Text("Restored previous model: ", style="yellow")
    restored.append(settings.model, style="bold")
    console.print(restored)
    return result


def ensure_server(
    settings: Settings,
    *,
    warmup: bool,
    attach_only: bool = False,
    switch_model: bool = True,
    console: Console | None = None,
):
    console = console or Console()
    manager = ServerManager(settings)
    existing = manager.matching_record()
    previous_settings: Settings | None = None
    switch_record = None
    if not attach_only and existing and not server_configuration_matches(existing, settings):
        if not switch_model:
            raise MlxChatError(
                f"Port {settings.port} is running a different managed model or set of "
                "MLX server arguments."
            )
        previous_settings = settings.merged(
            model=existing.model,
            server_args=list(existing.server_args),
        )
        switch_record = existing

    status = StartupStatus(
        console,
        settings.model,
        Text(
            ("Connecting to the local server…" if attach_only else "Preparing the local model…"),
            style="cyan",
        ),
    )
    result: StartResult | None = None
    try:

        def ready_check(_result: StartResult) -> None:
            status.update(
                Text(
                    "Loading the model and preparing the first response…",
                    style="cyan",
                )
            )
            ChatClient(settings).warmup(timeout=settings.startup_timeout)

        if switch_record is not None:
            with console.status(
                Text(f"Stopping {short_model(switch_record.model)}…", style="cyan")
            ):
                stop_result = stop_record(switch_record)
            if not stop_result.stopped and stop_result.detail != "already stopped":
                raise MlxChatError(f"Could not switch models: {stop_result.detail}")
        with status:
            start_or_attach = manager.attach if attach_only else manager.start
            result = start_or_attach(
                progress=lambda message: status.update(Text(f"{message}…", style="cyan")),
                ready_check=ready_check if warmup else None,
            )
        return result
    except (Exception, KeyboardInterrupt) as original_error:
        # A completed StartResult gives us an exact PID. If start() itself was
        # interrupted, ServerManager cleans only the Popen child it owns.
        _stop_new_server(settings, result, grace=2)
        if previous_settings is not None:
            try:
                _restore_server(previous_settings, warmup=warmup, console=console)
            except (Exception, KeyboardInterrupt) as restore_error:
                raise MlxChatError(
                    f"Could not start {settings.model}: {original_error}. "
                    f"Restoring {previous_settings.model} also failed: {restore_error}"
                ) from original_error
        raise


def run_chat(args: argparse.Namespace) -> int:
    from .chat import ChatApp

    settings = settings_from_args(args)
    plain = args.plain or is_plain_terminal()
    console = Console(no_color=plain)
    result = ensure_server(
        settings,
        warmup=not args.no_warmup,
        attach_only=args.attach,
        console=console,
    )
    try:
        app = ChatApp(
            settings,
            result,
            session_name=args.session,
            plain=plain,
        )
    except (Exception, KeyboardInterrupt):
        _stop_new_server(settings, result, grace=2)
        raise
    try:
        return app.run()
    finally:
        if args.ephemeral:
            _stop_new_server(settings, app.start_result)


def run_once(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    if not prompt.strip():
        raise MlxChatError("Prompt cannot be empty.")
    messages = system_messages(settings)
    messages.append({"role": "user", "content": prompt})
    messages = select_context(messages, settings).messages
    console = Console(stderr=True)
    result_info = ensure_server(
        settings,
        warmup=not args.no_warmup,
        attach_only=args.attach,
        console=console,
    )
    try:
        client = ChatClient(settings)
        result = client.collect(messages, think=settings.think and not settings.concise)
        if args.json:
            print(
                json.dumps(
                    {
                        "model": settings.model,
                        "answer": result.answer,
                        "reasoning": result.reasoning,
                        "usage": result.usage,
                        "finish_reason": result.finish_reason,
                        "elapsed": result.elapsed,
                        "first_token_seconds": result.first_token_seconds,
                        "tokens_per_second": result.tokens_per_second,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            if args.show_reasoning and result.reasoning:
                print("Reasoning\n---------")
                print(sanitize_terminal_text(result.reasoning).rstrip())
                print("\nAnswer\n------")
            print(sanitize_terminal_text(result.answer).rstrip())
        if result.finish_reason == "length" and result.reasoning and not result.answer:
            return 5
        return 0
    finally:
        if args.ephemeral:
            _stop_new_server(settings, result_info)


def run_serve(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    console = Console(stderr=args.json)
    result = ensure_server(
        settings,
        warmup=not args.no_warmup,
        console=console,
    )
    value = {
        "status": "ready",
        "model": result.model,
        "host": result.host,
        "port": result.port,
        "pid": result.pid,
        "managed": result.managed,
        "reused": result.reused,
        "log": str(result.log_path) if result.log_path else None,
    }
    if args.json:
        print(json.dumps(value))
    else:
        body = Text()
        body.append("● Ready", style="green")
        body.append("  ")
        body.append(result.model, style="bold")
        body.append(f"\n{settings.base_url}/v1 · PID {result.pid or 'external'}\n")
        body.append("Stop all managed servers with: mlx-chat stop", style="dim")
        console.print(
            Panel.fit(
                body,
                title="mlx-chat server",
                border_style="green",
            )
        )
    return 0


def run_restart(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    console = Console()
    manager = ServerManager(settings)
    record = manager.matching_record()
    previous_settings = (
        settings.merged(model=record.model, server_args=list(record.server_args))
        if record
        else None
    )
    warmup = not args.no_warmup
    try:
        if record:
            with console.status(Text("Stopping current server…", style="cyan")):
                stop_result = stop_record(record)
            if not stop_result.stopped and stop_result.detail != "already stopped":
                raise MlxChatError(stop_result.detail)
        result = ensure_server(settings, warmup=warmup, console=console)
    except (Exception, KeyboardInterrupt) as original_error:
        if previous_settings is not None:
            try:
                _restore_server(previous_settings, warmup=warmup, console=console)
            except (Exception, KeyboardInterrupt) as restore_error:
                raise MlxChatError(
                    f"Restart failed: {original_error}. Restoring {previous_settings.model} "
                    f"also failed: {restore_error}"
                ) from original_error
        raise
    message = Text("✓ ", style="green")
    message.append(f"Ready with {result.model} on {settings.base_url}.")
    console.print(message)
    return 0


def run_stop(args: argparse.Namespace) -> int:
    results = stop_all(port=args.port, grace=args.grace)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "pid": item.record.pid,
                        "model": item.record.model,
                        "port": item.record.port,
                        "stopped": item.stopped,
                        "detail": item.detail,
                    }
                    for item in results
                ]
            )
        )
    else:
        console = Console()
        if not results:
            console.print("[dim]No managed MLX-LM servers are running.[/dim]")
        for item in results:
            line = Text("✓ " if item.stopped else "• ", style="green" if item.stopped else "yellow")
            line.append(f"{item.record.model} · port {item.record.port} · {item.detail}")
            console.print(line)
        if results and all(item.stopped for item in results):
            console.print("[green]All selected servers stopped; model memory is released.[/green]")
    return 0 if all(item.stopped or item.detail == "already stopped" for item in results) else 4


def status_values() -> list[dict[str, Any]]:
    values = []
    for record in list_records():
        values.append(
            {
                "pid": record.pid,
                "model": record.model,
                "host": record.host,
                "port": record.port,
                "healthy": health(record.host, record.port),
                "started_at": record.started_at,
                "log": record.log_path,
            }
        )
    return values


def run_status(args: argparse.Namespace) -> int:
    values = status_values()
    if args.json:
        print(json.dumps(values, indent=2))
        return 0
    console = Console()
    if not values:
        console.print("[dim]No managed MLX-LM servers are running.[/dim]")
        console.print("Start one with: [cyan]mlx-chat[/cyan]")
        return 0
    table = Table("State", "PID", "Model", "Endpoint", "Started", box=box.ROUNDED)
    for item in values:
        state = Text("ready" if item["healthy"] else "unhealthy")
        state.stylize("green" if item["healthy"] else "red")
        table.add_row(
            state,
            Text(str(item["pid"])),
            Text(str(item["model"])),
            Text(f"http://{item['host']}:{item['port']}"),
            Text(str(item["started_at"])),
        )
    console.print(table)
    return 0 if all(item["healthy"] for item in values) else 4


def run_logs(args: argparse.Namespace) -> int:
    if args.lines < 1 or args.lines > 10000:
        raise MlxChatError("--lines must be between 1 and 10000.")
    path = latest_log()
    if not path:
        raise MlxChatError("No MLX-LM server logs were found.")
    print(sanitize_terminal_text(tail_log(path, lines=args.lines)))
    if not args.follow:
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    print(sanitize_terminal_text(line), end="", flush=True)
                else:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        return 0


def cached_models() -> list[str]:
    models: set[str] = set()
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_type == "model" and repo.repo_id:
                models.add(str(repo.repo_id))
    except Exception:
        # Fallback is deliberately conservative and does not import MLX.
        root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        if root.exists():
            for item in root.glob("models--*--*"):
                parts = item.name.split("--")
                if len(parts) >= 3:
                    models.add(f"{parts[1]}/{'--'.join(parts[2:])}")
    return sorted(models)


def run_models(args: argparse.Namespace) -> int:
    models = cached_models()
    if args.json:
        print(json.dumps(models, indent=2))
    else:
        console = Console()
        preset_table = Table("Size", "Recommended model", box=box.SIMPLE, pad_edge=False)
        for size, model in MODEL_PRESETS.items():
            preset_table.add_row(size, model)
        console.print(Panel(preset_table, title="Model presets", border_style="cyan"))
        if not models:
            console.print("[dim]No Hugging Face models are cached yet.[/dim]")
            console.print(f"The first [cyan]mlx-chat[/cyan] run will download {DEFAULT_MODEL}.")
        else:
            table = Table("Cached model", box=box.SIMPLE, pad_edge=False)
            for model in models:
                label = Text(model)
                if model == DEFAULT_MODEL:
                    label.append(" (default)", style="green")
                table.add_row(label)
            console.print(table)
    return 0


def _memory_gib() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size / 1024**3
    except (OSError, ValueError):
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return int(result.stdout.strip()) / 1024**3
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def doctor_values() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        values.append({"check": name, "ok": ok, "detail": detail})

    add("macOS", platform.system() == "Darwin", platform.platform())
    add("Apple Silicon", platform.machine() == "arm64", platform.machine())
    add(
        "Python",
        sys.version_info >= (3, 11),
        f"{platform.python_version()} at {sys.executable}",
    )
    try:
        mlx_version = metadata.version("mlx-lm")
        add("mlx-lm", True, mlx_version)
    except metadata.PackageNotFoundError:
        add("mlx-lm", False, "not installed; run ./setup.sh")
    try:
        command = " ".join(resolve_server_command())
        add("Server command", True, command)
    except MlxChatError as exc:
        add("Server command", False, str(exc))
    memory = _memory_gib()
    add(
        "Unified memory",
        memory is not None and memory >= 8,
        f"{memory:.1f} GiB" if memory else "unknown",
    )
    disk = shutil.disk_usage(Path.cwd())
    free_gib = disk.free / 1024**3
    add("Free disk", free_gib >= 5, f"{free_gib:.1f} GiB")
    records = list_records()
    add("Managed servers", True, f"{len(records)} running")
    return values


def run_doctor(args: argparse.Namespace) -> int:
    values = doctor_values()
    if args.json:
        print(json.dumps(values, indent=2))
    else:
        table = Table("", "Check", "Detail", box=box.ROUNDED)
        for value in values:
            icon = Text("✓" if value["ok"] else "✗", style="green" if value["ok"] else "red")
            table.add_row(icon, Text(value["check"]), Text(value["detail"]))
        Console().print(Panel(table, title="mlx-chat doctor", border_style="blue"))
    return 0 if all(item["ok"] for item in values) else 3


def parse_config_value(current: Any, value: str) -> Any:
    if isinstance(current, bool):
        lowered = value.lower()
        if lowered in {"true", "on", "yes", "1"}:
            return True
        if lowered in {"false", "off", "no", "0"}:
            return False
        raise MlxChatError("Boolean values must be on/off or true/false.")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise MlxChatError("List settings require a JSON array of strings.")
        return parsed
    return value


def run_config(args: argparse.Namespace) -> int:
    command = args.config_command or "show"
    path = config_path()
    if command == "path":
        print(path)
        return 0
    if command == "reset":
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        print(f"Reset configuration. Defaults will be used ({path}).")
        return 0
    settings = load_settings()
    if command == "set":
        key = args.key.replace("-", "_")
        if key not in asdict(settings):
            valid = ", ".join(sorted(asdict(settings)))
            raise MlxChatError(f"Unknown setting {args.key!r}. Valid keys: {valid}")
        value = parse_config_value(getattr(settings, key), args.value)
        setattr(settings, key, value)
        settings.validate()
        save_settings(settings)
        print(f"Saved {args.key} = {value!r}")
        return 0
    print(json.dumps(asdict(settings), ensure_ascii=False, indent=2))
    return 0


def dispatch(args: argparse.Namespace) -> int:
    command = args.command or "chat"
    if command == "chat":
        return run_chat(args)
    if command == "run":
        return run_once(args)
    if command in {"serve", "start"}:
        return run_serve(args)
    if command == "restart":
        return run_restart(args)
    if command == "stop":
        return run_stop(args)
    if command == "status":
        return run_status(args)
    if command == "logs":
        return run_logs(args)
    if command == "models":
        return run_models(args)
    if command == "doctor":
        return run_doctor(args)
    if command == "config":
        return run_config(args)
    raise MlxChatError(f"Unknown command: {command}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    values = normalize_argv(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(values)
    try:
        code = dispatch(args)
    except KeyboardInterrupt:
        Console(stderr=True).print("\n[yellow]Interrupted.[/yellow]")
        code = 130
    except (ApiError, ServerUnavailableError) as exc:
        message = Text("Inference error: ", style="bold red")
        message.append(str(exc))
        Console(stderr=True).print(message)
        code = 5
    except MlxChatError as exc:
        message = Text("Error: ", style="bold red")
        message.append(str(exc))
        Console(stderr=True).print(message)
        code = 4
    except (OSError, ValueError) as exc:
        message = Text("Error: ", style="bold red")
        message.append(str(exc))
        Console(stderr=True).print(message)
        code = 2
    raise SystemExit(code)
