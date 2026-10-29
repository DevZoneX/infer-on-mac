"""Unified memory sizing for models and the server's prompt cache."""

from __future__ import annotations

import os
import subprocess

# Approximate resident weights of each 4-bit preset, in GiB.
PRESET_WEIGHTS_GIB = {"tiny": 2.5, "small": 5.0, "medium": 17.5, "large": 40.0}
PROMPT_CACHE_FLAGS = ("--prompt-cache-bytes", "--prompt-cache-size")


def memory_gib() -> float | None:
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


def recommended_preset(memory: float | None) -> str:
    """The largest preset that leaves room for macOS, apps, and the KV cache.

    The 30B-A3B mixture-of-experts model activates about 3B parameters per token, so
    it decodes faster than the dense 70B model and is the better agent on big Macs too.
    """
    if memory is None or memory < 16:
        return "tiny"
    if memory < 32:
        return "small"
    return "medium"


def prompt_cache_bytes(memory: float | None) -> int:
    """Bound the server's reusable KV caches to an eighth of memory (1-8 GB)."""
    if memory is None:
        return 2 * 10**9
    return int(max(1, min(8, memory / 8)) * 10**9)


def tuned_server_args(server_args: list[str], memory: float | None) -> list[str]:
    """Add memory-safe defaults that the person has not set themselves.

    MLX-LM keeps up to ten prompt caches with no byte limit. An agent loop with a long
    context can otherwise pin several GB of KV state and push a 16 GB Mac into swap.
    """
    if any(item.split("=", 1)[0] in PROMPT_CACHE_FLAGS for item in server_args):
        return []
    return ["--prompt-cache-bytes", str(prompt_cache_bytes(memory))]
