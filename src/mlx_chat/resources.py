"""Dependency-free CPU/RAM snapshots used in the terminal's response metrics."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .server import ServerRecord, list_records


@dataclass(frozen=True)
class ProcessResources:
    port: int
    cpu_percent: float | None
    memory_mib: float | None


@dataclass(frozen=True)
class ResourceSnapshot:
    processes: list[ProcessResources]


def _process_resources(record: ServerRecord) -> ProcessResources:
    try:
        output = subprocess.run(
            ["ps", "-p", str(record.pid), "-o", "%cpu=,rss="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()
        fields = output.split()
        cpu = float(fields[0]) if fields else None
        memory = float(fields[1]) / 1024 if len(fields) > 1 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        cpu = memory = None
    return ProcessResources(record.port, cpu, memory)


def resource_snapshot() -> ResourceSnapshot:
    return ResourceSnapshot(processes=[_process_resources(record) for record in list_records()])
