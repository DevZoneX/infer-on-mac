"""Process CPU and resident memory."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .server import ServerRecord, list_records


@dataclass(frozen=True)
class ProcessResources:
    port: int
    cpu_percent: float | None
    memory_mib: float | None


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


def resource_snapshot(port: int) -> ProcessResources | None:
    record = next((item for item in list_records() if item.port == port), None)
    return _process_resources(record) if record is not None else None
