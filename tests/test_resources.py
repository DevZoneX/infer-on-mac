from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlx_chat import resources


def test_snapshot_reads_only_displayed_cpu_and_memory(monkeypatch):
    record = SimpleNamespace(pid=123, port=8080)
    monkeypatch.setattr(resources, "list_records", lambda: [record])
    run = Mock(return_value=SimpleNamespace(stdout="12.5 2048"))
    monkeypatch.setattr(resources.subprocess, "run", run)
    process = resources.resource_snapshot().processes[0]
    assert (process.port, process.cpu_percent, process.memory_mib) == (8080, 12.5, 2.0)
    assert run.call_args.args[0] == ["ps", "-p", "123", "-o", "%cpu=,rss="]
    run.assert_called_once()


@pytest.mark.parametrize("output", ["", "invalid data"])
def test_missing_or_invalid_process_metrics_are_optional(monkeypatch, output):
    monkeypatch.setattr(
        resources.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=output)
    )
    result = resources._process_resources(SimpleNamespace(pid=123, port=8080))
    assert result.cpu_percent is None
    assert result.memory_mib is None
