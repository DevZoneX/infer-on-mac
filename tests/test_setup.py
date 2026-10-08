"""Verify installer dependency choices without installing anything."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "options,extras",
    [
        ([], []),
        (["--voice"], ["voice"]),
        (["--web"], ["web"]),
        (["--voice", "--web"], ["voice", "web"]),
    ],
)
def test_installer_keeps_optional_dependencies_opt_in(tmp_path, options, extras):
    commands = tmp_path / "commands"
    commands.mkdir()
    (commands / "uname").write_text(
        '#!/bin/sh\ncase "$1" in -s) echo Darwin ;; -m) echo arm64 ;; esac\n'
    )
    (commands / "uv").write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$TASK_SETUP_LOG"\n')
    for command in commands.iterdir():
        command.chmod(0o755)
    log = tmp_path / "uv-args"
    result = subprocess.run(
        ["/bin/sh", str(ROOT / "setup.sh"), *options],
        env=dict(
            os.environ,
            PATH=f"{commands}:/usr/bin:/bin",
            MLX_CHAT_BIN_DIR=str(tmp_path / "bin"),
            TASK_SETUP_LOG=str(log),
        ),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        "sync",
        "--locked",
        "--python",
        "3.11",
        *[argument for extra in extras for argument in ("--extra", extra)],
    ]


def test_installer_rejects_unknown_options_before_changes(tmp_path):
    result = subprocess.run(
        ["/bin/sh", str(ROOT / "setup.sh"), "--unknown"],
        env=dict(os.environ, MLX_CHAT_BIN_DIR=str(tmp_path / "bin")),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not (tmp_path / "bin").exists()
