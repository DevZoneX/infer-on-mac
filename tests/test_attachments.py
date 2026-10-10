from __future__ import annotations

import pytest

from mlx_chat.attachments import (
    MAX_FILE_BYTES,
    MAX_FILES,
    Attachment,
    attachment_prompt,
    merge_attachments,
    read_attachments,
)
from mlx_chat.errors import MlxChatError


def test_file_snapshots_preserve_text_and_do_not_follow_later_edits(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("print('hello')\n", encoding="utf-8")
    files = read_attachments([str(path)])
    path.write_text("changed", encoding="utf-8")
    assert files[0].content == "print('hello')\n"
    prompt = attachment_prompt("Explain this", files)
    assert prompt.startswith("Explain this")
    assert "print('hello')" in prompt and "changed" not in prompt
    assert "untrusted reference data" in prompt
    assert "do not claim to inspect other files or execute code" in prompt


def test_project_scan_skips_hidden_generated_binary_and_symlink_files(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "source.py").write_text("useful code")
    (root / ".env").write_text("private")
    (root / "uv.lock").write_text("generated lock")
    (root / "binary.dat").write_bytes(b"\x00\x01")
    for name in (".git", "node_modules", "dist"):
        (root / name).mkdir()
        (root / name / "ignored.txt").write_text("ignored")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (root / "link.txt").symlink_to(outside)
    (root / "linked-folder").symlink_to(tmp_path, target_is_directory=True)
    files = read_attachments([str(root)])
    assert [item.path for item in files] == [str(root / "source.py")]


def test_explicit_hidden_file_is_allowed(tmp_path):
    path = tmp_path / ".notes"
    path.write_text("explicit text")
    assert read_attachments([str(path)])[0].content == "explicit text"


def test_clipping_at_utf8_boundary_keeps_valid_text_and_marks_truncation(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("a" * (MAX_FILE_BYTES - 1) + "界" + "tail", encoding="utf-8")
    item = read_attachments([str(path)])[0]
    assert item.truncated
    assert item.content == "a" * (MAX_FILE_BYTES - 1)
    assert item.size < MAX_FILE_BYTES


@pytest.mark.parametrize("contents", [b"\x00data", b"\xff\xfe"])
def test_explicit_binary_files_fail_with_friendly_error(tmp_path, contents):
    path = tmp_path / "binary"
    path.write_bytes(contents)
    with pytest.raises(MlxChatError, match="UTF-8"):
        read_attachments([str(path)])


def test_missing_or_empty_projects_have_clear_errors(tmp_path):
    with pytest.raises(MlxChatError, match="Could not attach"):
        read_attachments([str(tmp_path / "missing")])
    with pytest.raises(MlxChatError, match="No readable"):
        read_attachments([str(tmp_path)])


def test_duplicate_paths_are_deduplicated_and_explicit_refresh_replaces_snapshot(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("before")
    original = read_attachments([str(path), str(path)])
    path.write_text("after")
    refreshed = merge_attachments(original, read_attachments([str(path)]))
    assert len(original) == len(refreshed) == 1
    assert original[0].content == "before"
    assert refreshed[0].content == "after"


def test_project_scan_is_bounded_and_explicit_excess_fails(tmp_path):
    paths = []
    for index in range(MAX_FILES + 1):
        path = tmp_path / f"file-{index:02}.txt"
        path.write_text("x")
        paths.append(str(path))
    assert len(read_attachments([str(tmp_path)])) == MAX_FILES
    with pytest.raises(MlxChatError, match="limit"):
        read_attachments(paths)


def test_combined_pending_attachments_are_bounded():
    current = [
        Attachment("first", "a" * MAX_FILE_BYTES),
        Attachment("second", "b" * MAX_FILE_BYTES),
    ]
    with pytest.raises(MlxChatError, match="Use /detach"):
        merge_attachments(current, [Attachment("third", "c")])
    assert len(current) == 2
