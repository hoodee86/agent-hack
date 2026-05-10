"""Tests for phase-3 write skills: apply_patch and write_file."""

from __future__ import annotations

from pathlib import Path

from linux_agent.config import AgentConfig
from linux_agent.skills.write import apply_patch, write_file


def make_config(tmp_path: Path, **kwargs: object) -> AgentConfig:
    return AgentConfig(workspace_root=tmp_path, **kwargs)  # type: ignore[arg-type]


class TestApplyPatch:
    def test_apply_patch_updates_file_and_creates_backup(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        config = make_config(tmp_path)

        result = apply_patch(
            "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: note.txt",
                    "alpha",
                    "-beta",
                    "+gamma",
                    "*** End Patch",
                ]
            ),
            config,
            run_id="run-apply",
        )

        assert result["ok"] is True
        assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"
        assert result["changed_files"] == ["note.txt"]
        assert result["added_lines"] == 1
        assert result["removed_lines"] == 1
        assert len(result["backup_paths"]) == 1
        assert Path(result["backup_paths"][0]).read_text(encoding="utf-8") == "alpha\nbeta\n"
        assert "a/note.txt" in result["diff"]
        assert result["file_summaries"][0]["action"] == "update"

    def test_apply_patch_adds_new_file(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)

        result = apply_patch(
            "\n".join(
                [
                    "*** Begin Patch",
                    "*** Add File: docs/notes.md",
                    "+hello",
                    "+world",
                    "*** End Patch",
                ]
            ),
            config,
            run_id="run-add",
        )

        assert result["ok"] is False
        assert result["error"] == "parent directory does not exist"

        (tmp_path / "docs").mkdir()
        result = apply_patch(
            "\n".join(
                [
                    "*** Begin Patch",
                    "*** Add File: docs/notes.md",
                    "+hello",
                    "+world",
                    "*** End Patch",
                ]
            ),
            config,
            run_id="run-add",
        )

        assert result["ok"] is True
        assert (tmp_path / "docs" / "notes.md").read_text(encoding="utf-8") == "hello\nworld\n"
        assert result["backup_paths"] == []
        assert result["file_summaries"][0]["action"] == "add"

    def test_apply_patch_dry_run_rejects_mismatch_without_partial_write(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        config = make_config(tmp_path)

        result = apply_patch(
            "\n".join(
                [
                    "*** Begin Patch",
                    "*** Add File: staged.txt",
                    "+created",
                    "*** Update File: note.txt",
                    "-missing",
                    "+gamma",
                    "*** End Patch",
                ]
            ),
            config,
            run_id="run-dry",
        )

        assert result["ok"] is False
        assert "context did not match" in result["error"]
        assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
        assert not (tmp_path / "staged.txt").exists()
        assert result["backup_paths"] == []

    def test_apply_patch_rejects_binary_target(self, tmp_path: Path) -> None:
        target = tmp_path / "image.png"
        target.write_bytes(b"\x89PNG\r\n\x00\x00")
        config = make_config(tmp_path)

        result = apply_patch(
            "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: image.png",
                    "-old",
                    "+new",
                    "*** End Patch",
                ]
            ),
            config,
        )

        assert result["ok"] is False
        assert result["error"] == "binary files are not supported"
        assert target.read_bytes() == b"\x89PNG\r\n\x00\x00"

    def test_apply_patch_rejects_oversized_payload(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, max_patch_bytes=32)

        result = apply_patch(
            "*** Begin Patch\n*** Add File: note.txt\n+" + ("x" * 128) + "\n*** End Patch",
            config,
        )

        assert result["ok"] is False
        assert result["error"] == "patch payload exceeds max_patch_bytes"


class TestWriteFile:
    def test_create_only_creates_new_file(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        config = make_config(tmp_path)

        result = write_file(
            "docs/output.md",
            "hello\nworld\n",
            config,
            mode="create_only",
            run_id="run-create",
        )

        assert result["ok"] is True
        assert (tmp_path / "docs" / "output.md").read_text(encoding="utf-8") == "hello\nworld\n"
        assert result["changed_files"] == ["docs/output.md"]
        assert result["backup_paths"] == []
        assert result["created"] is True

    def test_append_updates_existing_file_and_creates_backup(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("alpha\n", encoding="utf-8")
        config = make_config(tmp_path)

        result = write_file(
            "note.txt",
            "beta\n",
            config,
            mode="append",
            run_id="run-append",
        )

        assert result["ok"] is True
        assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
        assert len(result["backup_paths"]) == 1
        assert Path(result["backup_paths"][0]).read_text(encoding="utf-8") == "alpha\n"
        assert result["added_lines"] == 1
        assert result["removed_lines"] == 0

    def test_overwrite_replaces_existing_text(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        config = make_config(tmp_path)

        result = write_file(
            "note.txt",
            "replaced\n",
            config,
            mode="overwrite",
            run_id="run-overwrite",
        )

        assert result["ok"] is True
        assert target.read_text(encoding="utf-8") == "replaced\n"
        assert result["removed_lines"] >= 1
        assert result["backup_paths"]
        assert "a/note.txt" in result["diff"]

    def test_create_only_refuses_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("alpha\n", encoding="utf-8")
        config = make_config(tmp_path)

        result = write_file("note.txt", "beta\n", config, mode="create_only")

        assert result["ok"] is False
        assert result["error"] == "file already exists"
        assert target.read_text(encoding="utf-8") == "alpha\n"

    def test_write_file_rejects_binary_target(self, tmp_path: Path) -> None:
        target = tmp_path / "icon.png"
        target.write_bytes(b"\x00\x01")
        config = make_config(tmp_path)

        result = write_file("icon.png", "hello", config, mode="overwrite")

        assert result["ok"] is False
        assert result["error"] == "binary files are not supported"

    def test_write_file_rejects_oversized_content(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        config = make_config(tmp_path, max_patch_bytes=16)

        result = write_file("docs/output.md", "x" * 32, config, mode="create_only")

        assert result["ok"] is False
        assert result["error"] == "write_file content exceeds max_patch_bytes"