"""
Unit tests for T5 (policy), T6 (list_dir), T7 (read_file), T8 (search_text).

All tests use tmp_path fixtures; no real filesystem state is shared.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from linux_agent.config import AgentConfig
from linux_agent.policy import (
    PolicyViolation,
    evaluate_tool_call,
    resolve_safe_path,
)
from linux_agent.skills.filesystem import list_dir, read_file
from linux_agent.skills.search import search_text
from linux_agent.state import ToolCall


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_config(tmp_path: Path, **kwargs: object) -> AgentConfig:
    return AgentConfig(workspace_root=tmp_path, **kwargs)  # type: ignore[arg-type]


def make_tool_call(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(
        id=f"{name}_test",
        name=name,
        args=args or {},
        risk_level="low",
    )


# ─────────────────────────────────────────────────────────────────────────────
# T5 – policy.resolve_safe_path
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveSafePath:
    def test_normal_relative_path_allowed(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        result = resolve_safe_path(tmp_path, "src/main.py")
        assert result == (tmp_path / "src" / "main.py").resolve()

    def test_root_itself_allowed(self, tmp_path: Path) -> None:
        result = resolve_safe_path(tmp_path, ".")
        assert result == tmp_path.resolve()

    def test_parent_traversal_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyViolation, match="escapes workspace root"):
            resolve_safe_path(tmp_path, "../secret")

    def test_absolute_path_outside_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyViolation, match="escapes workspace root"):
            resolve_safe_path(tmp_path, "/etc/shadow")

    def test_absolute_path_inside_allowed(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        result = resolve_safe_path(tmp_path, str(sub))
        assert result == sub.resolve()

    def test_sensitive_ssh_rejected(self, tmp_path: Path) -> None:
        (tmp_path / ".ssh").mkdir()
        with pytest.raises(PolicyViolation, match="sensitive"):
            resolve_safe_path(tmp_path, ".ssh/id_rsa")

    def test_sensitive_shadow_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "etc").mkdir()
        (tmp_path / "etc" / "shadow").touch()
        with pytest.raises(PolicyViolation, match="sensitive"):
            resolve_safe_path(tmp_path, "etc/shadow")

    def test_symlink_inside_workspace_allowed(self, tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.write_text("hi")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        result = resolve_safe_path(tmp_path, "link.txt")
        assert result == target.resolve()

    def test_symlink_escaping_workspace_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret")
        link = tmp_path / "evil_link"
        link.symlink_to(outside)
        with pytest.raises(PolicyViolation, match="escapes workspace root"):
            resolve_safe_path(tmp_path, "evil_link")

    def test_nonexistent_relative_path_allowed(self, tmp_path: Path) -> None:
        # Non-existent paths within the workspace should be allowed (e.g. new files)
        result = resolve_safe_path(tmp_path, "new_file.txt")
        assert str(result).startswith(str(tmp_path))

    def test_nonexistent_traversal_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyViolation, match="escapes workspace root"):
            resolve_safe_path(tmp_path, "../../outside.txt")


# ─────────────────────────────────────────────────────────────────────────────
# T5 – evaluate_tool_call
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateToolCall:
    def test_read_file_allowed(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        tc = make_tool_call("read_file", {"path": "README.md"})
        assert evaluate_tool_call(tc, cfg) == "allow"

    def test_list_dir_allowed(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        tc = make_tool_call("list_dir", {"path": "."})
        assert evaluate_tool_call(tc, cfg) == "allow"

    def test_write_file_denied(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        tc = make_tool_call("write_file", {"path": "out.txt", "content": "x"})
        assert evaluate_tool_call(tc, cfg) == "deny"

    def test_path_traversal_denied(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        tc = make_tool_call("read_file", {"path": "../etc/passwd"})
        assert evaluate_tool_call(tc, cfg) == "deny"

    def test_sensitive_path_denied(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        tc = make_tool_call("read_file", {"path": ".ssh/id_rsa"})
        assert evaluate_tool_call(tc, cfg) == "deny"

    def test_tool_without_path_arg_allowed(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        tc = make_tool_call("list_dir", {})
        assert evaluate_tool_call(tc, cfg) == "allow"


# ─────────────────────────────────────────────────────────────────────────────
# T6 – list_dir
# ─────────────────────────────────────────────────────────────────────────────

class TestListDir:
    def test_basic_listing(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        (tmp_path / "sub").mkdir()
        cfg = make_config(tmp_path)
        result = list_dir(".", cfg)
        assert result["ok"] is True
        names = {e["name"] for e in result["entries"]}
        assert names == {"a.py", "b.txt", "sub"}

    def test_types_reported_correctly(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hi")
        (tmp_path / "subdir").mkdir()
        cfg = make_config(tmp_path)
        result = list_dir(".", cfg)
        by_name = {e["name"]: e for e in result["entries"]}
        assert by_name["file.txt"]["type"] == "file"
        assert by_name["subdir"]["type"] == "directory"
        assert by_name["subdir"]["size"] is None

    def test_truncation(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"f{i}.txt").write_text("")
        cfg = make_config(tmp_path)
        result = list_dir(".", cfg, max_entries=5)
        assert result["ok"] is True
        assert result["truncated"] is True
        assert len(result["entries"]) == 5

    def test_not_truncated_when_within_limit(self, tmp_path: Path) -> None:
        (tmp_path / "only.txt").write_text("")
        cfg = make_config(tmp_path)
        result = list_dir(".", cfg)
        assert result["truncated"] is False

    def test_nonexistent_path_returns_error(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        result = list_dir("nonexistent", cfg)
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_file_as_directory_returns_error(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hi")
        cfg = make_config(tmp_path)
        result = list_dir("file.txt", cfg)
        assert result["ok"] is False
        assert "not a directory" in result["error"]

    def test_recursive_lists_subdirectory_contents(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("")
        cfg = make_config(tmp_path)
        result = list_dir(".", cfg, recursive=True)
        paths = {e["path"] for e in result["entries"]}
        assert "sub/nested.py" in paths

    def test_recursive_skips_hidden_dirs(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "config").write_text("")
        cfg = make_config(tmp_path)
        result = list_dir(".", cfg, recursive=True)
        paths = {e["path"] for e in result["entries"]}
        assert ".git/config" not in paths

    def test_path_traversal_raises_policy_violation(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        with pytest.raises(PolicyViolation):
            list_dir("../outside", cfg)

    def test_relative_paths_in_output(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("")
        cfg = make_config(tmp_path)
        result = list_dir("src", cfg)
        assert result["entries"][0]["path"] == "src/main.py"


# ─────────────────────────────────────────────────────────────────────────────
# T7 – read_file
# ─────────────────────────────────────────────────────────────────────────────

class TestReadFile:
    def test_reads_full_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\n")
        cfg = make_config(tmp_path)
        result = read_file("hello.txt", cfg)
        assert result["ok"] is True
        assert "line1" in result["content"]
        assert result["total_lines"] == 3

    def test_line_range_slice(self, tmp_path: Path) -> None:
        f = tmp_path / "multi.txt"
        f.write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n")
        cfg = make_config(tmp_path)
        result = read_file("multi.txt", cfg, start_line=3, end_line=5)
        assert result["ok"] is True
        assert "L3" in result["content"]
        assert "L5" in result["content"]
        assert "L6" not in result["content"]
        assert result["start_line"] == 3
        assert result["end_line"] == 5

    def test_end_line_clamped_to_eof(self, tmp_path: Path) -> None:
        f = tmp_path / "short.txt"
        f.write_text("a\nb\n")
        cfg = make_config(tmp_path)
        result = read_file("short.txt", cfg, start_line=1, end_line=9999)
        assert result["ok"] is True
        assert result["total_lines"] == 2

    def test_byte_limit_truncates(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_text("x" * 200)
        cfg = make_config(tmp_path, max_read_bytes=50)
        result = read_file("big.txt", cfg)
        assert result["ok"] is True
        assert result["truncated"] is True
        assert len(result["content"].encode()) <= 50
        assert result["total_lines"] == 1

    def test_nonexistent_file_returns_error(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        result = read_file("ghost.txt", cfg)
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_directory_as_file_returns_error(self, tmp_path: Path) -> None:
        (tmp_path / "adir").mkdir()
        cfg = make_config(tmp_path)
        result = read_file("adir", cfg)
        assert result["ok"] is False
        assert "directory" in result["error"]

    def test_non_utf8_file_does_not_crash(self, tmp_path: Path) -> None:
        f = tmp_path / "binary.txt"
        f.write_bytes(b"hello\xff\xfeworld")
        cfg = make_config(tmp_path)
        result = read_file("binary.txt", cfg)
        assert result["ok"] is True
        assert isinstance(result["content"], str)

    def test_path_traversal_raises_policy_violation(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        with pytest.raises(PolicyViolation):
            read_file("../../etc/passwd", cfg)


# ─────────────────────────────────────────────────────────────────────────────
# T8 – search_text
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchText:
    def _make_workspace(self, tmp_path: Path) -> AgentConfig:
        (tmp_path / "main.py").write_text("def main():\n    pass\n")
        (tmp_path / "utils.py").write_text("def helper():\n    return 42\n")
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_text("class MyClass:\n    def main(self):\n        pass\n")
        return make_config(tmp_path)

    def test_finds_existing_keyword(self, tmp_path: Path) -> None:
        cfg = self._make_workspace(tmp_path)
        result = search_text("def main", cfg)
        assert result["ok"] is True
        assert result["total_matches"] >= 1
        files = {m["file"] for m in result["matches"]}
        assert any("main.py" in f for f in files)

    def test_no_matches_returns_empty(self, tmp_path: Path) -> None:
        cfg = self._make_workspace(tmp_path)
        result = search_text("XYZZY_NONEXISTENT_TOKEN", cfg)
        assert result["ok"] is True
        assert result["matches"] == []
        assert result["total_matches"] == 0

    def test_result_truncation(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f"f{i}.py").write_text("needle\n")
        cfg = make_config(tmp_path)
        result = search_text("needle", cfg, max_results=5)
        assert result["ok"] is True
        assert len(result["matches"]) == 5
        assert result["truncated"] is True

    def test_subpath_search(self, tmp_path: Path) -> None:
        cfg = self._make_workspace(tmp_path)
        result = search_text("def main", cfg, path="pkg")
        assert result["ok"] is True
        # Should only find the match in pkg/mod.py
        for m in result["matches"]:
            assert m["file"].startswith("pkg/")

    def test_glob_filter(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("needle")
        (tmp_path / "b.txt").write_text("needle")
        cfg = make_config(tmp_path)
        result = search_text("needle", cfg, glob="**/*.py")
        assert result["ok"] is True
        for m in result["matches"]:
            assert m["file"].endswith(".py")

    def test_context_lines_present(self, tmp_path: Path) -> None:
        (tmp_path / "ctx.py").write_text("before\ntarget_line\nafter\n")
        cfg = make_config(tmp_path)
        result = search_text("target_line", cfg, context_lines=1)
        assert result["ok"] is True
        assert result["total_matches"] == 1
        m = result["matches"][0]
        assert len(m["context_before"]) >= 1 or len(m["context_after"]) >= 1

    def test_nonexistent_search_path_returns_error(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        result = search_text("x", cfg, path="no_such_dir")
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_path_traversal_raises_policy_violation(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        with pytest.raises(PolicyViolation):
            search_text("x", cfg, path="../../etc")

    def test_long_line_truncated(self, tmp_path: Path) -> None:
        (tmp_path / "long.py").write_text("needle " + "x" * 300 + "\n")
        cfg = make_config(tmp_path)
        result = search_text("needle", cfg)
        assert result["ok"] is True
        if result["total_matches"]:
            assert len(result["matches"][0]["line"]) <= 205  # 200 + " …"
