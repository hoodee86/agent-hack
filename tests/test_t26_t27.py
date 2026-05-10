"""Tests for phase-3 foundations: approval state/config and write policy."""

from __future__ import annotations

import textwrap
from pathlib import Path

from linux_agent.config import AgentConfig, load_config
from linux_agent.policy import assess_tool_call
from linux_agent.state import AgentState, ApprovalRequest, ToolCall


def make_config(tmp_path: Path, **kwargs: object) -> AgentConfig:
    return AgentConfig(workspace_root=tmp_path, **kwargs)  # type: ignore[arg-type]


def make_tool_call(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(
        id=f"{name}_test",
        name=name,
        args=args or {},
        risk_level="high",
    )


class TestT26ConfigAndState:
    def test_write_defaults_are_loaded(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LINUX_AGENT_WORKSPACE", str(tmp_path))

        config = load_config()

        assert config.write_requires_approval is True
        assert config.max_patch_bytes == 32768
        assert config.max_patch_hunks == 24
        assert config.backup_dir == Path(".linux-agent/backups")
        assert config.auto_rollback_on_verify_failure is True

    def test_yaml_can_override_write_policy_fields(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent(
                """\
                workspace_root: .
                write_requires_approval: true
                max_patch_bytes: 8192
                max_patch_hunks: 6
                backup_dir: .agent-backups
                auto_rollback_on_verify_failure: false
                """
            ),
            encoding="utf-8",
        )

        config = load_config(str(config_file))

        assert config.write_requires_approval is True
        assert config.max_patch_bytes == 8192
        assert config.max_patch_hunks == 6
        assert config.backup_dir == (tmp_path / ".agent-backups").resolve()
        assert config.auto_rollback_on_verify_failure is False

    def test_agent_state_can_hold_pending_approval(self) -> None:
        approval = ApprovalRequest(
            id="approval_1",
            tool="write_file",
            args={"path": "note.txt", "content": "hello"},
            reason="Write operations require explicit approval before execution.",
            impact_summary="This request would create workspace file 'note.txt'.",
            diff_preview="hello",
            backup_plan="Backups will be written under .linux-agent/backups/run-1 before any overwrite.",
        )

        state = AgentState(
            run_id="run-1",
            user_goal="Write note.txt",
            workspace_root="/tmp/workspace",
            messages=[],
            plan=[],
            current_step=None,
            proposed_tool_call=None,
            observations=[],
            risk_decision="needs_approval",
            pending_approval=approval,
            resume_action=None,
            pending_verification=None,
            last_write=None,
            last_verification=None,
            last_rollback=None,
            iteration_count=0,
            consecutive_failures=0,
            final_answer="Approval required before executing tool 'write_file'.",
        )

        assert state["pending_approval"] is approval
        assert state["risk_decision"] == "needs_approval"


class TestT27StructuredPolicy:
    def test_write_file_approval_request_describes_overwrite(self, tmp_path: Path) -> None:
        (tmp_path / "note.txt").write_text("old\n", encoding="utf-8")
        config = make_config(tmp_path)
        assessment = assess_tool_call(
            make_tool_call("write_file", {"path": "note.txt", "content": "new\n"}),
            config,
            run_id="run-1",
        )

        request = assessment["approval_request"]
        assert assessment["decision"] == "needs_approval"
        assert request is not None
        assert request["tool"] == "write_file"
        assert "overwrite" in request["impact_summary"]
        assert request["diff_preview"] == "new"

    def test_apply_patch_approval_request_contains_preview(self, tmp_path: Path) -> None:
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "module.py").write_text("x = 1\n", encoding="utf-8")
        config = make_config(tmp_path)
        assessment = assess_tool_call(
            make_tool_call(
                "apply_patch",
                {
                    "patch": (
                        "*** Begin Patch\n"
                        "*** Update File: pkg/module.py\n"
                        "@@\n"
                        "-x = 1\n"
                        "+x = 2\n"
                        "*** End Patch"
                    )
                },
            ),
            config,
            run_id="run-2",
        )

        request = assessment["approval_request"]
        assert assessment["decision"] == "needs_approval"
        assert request is not None
        assert "pkg/module.py" in request["impact_summary"]
        assert "Update File: pkg/module.py" in (request["diff_preview"] or "")