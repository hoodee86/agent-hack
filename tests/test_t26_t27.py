"""Tests for phase-3 foundations: approval state/config and write policy."""

from __future__ import annotations

import textwrap
from pathlib import Path

from linux_agent.approval_ui import build_approval_view, format_approval_view
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

    def test_write_file_approval_request_includes_phase4_fields_and_truncates_preview(
        self,
        tmp_path: Path,
    ) -> None:
        config = make_config(tmp_path)
        hidden_tail = "SECRET_SHOULD_NOT_APPEAR_IN_APPROVAL_PREVIEW"
        content = "prefix-" + ("a" * 360) + hidden_tail

        assessment = assess_tool_call(
            make_tool_call(
                "write_file",
                {
                    "path": "pkg/module.py",
                    "content": content,
                    "mode": "create_only",
                },
            ),
            config,
            run_id="run-9",
        )

        request = assessment["approval_request"]
        assert assessment["decision"] == "needs_approval"
        assert request is not None
        assert request["affected_files"] == ["pkg/module.py"]
        assert request["risk_level"] == "high"
        assert request["rollback_command"] == "--rollback-run run-9"
        assert request["suggested_verification_command"] == "uv run pytest"
        assert request["diff_preview"] is not None
        assert request["diff_preview"].endswith(" …")
        assert hidden_tail not in request["diff_preview"]

    def test_approval_view_reuses_clipped_preview_and_budget_context(
        self,
        tmp_path: Path,
    ) -> None:
        config = make_config(
            tmp_path,
            approval_ui_mode="detailed",
            max_command_count=8,
            max_plan_revisions=3,
            max_recovery_attempts_per_issue=2,
        )
        hidden_tail = "TOP_SECRET_TAIL"
        assessment = assess_tool_call(
            make_tool_call(
                "write_file",
                {
                    "path": "pkg/module.py",
                    "content": "prefix-" + ("b" * 360) + hidden_tail,
                    "mode": "create_only",
                },
            ),
            config,
            run_id="run-10",
        )
        request = assessment["approval_request"]
        assert request is not None

        state = AgentState(
            run_id="run-10",
            user_goal="Create module.py",
            workspace_root=str(tmp_path),
            messages=[],
            plan=["Inspect the package", "Create module.py"],
            plan_version=2,
            plan_revision_count=1,
            plan_steps=[
                {
                    "id": "step_1",
                    "title": "Inspect the package",
                    "status": "completed",
                    "rationale": None,
                    "evidence_refs": [1],
                },
                {
                    "id": "step_2",
                    "title": "Create module.py",
                    "status": "in_progress",
                    "rationale": "Narrow approved write.",
                    "evidence_refs": [2],
                },
            ],
            current_step="Create module.py",
            proposed_tool_call=None,
            observations=[],
            risk_decision="needs_approval",
            pending_approval=request,
            resume_action=None,
            pending_verification=None,
            last_write=None,
            last_verification=None,
            last_rollback=None,
            recovery_state={
                "issue_type": "command_failure",
                "fingerprint": "run_command:pytest",
                "attempt_count": 1,
                "last_action": "Read the failing test first",
                "can_retry": True,
            },
            recovery_attempt_total=1,
            budget_status={
                "iteration_count": 2,
                "command_count": 1,
                "elapsed_seconds": 120,
                "warning_triggered": True,
            },
            budget_stop_reason=None,
            iteration_count=2,
            consecutive_failures=0,
            final_answer=None,
        )

        view = build_approval_view(
            state,
            config,
            state_path=tmp_path / "logs" / "state" / "run-10.json",
        )
        rendered = format_approval_view(view, mode="detailed")

        assert view["budget_remaining"] == {
            "iterations_remaining": 10,
            "commands_remaining": 7,
            "runtime_remaining_seconds": 780,
            "plan_revisions_remaining": 2,
            "recovery_attempts_remaining": 1,
        }
        assert "Plan Steps:" in rendered
        assert "Recovery State:" in rendered
        assert "Review Again: --show-pending-run run-10" in rendered
        assert hidden_tail not in rendered