"""Tests for phase-4 foundations: config, state, and persistence (T36)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from linux_agent.config import load_config
from linux_agent.run_store import load_run_state, save_run_state
from linux_agent.state import AgentState


class TestT36Config:
    def test_phase4_defaults_are_loaded(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LINUX_AGENT_WORKSPACE", str(tmp_path))

        config = load_config()

        assert config.max_command_count == 8
        assert config.max_runtime_seconds == 900
        assert config.max_plan_revisions == 3
        assert config.max_recovery_attempts_per_issue == 2
        assert config.budget_warning_ratio == 0.8
        assert config.reflection_replan_threshold == 60
        assert config.reflection_stop_threshold == 30
        assert config.approval_ui_mode == "compact"

    def test_yaml_can_override_phase4_fields(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent(
                """\
                workspace_root: .
                max_command_count: 5
                max_runtime_seconds: 300
                max_plan_revisions: 2
                max_recovery_attempts_per_issue: 1
                budget_warning_ratio: 0.65
                reflection_replan_threshold: 55
                reflection_stop_threshold: 25
                approval_ui_mode: detailed
                """
            ),
            encoding="utf-8",
        )

        config = load_config(str(config_file))

        assert config.max_command_count == 5
        assert config.max_runtime_seconds == 300
        assert config.max_plan_revisions == 2
        assert config.max_recovery_attempts_per_issue == 1
        assert config.budget_warning_ratio == 0.65
        assert config.reflection_replan_threshold == 55
        assert config.reflection_stop_threshold == 25
        assert config.approval_ui_mode == "detailed"


class TestT36RunStore:
    def test_legacy_state_load_backfills_phase4_defaults(self, tmp_path: Path) -> None:
        config = load_config(str(_write_config(tmp_path)))
        legacy_state = AgentState(
            run_id="run-legacy",
            user_goal="Inspect the workspace",
            workspace_root=str(tmp_path),
            messages=[],
            plan=["Inspect the workspace", "Summarize findings"],
            current_step="Inspect the workspace",
            proposed_tool_call=None,
            observations=[],
            risk_decision=None,
            pending_approval=None,
            resume_action=None,
            pending_verification=None,
            last_write=None,
            last_verification=None,
            last_rollback=None,
            iteration_count=2,
            consecutive_failures=0,
            final_answer=None,
        )

        save_run_state(legacy_state, config)
        loaded = load_run_state("run-legacy", config)

        assert loaded["started_at"] is None
        assert loaded["command_count"] == 0
        assert loaded["plan_version"] == 1
        assert loaded["plan_revision_count"] == 0
        assert loaded["plan_steps"][0]["id"] == "step_1"
        assert loaded["plan_steps"][0]["status"] == "in_progress"
        assert loaded["plan_steps"][1]["status"] == "pending"
        assert loaded["last_reflection"] is None
        assert loaded["recovery_state"] is None
        assert loaded["budget_status"] == {
            "iteration_count": 2,
            "command_count": 0,
            "elapsed_seconds": 0,
            "warning_triggered": False,
        }
        assert loaded["budget_stop_reason"] is None

    def test_phase4_metadata_round_trips_through_run_store(self, tmp_path: Path) -> None:
        config = load_config(str(_write_config(tmp_path)))
        state = AgentState(
            run_id="run-phase4",
            user_goal="Fix the test",
            workspace_root=str(tmp_path),
            started_at="2026-05-10T00:00:00+00:00",
            messages=[],
            plan=["Read failing test", "Patch code"],
            command_count=3,
            plan_version=2,
            plan_revision_count=1,
            plan_steps=[
                {
                    "id": "step_1",
                    "title": "Read failing test",
                    "status": "completed",
                    "rationale": "Initial context gathered.",
                    "evidence_refs": [0],
                },
                {
                    "id": "step_2",
                    "title": "Patch code",
                    "status": "in_progress",
                    "rationale": "Validation output points to a single file.",
                    "evidence_refs": [1, 2],
                },
            ],
            last_reflection={
                "score": 58,
                "outcome": "replan",
                "reason": "The first attempt surfaced new compiler output.",
                "retryable": True,
                "recommended_next_action": "Read the failing file before editing.",
            },
            recovery_state={
                "issue_type": "command_failure",
                "fingerprint": "pytest::tests/test_graph.py::1",
                "attempt_count": 1,
                "last_action": "read_file tests/test_graph.py",
                "can_retry": True,
            },
            budget_status={
                "iteration_count": 4,
                "command_count": 3,
                "elapsed_seconds": 42,
                "warning_triggered": True,
            },
            budget_stop_reason=None,
            current_step="Patch code",
            proposed_tool_call=None,
            observations=[],
            risk_decision=None,
            pending_approval=None,
            resume_action=None,
            pending_verification=None,
            last_write=None,
            last_verification=None,
            last_rollback=None,
            iteration_count=4,
            consecutive_failures=1,
            final_answer=None,
        )

        save_run_state(state, config)
        loaded = load_run_state("run-phase4", config)

        assert loaded["started_at"] == "2026-05-10T00:00:00+00:00"
        assert loaded["command_count"] == 3
        assert loaded["plan_version"] == 2
        assert loaded["plan_revision_count"] == 1
        assert loaded["plan_steps"][1]["title"] == "Patch code"
        assert loaded["last_reflection"] is not None
        assert loaded["last_reflection"]["outcome"] == "replan"
        assert loaded["recovery_state"] is not None
        assert loaded["recovery_state"]["fingerprint"] == "pytest::tests/test_graph.py::1"
        assert loaded["budget_status"]["elapsed_seconds"] == 42
        assert loaded["budget_status"]["warning_triggered"] is True
        assert loaded["budget_stop_reason"] is None


def _write_config(tmp_path: Path) -> Path:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("workspace_root: .\n", encoding="utf-8")
    return config_file