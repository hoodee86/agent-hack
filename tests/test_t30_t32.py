"""Tests for approval persistence and resume support (T30-T32)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from linux_agent.config import AgentConfig
from linux_agent.run_store import delete_run_state, load_run_state, save_run_state, state_path
from linux_agent.state import AgentState


def make_config(tmp_path: Path, **kwargs: object) -> AgentConfig:
    return AgentConfig(workspace_root=tmp_path, log_dir=tmp_path / "logs", **kwargs)  # type: ignore[arg-type]


class TestRunStore:
    def test_save_and_load_run_state_round_trip(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        state = AgentState(
            run_id="run-1",
            user_goal="Create notes.txt",
            workspace_root=str(tmp_path),
            messages=[
                HumanMessage(content="Create notes.txt"),
                AIMessage(
                    content="I need approval before writing.",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"path": "notes.txt", "content": "hello\n"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
            ],
            plan=["Create notes.txt"],
            current_step="Create notes.txt",
            proposed_tool_call={
                "id": "call_1",
                "name": "write_file",
                "args": {"path": "notes.txt", "content": "hello\n"},
                "risk_level": "high",
            },
            observations=[],
            risk_decision="needs_approval",
            pending_approval={
                "id": "approval-1",
                "tool": "write_file",
                "args": {"path": "notes.txt", "content": "hello\n"},
                "reason": "Write operations require explicit approval before execution.",
                "impact_summary": "This request would create workspace file 'notes.txt'.",
                "diff_preview": "hello",
                "backup_plan": "Backups will be written before overwrite.",
            },
            resume_action=None,
            iteration_count=0,
            consecutive_failures=0,
            final_answer="Approval required before executing tool 'write_file'.",
        )

        path = save_run_state(state, config)
        loaded = load_run_state("run-1", config)

        assert path == state_path("run-1", config)
        assert loaded["run_id"] == "run-1"
        assert loaded["pending_approval"] is not None
        assert loaded["pending_approval"]["id"] == "approval-1"
        assert loaded["resume_action"] is None
        assert len(loaded["messages"]) == 2
        assert isinstance(loaded["messages"][0], HumanMessage)
        assert isinstance(loaded["messages"][1], AIMessage)
        assert loaded["messages"][1].tool_calls[0]["name"] == "write_file"

    def test_delete_run_state_removes_snapshot(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        state = AgentState(
            run_id="run-2",
            user_goal="noop",
            workspace_root=str(tmp_path),
            messages=[],
            plan=[],
            current_step=None,
            proposed_tool_call=None,
            observations=[],
            risk_decision=None,
            pending_approval=None,
            resume_action=None,
            iteration_count=0,
            consecutive_failures=0,
            final_answer=None,
        )

        save_run_state(state, config)
        assert state_path("run-2", config).exists()

        delete_run_state("run-2", config)

        assert not state_path("run-2", config).exists()