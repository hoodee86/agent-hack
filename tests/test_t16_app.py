"""
T16 – Unit tests for app.py (CLI entry point).

Tests exercise argument parsing and config resolution without hitting
the LangGraph runtime (the graph call is monkeypatched).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from linux_agent.audit import (
    EVENT_MODEL_INPUT,
    EVENT_PLAN_UPDATE,
    EVENT_POLICY_DECISION,
    EVENT_TOOL_PROPOSED,
    EVENT_TOOL_RESULT,
)
from linux_agent.app import main
from linux_agent.state import AgentState


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fake_final_state(answer: str, iterations: int = 1) -> AgentState:
    return AgentState(
        run_id="test-run",
        user_goal="test goal",
        workspace_root="/tmp",
        messages=[],
        plan=[],
        current_step=None,
        proposed_tool_call=None,
        observations=[],
        risk_decision=None,
        pending_approval=None,
        resume_action=None,
        iteration_count=iterations,
        consecutive_failures=0,
        final_answer=answer,
    )


def _mock_graph(answer: str = "OK") -> MagicMock:
    """Return a mock compiled graph whose invoke() returns a fixed final state."""
    app = MagicMock()
    app.invoke.return_value = _fake_final_state(answer)
    graph_mock = MagicMock(return_value=app)
    return graph_mock


def _mock_graph_with_events(answer: str = "OK") -> MagicMock:
    """Return a mock compiled graph that emits verbose events before returning."""

    def _factory(*args: Any, **kwargs: Any) -> MagicMock:
        event_listener = kwargs.get("event_listener")
        prompt_trace_listener = kwargs.get("prompt_trace_listener")
        app = MagicMock()

        def _invoke(state: AgentState) -> AgentState:
            if prompt_trace_listener is not None:
                prompt_trace_listener(
                    {
                        "run_id": state["run_id"],
                        "ts": "2026-05-10T00:00:00+00:00",
                        "event": EVENT_MODEL_INPUT,
                        "data": {
                            "message_count": 2,
                            "messages": [
                                {
                                    "type": "system",
                                    "content": "You are a controlled Linux workspace agent.",
                                },
                                {
                                    "type": "human",
                                    "content": "Goal: goal\n\nWorkspace root: /tmp/project",
                                },
                            ],
                        },
                    }
                )
            assert event_listener is not None
            event_listener(
                {
                    "run_id": state["run_id"],
                    "ts": "2026-05-10T00:00:00+00:00",
                    "event": EVENT_PLAN_UPDATE,
                    "data": {
                        "plan": ["Search for main"],
                        "current_step": "Searching for def main",
                        "assistant_content": "I will search for def main.",
                        "final_answer": None,
                    },
                }
            )
            event_listener(
                {
                    "run_id": state["run_id"],
                    "ts": "2026-05-10T00:00:01+00:00",
                    "event": EVENT_TOOL_RESULT,
                    "data": {
                        "tool": "search_text",
                        "tool_call_id": "call_0",
                        "ok": True,
                        "duration_ms": 12,
                        "error": None,
                        "result": {
                            "ok": True,
                            "matches": [{"file": "src/linux_agent/app.py", "line_number": 73}],
                        },
                    },
                }
            )
            return _fake_final_state(answer)

        app.invoke.side_effect = _invoke
        return app

    return MagicMock(side_effect=_factory)


def _mock_config(tmp_path: Path) -> MagicMock:
    return MagicMock(
        workspace_root=tmp_path,
        log_dir=tmp_path / "logs",
        max_iterations=12,
        max_consecutive_failures=3,
        llm_model="deepseek-v4-pro",
        model_dump=MagicMock(
            return_value={
                "workspace_root": tmp_path,
                "log_dir": tmp_path / "logs",
                "max_iterations": 12,
                "max_consecutive_failures": 3,
                "llm_model": "deepseek-v4-pro",
                "llm_temperature": 0.0,
                "max_read_bytes": 65536,
                "max_search_results": 100,
                "max_list_entries": 200,
                "sensitive_path_parts": [],
                "backup_dir": tmp_path / ".linux-agent" / "backups",
                "max_patch_bytes": 65536,
                "max_patch_hunks": 16,
                "auto_rollback_on_verify_failure": False,
            }
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIArguments:
    def test_missing_goal_exits_nonzero(self, tmp_path: Path, capsys: Any) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_goal_only_succeeds(self, tmp_path: Path, capsys: Any) -> None:
        """Goal-only invocation uses --workspace derived from env (patched config)."""
        with (
            patch("linux_agent.app.build_graph", _mock_graph("Done")),
            patch(
                "linux_agent.app.load_config",
                return_value=MagicMock(
                    workspace_root=tmp_path,
                    log_dir=tmp_path / "logs",
                    max_iterations=12,
                    max_consecutive_failures=3,
                        llm_model="deepseek-v4-pro",
                    model_dump=MagicMock(
                        return_value={
                            "workspace_root": tmp_path,
                            "log_dir": tmp_path / "logs",
                            "max_iterations": 12,
                            "max_consecutive_failures": 3,
                                "llm_model": "deepseek-v4-pro",
                            "llm_temperature": 0.0,
                            "max_read_bytes": 65536,
                            "max_search_results": 100,
                            "max_list_entries": 200,
                            "sensitive_path_parts": [],
                        }
                    ),
                ),
            ),
        ):
            code = main(["What files are here?"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Done" in out

    def test_workspace_override(self, tmp_path: Path, capsys: Any) -> None:
        """--workspace flag overrides workspace_root from config."""
        ws = tmp_path / "ws"
        ws.mkdir()
        with (
            patch("linux_agent.app.build_graph", _mock_graph("WorkspaceOK")),
            patch(
                "linux_agent.app.load_config",
                return_value=MagicMock(
                    workspace_root=tmp_path,
                    log_dir=tmp_path / "logs",
                    max_iterations=12,
                    max_consecutive_failures=3,
                        llm_model="deepseek-v4-pro",
                    model_dump=MagicMock(
                        return_value={
                            "workspace_root": tmp_path,
                            "log_dir": tmp_path / "logs",
                            "max_iterations": 12,
                            "max_consecutive_failures": 3,
                                "llm_model": "deepseek-v4-pro",
                            "llm_temperature": 0.0,
                            "max_read_bytes": 65536,
                            "max_search_results": 100,
                            "max_list_entries": 200,
                            "sensitive_path_parts": [],
                        }
                    ),
                ),
            ),
        ):
            code = main(["goal", "--workspace", str(ws)])
        assert code == 0

    def test_nonexistent_workspace_returns_error(self, tmp_path: Path) -> None:
        """--workspace pointing at a nonexistent path should return exit code 1."""
        with patch(
            "linux_agent.app.load_config",
            return_value=MagicMock(
                workspace_root=tmp_path,
                log_dir=tmp_path / "logs",
                max_iterations=12,
                max_consecutive_failures=3,
                    llm_model="deepseek-v4-pro",
                model_dump=MagicMock(return_value={}),
            ),
        ):
            code = main(["goal", "--workspace", "/nonexistent/path/xyz"])
        assert code == 1

    def test_config_error_returns_exit_1(self, tmp_path: Path) -> None:
        """A ValueError from load_config should result in exit code 1."""
        with patch(
            "linux_agent.app.load_config",
            side_effect=ValueError("bad workspace"),
        ):
            code = main(["goal"])
        assert code == 1

    def test_resume_run_requires_decision(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--resume-run", "run-1"])
        assert exc_info.value.code != 0

    def test_verbose_flag_prints_to_stderr(self, tmp_path: Path, capsys: Any) -> None:
        with (
            patch("linux_agent.app.build_graph", _mock_graph("Verbose")),
            patch(
                "linux_agent.app.load_config",
                return_value=MagicMock(
                    workspace_root=tmp_path,
                    log_dir=tmp_path / "logs",
                    max_iterations=12,
                    max_consecutive_failures=3,
                        llm_model="deepseek-v4-pro",
                    model_dump=MagicMock(
                        return_value={
                            "workspace_root": tmp_path,
                            "log_dir": tmp_path / "logs",
                            "max_iterations": 12,
                            "max_consecutive_failures": 3,
                                "llm_model": "deepseek-v4-pro",
                            "llm_temperature": 0.0,
                            "max_read_bytes": 65536,
                            "max_search_results": 100,
                            "max_list_entries": 200,
                            "sensitive_path_parts": [],
                        }
                    ),
                ),
            ),
        ):
            code = main(["goal", "--verbose"])
        assert code == 0
        err = capsys.readouterr().err
        assert "Run ID:" in err

    def test_verbose_prints_detailed_iteration_events(self, tmp_path: Path, capsys: Any) -> None:
        with (
            patch("linux_agent.app.build_graph", _mock_graph_with_events("Verbose")),
            patch(
                "linux_agent.app.load_config",
                return_value=MagicMock(
                    workspace_root=tmp_path,
                    log_dir=tmp_path / "logs",
                    max_iterations=12,
                    max_consecutive_failures=3,
                    llm_model="deepseek-v4-pro",
                    model_dump=MagicMock(
                        return_value={
                            "workspace_root": tmp_path,
                            "log_dir": tmp_path / "logs",
                            "max_iterations": 12,
                            "max_consecutive_failures": 3,
                            "llm_model": "deepseek-v4-pro",
                            "llm_temperature": 0.0,
                            "max_read_bytes": 65536,
                            "max_search_results": 100,
                            "max_list_entries": 200,
                            "sensitive_path_parts": [],
                        }
                    ),
                ),
            ),
        ):
            code = main(["goal", "--verbose"])

        assert code == 0
        err = capsys.readouterr().err
        assert "[linux-agent] Run Start" in err
        assert "[linux-agent] Iteration 1 | Planner" in err
        assert "Searching for def main" in err
        assert "[linux-agent] Iteration 1 | Tool Result" in err
        assert "Tool: search_text" in err

    def test_show_prompts_prints_model_input_sequence(self, tmp_path: Path, capsys: Any) -> None:
        with (
            patch("linux_agent.app.build_graph", _mock_graph_with_events("Verbose")),
            patch(
                "linux_agent.app.load_config",
                return_value=MagicMock(
                    workspace_root=tmp_path,
                    log_dir=tmp_path / "logs",
                    max_iterations=12,
                    max_consecutive_failures=3,
                    llm_model="deepseek-v4-pro",
                    model_dump=MagicMock(
                        return_value={
                            "workspace_root": tmp_path,
                            "log_dir": tmp_path / "logs",
                            "max_iterations": 12,
                            "max_consecutive_failures": 3,
                            "llm_model": "deepseek-v4-pro",
                            "llm_temperature": 0.0,
                            "max_read_bytes": 65536,
                            "max_search_results": 100,
                            "max_list_entries": 200,
                            "sensitive_path_parts": [],
                        }
                    ),
                ),
            ),
        ):
            code = main(["goal", "--show-prompts"])

        assert code == 0
        err = capsys.readouterr().err
        assert "[linux-agent] Iteration 1 | Model Input" in err
        assert "Message Count: 2" in err
        assert "[1] system" in err
        assert "[2] human" in err
        assert "You are a controlled Linux workspace agent." in err

    def test_verbose_prints_command_execution_details(self, tmp_path: Path, capsys: Any) -> None:
        def _mock_graph_with_command_events(*args: Any, **kwargs: Any) -> MagicMock:
            event_listener = kwargs.get("event_listener")
            app = MagicMock()

            def _invoke(state: AgentState) -> AgentState:
                assert event_listener is not None
                event_listener(
                    {
                        "run_id": state["run_id"],
                        "ts": "2026-05-10T00:00:00+00:00",
                        "event": EVENT_PLAN_UPDATE,
                        "data": {
                            "plan": ["Run pytest"],
                            "current_step": "Running pytest to inspect failures",
                            "assistant_content": "I will run pytest -q.",
                            "final_answer": None,
                        },
                    }
                )
                event_listener(
                    {
                        "run_id": state["run_id"],
                        "ts": "2026-05-10T00:00:01+00:00",
                        "event": EVENT_TOOL_PROPOSED,
                        "data": {
                            "tool": "run_command",
                            "tool_call_id": "call_1",
                            "risk_level": "low",
                            "args": {"command": "uv run pytest -q", "cwd": "."},
                            "command": "uv run pytest -q",
                            "argv": ["uv", "run", "pytest", "-q"],
                            "cwd": ".",
                        },
                    }
                )
                event_listener(
                    {
                        "run_id": state["run_id"],
                        "ts": "2026-05-10T00:00:02+00:00",
                        "event": EVENT_POLICY_DECISION,
                        "data": {
                            "tool": "run_command",
                            "tool_call_id": "call_1",
                            "risk_level": "low",
                            "args": {"command": "uv run pytest -q", "cwd": "."},
                            "command": "uv run pytest -q",
                            "argv": ["uv", "run", "pytest", "-q"],
                            "cwd": ".",
                            "decision": "allow",
                        },
                    }
                )
                event_listener(
                    {
                        "run_id": state["run_id"],
                        "ts": "2026-05-10T00:00:03+00:00",
                        "event": EVENT_TOOL_RESULT,
                        "data": {
                            "tool": "run_command",
                            "tool_call_id": "call_1",
                            "risk_level": "low",
                            "ok": False,
                            "duration_ms": 321,
                            "error": "command exited with status 1",
                            "command": "uv run pytest -q",
                            "argv": ["uv", "run", "pytest", "-q"],
                            "cwd": ".",
                            "exit_code": 1,
                            "timed_out": False,
                            "truncated": True,
                            "stdout_preview": "1 failed, 3 passed",
                            "stderr_preview": "tests/test_example.py:12: AssertionError",
                            "result": {
                                "ok": False,
                                "command": "uv run pytest -q",
                                "argv": ["uv", "run", "pytest", "-q"],
                                "cwd": ".",
                                "exit_code": 1,
                                "stdout": "1 failed, 3 passed",
                                "stderr": "tests/test_example.py:12: AssertionError",
                                "duration_ms": 321,
                                "timed_out": False,
                                "truncated": True,
                            },
                        },
                    }
                )
                return _fake_final_state("Command summary")

            app.invoke.side_effect = _invoke
            return app

        with (
            patch(
                "linux_agent.app.build_graph",
                MagicMock(side_effect=_mock_graph_with_command_events),
            ),
            patch(
                "linux_agent.app.load_config",
                return_value=MagicMock(
                    workspace_root=tmp_path,
                    log_dir=tmp_path / "logs",
                    max_iterations=12,
                    max_consecutive_failures=3,
                    llm_model="deepseek-v4-pro",
                    model_dump=MagicMock(
                        return_value={
                            "workspace_root": tmp_path,
                            "log_dir": tmp_path / "logs",
                            "max_iterations": 12,
                            "max_consecutive_failures": 3,
                            "llm_model": "deepseek-v4-pro",
                            "llm_temperature": 0.0,
                            "max_read_bytes": 65536,
                            "max_search_results": 100,
                            "max_list_entries": 200,
                            "sensitive_path_parts": [],
                        }
                    ),
                ),
            ),
        ):
            code = main(["goal", "--verbose"])

        assert code == 0
        err = capsys.readouterr().err
        assert "[linux-agent] Iteration 1 | Tool Proposal" in err
        assert "Command: uv run pytest -q" in err
        assert "Working Directory: ." in err
        assert "[linux-agent] Iteration 1 | Policy Guard" in err
        assert "Decision: allow" in err
        assert "[linux-agent] Iteration 1 | Tool Result" in err
        assert "Exit Code: 1" in err
        assert "Truncated: True" in err
        assert "Stdout:" in err
        assert "Stderr:" in err

    def test_output_printed_to_stdout(self, tmp_path: Path, capsys: Any) -> None:
        with (
            patch("linux_agent.app.build_graph", _mock_graph("The answer is 42")),
            patch(
                "linux_agent.app.load_config",
                return_value=MagicMock(
                    workspace_root=tmp_path,
                    log_dir=tmp_path / "logs",
                    max_iterations=12,
                    max_consecutive_failures=3,
                        llm_model="deepseek-v4-pro",
                    model_dump=MagicMock(
                        return_value={
                            "workspace_root": tmp_path,
                            "log_dir": tmp_path / "logs",
                            "max_iterations": 12,
                            "max_consecutive_failures": 3,
                                "llm_model": "deepseek-v4-pro",
                            "llm_temperature": 0.0,
                            "max_read_bytes": 65536,
                            "max_search_results": 100,
                            "max_list_entries": 200,
                            "sensitive_path_parts": [],
                        }
                    ),
                ),
            ),
        ):
            code = main(["What is 6*7?"])
        assert code == 0
        assert "The answer is 42" in capsys.readouterr().out

    def test_approval_pause_returns_exit_code_2(self, tmp_path: Path, capsys: Any) -> None:
        paused_state = AgentState(
            **{
                **_fake_final_state("Approval required before executing tool 'write_file'."),
                "run_id": "paused-run",
                "risk_decision": "needs_approval",
                "pending_approval": {
                    "id": "approval-1",
                    "tool": "write_file",
                    "args": {"path": "notes.txt", "content": "hello"},
                    "reason": "Write operations require explicit approval before execution.",
                    "impact_summary": "This request would create workspace file 'notes.txt'.",
                    "diff_preview": "hello",
                    "backup_plan": "Backups will be written before overwrite.",
                },
            }
        )
        app = MagicMock()
        app.invoke.return_value = paused_state

        with (
            patch("linux_agent.app.build_graph", MagicMock(return_value=app)),
            patch("linux_agent.app.load_config", return_value=_mock_config(tmp_path)),
        ):
            code = main(["goal"])

        assert code == 2
        assert "Approval required" in capsys.readouterr().out

    def test_resume_run_approve_loads_saved_state(self, tmp_path: Path, capsys: Any) -> None:
        saved_state = AgentState(
            run_id="resume-1",
            user_goal="Create notes.txt",
            workspace_root=str(tmp_path),
            messages=[],
            plan=["Create notes.txt"],
            current_step="Create notes.txt",
            proposed_tool_call={
                "id": "call_1",
                "name": "write_file",
                "args": {"path": "notes.txt", "content": "hello\n", "mode": "create_only"},
                "risk_level": "high",
            },
            observations=[],
            risk_decision="needs_approval",
            pending_approval={
                "id": "approval-1",
                "tool": "write_file",
                "args": {"path": "notes.txt", "content": "hello\n", "mode": "create_only"},
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

        captured_state: dict[str, Any] = {}

        def _build_graph(*args: Any, **kwargs: Any) -> MagicMock:
            app = MagicMock()

            def _invoke(state: AgentState) -> AgentState:
                captured_state.update(state)
                return _fake_final_state("Write completed.")

            app.invoke.side_effect = _invoke
            return app

        with (
            patch("linux_agent.app.build_graph", MagicMock(side_effect=_build_graph)),
            patch("linux_agent.app.load_config", return_value=_mock_config(tmp_path)),
            patch("linux_agent.app.load_run_state", return_value=saved_state),
        ):
            code = main(["--resume-run", "resume-1", "--approve"])

        assert code == 0
        assert captured_state["resume_action"] == "approve"
        assert captured_state["user_goal"] == "Create notes.txt"
        assert "Write completed." in capsys.readouterr().out

    def test_rollback_run_calls_helper(self, tmp_path: Path, capsys: Any) -> None:
        with (
            patch("linux_agent.app.load_config", return_value=_mock_config(tmp_path)),
            patch(
                "linux_agent.app.rollback_run",
                return_value={
                    "ok": True,
                    "run_id": "run-1",
                    "manifest_path": str(tmp_path / "logs" / "manifest.json"),
                    "backup_root": str(tmp_path / ".linux-agent" / "backups" / "run-1"),
                    "restored_files": ["notes.txt"],
                    "removed_files": ["docs/new.txt"],
                    "error": None,
                },
            ),
        ):
            code = main(["--rollback-run", "run-1"])

        assert code == 0
        out = capsys.readouterr().out
        assert "Rolled back run run-1." in out
        assert "Restored: notes.txt" in out
        assert "Removed: docs/new.txt" in out

    def test_audit_run_start_log_written(self, tmp_path: Path) -> None:
        """main() must write a run_start event to the JSONL audit log."""
        import json

        log_dir = tmp_path / "logs"
        with (
            patch("linux_agent.app.build_graph", _mock_graph("OK")),
            patch(
                "linux_agent.app.load_config",
                return_value=MagicMock(
                    workspace_root=tmp_path,
                    log_dir=log_dir,
                    max_iterations=5,
                    max_consecutive_failures=2,
                        llm_model="deepseek-v4-pro",
                    model_dump=MagicMock(
                        return_value={
                            "workspace_root": tmp_path,
                            "log_dir": log_dir,
                            "max_iterations": 5,
                            "max_consecutive_failures": 2,
                                "llm_model": "deepseek-v4-pro",
                            "llm_temperature": 0.0,
                            "max_read_bytes": 65536,
                            "max_search_results": 100,
                            "max_list_entries": 200,
                            "sensitive_path_parts": [],
                        }
                    ),
                ),
            ),
        ):
            main(["Hello"])

        log_files = list(log_dir.glob("*.jsonl"))
        assert len(log_files) == 1
        events = [
            json.loads(ln)
            for ln in log_files[0].read_text().splitlines()
            if ln.strip()
        ]
        assert any(e["event"] == "run_start" for e in events)
