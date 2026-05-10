"""Tests for phase-4 plan lifecycle and budget execution (T37/T40)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
import uuid
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from linux_agent.app import _make_verbose_event_printer
from linux_agent.audit import EVENT_PLAN_UPDATE, EVENT_REFLECTOR_ACTION, EVENT_RUN_END
from linux_agent.config import AgentConfig
from linux_agent.graph import build_graph
from linux_agent.state import AgentState


def _make_config(tmp_path: Path, **kwargs: Any) -> AgentConfig:
    return AgentConfig(workspace_root=tmp_path, log_dir=tmp_path / "logs", **kwargs)  # type: ignore[arg-type]


def _initial_state(goal: str, workspace_root: str, **updates: Any) -> AgentState:
    state = AgentState(
        run_id=str(uuid.uuid4()),
        user_goal=goal,
        workspace_root=workspace_root,
        messages=[],
        plan=[],
        current_step=None,
        proposed_tool_call=None,
        observations=[],
        risk_decision=None,
        pending_approval=None,
        resume_action=None,
        pending_verification=None,
        last_write=None,
        last_verification=None,
        last_rollback=None,
        iteration_count=0,
        consecutive_failures=0,
        final_answer=None,
    )
    return AgentState(**{**state, **updates})  # type: ignore[misc]


class _StubTurn(TypedDict):
    content: str
    tool_name: str | None
    tool_args: dict[str, object]


def _tool_turn(
    tool_name: str,
    tool_args: dict[str, object],
    *,
    content: str = "",
) -> _StubTurn:
    return {
        "content": content,
        "tool_name": tool_name,
        "tool_args": tool_args,
    }


def _final_turn(content: str) -> _StubTurn:
    return {"content": content, "tool_name": None, "tool_args": {}}


class _StubLLM(BaseChatModel):
    turns: list[_StubTurn]
    captures: list[list[BaseMessage]] = []
    _call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=MagicMock())])

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        turns = self.turns

        def _invoke(messages: Any, **kw: Any) -> AIMessage:
            self.captures.append(list(messages))
            idx = min(self._call_count, len(turns) - 1)
            self._call_count += 1  # type: ignore[misc]
            turn = turns[idx]
            if turn["tool_name"] is None:
                return AIMessage(content=turn["content"])
            return AIMessage(
                content=turn["content"],
                tool_calls=[
                    {
                        "name": turn["tool_name"],
                        "args": turn["tool_args"],
                        "id": f"call_{idx}",
                        "type": "tool_call",
                    }
                ],
            )

        return RunnableLambda(_invoke)


def _stub_llm(*turns: _StubTurn) -> _StubLLM:
    return _StubLLM(turns=list(turns), captures=[])


class TestPlanLifecycle:
    def test_failed_command_creates_structured_plan_revision(self, tmp_path: Path) -> None:
        (tmp_path / "test_failure.py").write_text(
            "def test_failure() -> None:\n    assert 1 == 2\n",
            encoding="utf-8",
        )
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            _tool_turn(
                "run_command",
                {"command": "pytest -q test_failure.py", "cwd": "."},
                content="Run the failing test first",
            ),
            _tool_turn(
                "read_file",
                {"path": "test_failure.py"},
                content="Read the failing test file before summarizing",
            ),
            _final_turn("The assertion in test_failure.py expects 1 == 2."),
        )
        events: list[dict[str, Any]] = []
        app = build_graph(cfg, chat_model=llm, event_listener=events.append)

        result = app.invoke(_initial_state("Diagnose the failing test", str(tmp_path)))

        assert result["plan_version"] == 2
        assert result["plan_revision_count"] == 1
        assert [step["title"] for step in result["plan_steps"]] == [
            "Run the failing test first",
            "Read the failing test file before summarizing",
        ]
        assert result["plan_steps"][0]["status"] == "blocked"
        assert result["plan_steps"][1]["status"] == "completed"
        assert any(
            event["event"] == "plan_update"
            and event["data"].get("plan_version") == 2
            and event["data"].get("plan_revision_reason") == "appended_step_after_failed_observation"
            for event in events
        )


class TestBudgetExecution:
    def test_command_budget_stops_second_command_before_execution(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, max_command_count=1)
        llm = _stub_llm(
            _tool_turn(
                "run_command",
                {"command": "pwd", "cwd": "."},
                content="Run pwd first",
            ),
            _tool_turn(
                "run_command",
                {"command": "ls", "cwd": "."},
                content="Run ls next",
            ),
        )
        events: list[dict[str, Any]] = []
        app = build_graph(cfg, chat_model=llm, event_listener=events.append)

        result = app.invoke(_initial_state("Inspect the workspace", str(tmp_path)))

        assert len(result["observations"]) == 1
        assert result["observations"][0]["tool"] == "run_command"
        assert result["command_count"] == 1
        assert result["budget_stop_reason"] == "max_command_count"
        assert "command budget exhausted" in (result["final_answer"] or "").lower()
        assert any(
            event["event"] == "reflector_action"
            and event["data"].get("budget_stop_reason") == "max_command_count"
            for event in events
        )

    def test_runtime_budget_stops_before_new_planner_turn(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, max_runtime_seconds=1)
        llm = _stub_llm(
            _tool_turn(
                "list_dir",
                {"path": "."},
                content="List the workspace",
            )
        )
        started_at = (datetime.now(tz=timezone.utc) - timedelta(seconds=5)).isoformat()
        app = build_graph(cfg, chat_model=llm)

        result = app.invoke(
            _initial_state(
                "List files",
                str(tmp_path),
                started_at=started_at,
            )
        )

        assert result["observations"] == []
        assert result["budget_stop_reason"] == "max_runtime_seconds"
        assert "runtime budget exhausted" in (result["final_answer"] or "").lower()


class TestVerboseBudgetRendering:
    def test_verbose_printer_renders_plan_versions_and_budget_fields(self) -> None:
        stream = StringIO()
        emit = _make_verbose_event_printer(stream)
        emit(
            {
                "run_id": "run-1",
                "ts": "2026-05-10T00:00:00+00:00",
                "event": EVENT_PLAN_UPDATE,
                "data": {
                    "plan": ["Run pytest", "Read failing file"],
                    "current_step": "Read failing file",
                    "plan_steps": [
                        {"id": "step_1", "title": "Run pytest", "status": "blocked", "rationale": None, "evidence_refs": [1]},
                        {"id": "step_2", "title": "Read failing file", "status": "in_progress", "rationale": "Follow-up after run_command failed.", "evidence_refs": [1]},
                    ],
                    "plan_version": 2,
                    "plan_revision_count": 1,
                    "plan_revision_reason": "appended_step_after_failed_observation",
                    "assistant_content": "Read the failing file before trying again.",
                    "final_answer": None,
                    "budget_status": {
                        "iteration_count": 1,
                        "command_count": 1,
                        "elapsed_seconds": 10,
                        "warning_triggered": True,
                    },
                    "budget_remaining": {
                        "iterations_remaining": 11,
                        "commands_remaining": 0,
                        "runtime_remaining_seconds": 890,
                        "plan_revisions_remaining": 2,
                        "recovery_attempts_remaining": 2,
                    },
                    "last_reflection": None,
                    "recovery_state": None,
                },
            }
        )
        emit(
            {
                "run_id": "run-1",
                "ts": "2026-05-10T00:00:01+00:00",
                "event": EVENT_REFLECTOR_ACTION,
                "data": {
                    "reason": "budget_warning",
                    "dimensions": ["max_command_count"],
                    "budget_status": {
                        "iteration_count": 1,
                        "command_count": 1,
                        "elapsed_seconds": 10,
                        "warning_triggered": True,
                    },
                    "budget_remaining": {
                        "iterations_remaining": 11,
                        "commands_remaining": 0,
                        "runtime_remaining_seconds": 890,
                        "plan_revisions_remaining": 2,
                        "recovery_attempts_remaining": 2,
                    },
                },
            }
        )
        emit(
            {
                "run_id": "run-1",
                "ts": "2026-05-10T00:00:02+00:00",
                "event": EVENT_RUN_END,
                "data": {
                    "status": "budget_exhausted",
                    "iteration_count": 1,
                    "observation_count": 1,
                    "command_count": 1,
                    "command_summaries": ["1. pwd [cwd=.] -> ok (exit 0)"],
                    "write_count": 0,
                    "write_summaries": [],
                    "plan_version": 2,
                    "plan_revision_count": 1,
                    "plan_steps": [
                        {"id": "step_1", "title": "Run pytest", "status": "blocked", "rationale": None, "evidence_refs": [1]},
                        {"id": "step_2", "title": "Read failing file", "status": "in_progress", "rationale": "Follow-up after run_command failed.", "evidence_refs": [1]},
                    ],
                    "budget_status": {
                        "iteration_count": 1,
                        "command_count": 1,
                        "elapsed_seconds": 10,
                        "warning_triggered": True,
                    },
                    "budget_remaining": {
                        "iterations_remaining": 11,
                        "commands_remaining": 0,
                        "runtime_remaining_seconds": 890,
                        "plan_revisions_remaining": 2,
                        "recovery_attempts_remaining": 2,
                    },
                    "budget_stop_reason": "max_command_count",
                    "verification_status": None,
                    "verification_command": None,
                    "verification_exit_code": None,
                    "rollback_result": None,
                    "final_answer": "Agent stopped: command budget exhausted.",
                },
            }
        )

        rendered = stream.getvalue()

        assert "Plan Version: 2" in rendered
        assert "Plan Revision Count: 1" in rendered
        assert "Budget Status:" in rendered
        assert "Budget Remaining:" in rendered
        assert "Budget Stop Reason: max_command_count" in rendered