"""Tests for phase-4 reflection scoring and bounded recovery (T38/T39)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from linux_agent.config import AgentConfig
from linux_agent.graph import build_graph
from linux_agent.state import AgentState


def _make_config(tmp_path: Path, **kwargs: Any) -> AgentConfig:
    log_dir = kwargs.pop("log_dir", tmp_path / "logs")
    return AgentConfig(workspace_root=tmp_path, log_dir=log_dir, **kwargs)  # type: ignore[arg-type]


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


class TestReflectionScoring:
    def test_failed_command_populates_structured_reflection_and_recovery(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            _tool_turn(
                "run_command",
                {"command": "find missing-dir", "cwd": "."},
                content="Run find first",
            ),
            _final_turn("The command failed; more context is needed."),
        )
        events: list[dict[str, Any]] = []
        app = build_graph(cfg, chat_model=llm, event_listener=events.append)

        result = app.invoke(_initial_state("Inspect missing-dir", str(tmp_path)))

        assert result["last_reflection"] is not None
        assert result["last_reflection"]["outcome"] == "retry"
        assert result["last_reflection"]["score"] < 100
        assert result["recovery_state"] is not None
        assert result["recovery_state"]["issue_type"] == "command_failure"
        assert result["recovery_state"]["attempt_count"] == 1
        assert result["recovery_state"]["can_retry"] is True
        assert any(
            event["event"] == "reflector_action"
            and event["data"].get("reason") == "reflection_scored"
            and event["data"].get("outcome") == "retry"
            for event in events
        )
        run_end = next(event for event in reversed(events) if event["event"] == "run_end")
        assert run_end["data"]["last_reflection"]["outcome"] == "retry"
        assert run_end["data"]["recovery_state"]["issue_type"] == "command_failure"

    def test_search_no_results_causes_replan_reflection(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, log_dir=tmp_path.parent / f"{tmp_path.name}-logs")
        llm = _stub_llm(
            _tool_turn(
                "search_text",
                {"query": "needle"},
                content="Search for a symbol that is probably absent",
            ),
            _final_turn("No matches found."),
        )
        app = build_graph(cfg, chat_model=llm)

        result = app.invoke(_initial_state("Find needle", str(tmp_path)))

        assert result["last_reflection"] is not None
        assert result["last_reflection"]["outcome"] == "replan"
        assert result["recovery_state"] is not None
        assert result["recovery_state"]["issue_type"] == "search_no_results"
        assert "alternate symbol" in (result["last_reflection"]["recommended_next_action"] or "")


class TestBoundedRecovery:
    def test_repeated_same_failure_exhausts_recovery_budget(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, max_recovery_attempts_per_issue=1)
        llm = _stub_llm(
            _tool_turn(
                "run_command",
                {"command": "find missing-dir", "cwd": "."},
                content="Run find first",
            ),
            _tool_turn(
                "run_command",
                {"command": "find missing-dir", "cwd": "."},
                content="Try the same failing command again",
            ),
        )
        events: list[dict[str, Any]] = []
        app = build_graph(cfg, chat_model=llm, event_listener=events.append)

        result = app.invoke(_initial_state("Inspect missing-dir", str(tmp_path)))

        assert len(result["observations"]) == 2
        assert result["budget_stop_reason"] == "max_recovery_attempts"
        assert result["last_reflection"] is not None
        assert result["last_reflection"]["outcome"] == "stop"
        assert result["recovery_state"] is not None
        assert result["recovery_state"]["attempt_count"] == 2
        assert result["recovery_state"]["can_retry"] is False
        assert "recovery budget exhausted" in (result["final_answer"] or "").lower()
        assert any(
            event["event"] == "reflector_action"
            and event["data"].get("reason") == "recovery_exhausted"
            for event in events
        )

    def test_successful_recovery_clears_recovery_state_and_continues(self, tmp_path: Path) -> None:
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

        assert [obs["tool"] for obs in result["observations"]] == ["run_command", "read_file"]
        assert result["recovery_state"] is None
        assert result["last_reflection"] is not None
        assert result["last_reflection"]["outcome"] == "continue"
        assert any(
            event["event"] == "reflector_action"
            and event["data"].get("reason") == "recovery_cleared"
            for event in events
        )