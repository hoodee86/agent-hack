"""
Integration tests for the LangGraph state machine (T9-T14).

All tests use a stub LangChain chat model so no real API key is needed.
The stub is a simple BaseChatModel subclass whose ``with_structured_output``
method returns a pre-configured Runnable that yields a fixed PlannerDecision.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterator, Sequence
from unittest.mock import MagicMock

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from linux_agent.config import AgentConfig
from linux_agent.graph import PlannerDecision, build_graph
from linux_agent.state import AgentState


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(tmp_path: Path, **kwargs: Any) -> AgentConfig:
    return AgentConfig(workspace_root=tmp_path, **kwargs)  # type: ignore[arg-type]


def _initial_state(
    goal: str,
    workspace_root: str,
) -> AgentState:
    return AgentState(
        run_id=str(uuid.uuid4()),
        user_goal=goal,
        workspace_root=workspace_root,
        messages=[],
        plan=[],
        current_step=None,
        proposed_tool_call=None,
        observations=[],
        risk_decision=None,
        iteration_count=0,
        consecutive_failures=0,
        final_answer=None,
    )


class _StubLLM(BaseChatModel):
    """Minimal stub that cycles through a list of PlannerDecision objects."""

    decisions: list[PlannerDecision]
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
        decisions = self.decisions

        def _invoke(messages: Any, **kw: Any) -> AIMessage:
            idx = min(self._call_count, len(decisions) - 1)
            self._call_count += 1  # type: ignore[misc]
            decision = decisions[idx]
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "PlannerDecision",
                        "args": decision.model_dump(),
                        "id": f"call_{idx}",
                        "type": "tool_call",
                    }
                ],
            )

        return RunnableLambda(_invoke)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        decisions = self.decisions

        def _invoke(messages: Any, **kw: Any) -> PlannerDecision:
            idx = min(self._call_count, len(decisions) - 1)
            self._call_count += 1  # type: ignore[misc]
            return decisions[idx]

        return RunnableLambda(_invoke)


def _stub_llm(*decisions: PlannerDecision) -> _StubLLM:
    return _StubLLM(decisions=list(decisions))


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphHappyPath:
    def test_immediate_final_answer(self, tmp_path: Path) -> None:
        """Planner immediately returns a final_answer without calling any tool."""
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            PlannerDecision(
                thought="I already know the answer.",
                plan=["Done"],
                current_step="Answering directly",
                tool_name=None,
                tool_args={},
                final_answer="42",
            )
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("What is 6*7?", str(tmp_path)))
        assert result["final_answer"] == "42"

    def test_list_dir_then_done(self, tmp_path: Path) -> None:
        """Planner calls list_dir once, then finalises."""
        (tmp_path / "a.py").write_text("pass")
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            PlannerDecision(
                thought="Let me list the workspace.",
                plan=["List workspace", "Report"],
                current_step="Listing workspace",
                tool_name="list_dir",
                tool_args={"path": "."},
            ),
            PlannerDecision(
                thought="I have the listing.",
                plan=["List workspace", "Report"],
                current_step="Reporting",
                final_answer="Found: a.py",
            ),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("What files are here?", str(tmp_path)))
        assert result["final_answer"] == "Found: a.py"
        assert result["iteration_count"] == 1
        assert result["observations"][0]["ok"] is True
        assert result["observations"][0]["tool"] == "list_dir"

    def test_read_file_then_done(self, tmp_path: Path) -> None:
        """Planner calls read_file once, then finalises."""
        (tmp_path / "note.txt").write_text("hello world\n")
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            PlannerDecision(
                thought="Read note.txt",
                plan=["Read file"],
                current_step="Reading note.txt",
                tool_name="read_file",
                tool_args={"path": "note.txt"},
            ),
            PlannerDecision(
                thought="Got the content.",
                plan=["Read file"],
                current_step="Reporting",
                final_answer="File says: hello world",
            ),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("What is in note.txt?", str(tmp_path)))
        assert result["final_answer"] == "File says: hello world"
        assert result["observations"][0]["ok"] is True

    def test_search_text_then_done(self, tmp_path: Path) -> None:
        """Planner calls search_text once, then finalises."""
        (tmp_path / "src.py").write_text("def main():\n    pass\n")
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            PlannerDecision(
                thought="Search for main",
                plan=["Search", "Report"],
                current_step="Searching for 'def main'",
                tool_name="search_text",
                tool_args={"query": "def main"},
            ),
            PlannerDecision(
                thought="Found it.",
                plan=["Search", "Report"],
                current_step="Reporting",
                final_answer="Found def main in src.py",
            ),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Where is main defined?", str(tmp_path)))
        assert result["final_answer"] == "Found def main in src.py"
        assert result["iteration_count"] == 1


class TestGraphSecurityBlocking:
    def test_path_traversal_denied(self, tmp_path: Path) -> None:
        """PolicyGuard denies a path-traversal tool call."""
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            PlannerDecision(
                thought="Try to escape",
                plan=["Escape"],
                current_step="Reading /etc/passwd",
                tool_name="read_file",
                tool_args={"path": "../../../etc/passwd"},
            ),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Read /etc/passwd", str(tmp_path)))
        assert result["risk_decision"] == "deny"
        assert "denied" in (result["final_answer"] or "").lower()
        assert result["iteration_count"] == 0  # ToolExecutor never ran

    def test_write_tool_denied(self, tmp_path: Path) -> None:
        """PolicyGuard denies a non-read-only tool."""
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            PlannerDecision(
                thought="Write a file",
                plan=["Write"],
                current_step="Writing file",
                tool_name="write_file",
                tool_args={"path": "evil.txt", "content": "boom"},
            ),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Write a file", str(tmp_path)))
        assert result["risk_decision"] == "deny"
        assert result["iteration_count"] == 0


class TestGraphCircuitBreaker:
    def test_max_iterations_stops_agent(self, tmp_path: Path) -> None:
        """Reflector stops the agent when max_iterations is reached."""
        (tmp_path / "f.txt").write_text("x")
        cfg = _make_config(tmp_path, max_iterations=2)
        # Planner always wants to list_dir; will never set final_answer
        always_list = PlannerDecision(
            thought="Keep listing",
            plan=["List forever"],
            current_step="Listing",
            tool_name="list_dir",
            tool_args={"path": "."},
        )
        llm = _stub_llm(always_list)
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Loop forever", str(tmp_path)))
        assert result["iteration_count"] >= cfg.max_iterations
        assert result["final_answer"] is not None
        assert "maximum iteration" in (result["final_answer"] or "")

    def test_consecutive_failures_stops_agent(self, tmp_path: Path) -> None:
        """Reflector stops when consecutive_failures reaches the limit."""
        cfg = _make_config(tmp_path, max_consecutive_failures=2)
        # Planner always tries to read a nonexistent file
        always_bad = PlannerDecision(
            thought="Read ghost file",
            plan=["Read ghost"],
            current_step="Reading ghost.txt",
            tool_name="read_file",
            tool_args={"path": "ghost.txt"},
        )
        llm = _stub_llm(always_bad)
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Read ghost file", str(tmp_path)))
        assert result["final_answer"] is not None
        assert "consecutive" in (result["final_answer"] or "").lower()


class TestAuditLogging:
    def test_run_end_log_written(self, tmp_path: Path) -> None:
        """A run_end JSONL entry is written after every completed run."""
        import json as _json

        log_dir = tmp_path / "logs"
        cfg = _make_config(tmp_path, log_dir=log_dir)
        run_id = str(uuid.uuid4())

        llm = _stub_llm(
            PlannerDecision(
                thought="Done",
                plan=[],
                current_step="Done",
                final_answer="OK",
            )
        )
        app = build_graph(cfg, chat_model=llm)
        state = _initial_state("test", str(tmp_path))
        state = AgentState(**{**state, "run_id": run_id})  # type: ignore[misc]
        app.invoke(state)

        log_file = log_dir / f"{run_id}.jsonl"
        assert log_file.exists(), "JSONL log file not created"
        events = [_json.loads(ln) for ln in log_file.read_text().splitlines() if ln]
        event_types = [e["event"] for e in events]
        assert "run_end" in event_types
