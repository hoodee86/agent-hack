"""Integration tests for the LangGraph state machine (T9-T14)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from linux_agent.audit import EVENT_MODEL_INPUT
from linux_agent.config import AgentConfig
from linux_agent.graph import build_graph
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
    """Minimal stub that cycles through a list of tool-calling turns."""

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


class _MultiToolStubLLM(_StubLLM):
    """Stub that emits multiple tool calls in a single assistant turn."""

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
                        "id": f"call_{idx}_0",
                        "type": "tool_call",
                    },
                    {
                        "name": "list_dir",
                        "args": {"path": "."},
                        "id": f"call_{idx}_1",
                        "type": "tool_call",
                    },
                ],
            )

        return RunnableLambda(_invoke)


def _stub_llm(*turns: _StubTurn) -> _StubLLM:
    return _StubLLM(turns=list(turns))


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphHappyPath:
    def test_immediate_final_answer(self, tmp_path: Path) -> None:
        """Planner immediately returns a final_answer without calling any tool."""
        cfg = _make_config(tmp_path)
        llm = _stub_llm(_final_turn("42"))
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("What is 6*7?", str(tmp_path)))
        assert result["final_answer"] == "42"

    def test_list_dir_then_done(self, tmp_path: Path) -> None:
        """Planner calls list_dir once, then finalises."""
        (tmp_path / "a.py").write_text("pass")
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            _tool_turn("list_dir", {"path": "."}, content="Listing workspace"),
            _final_turn("Found: a.py"),
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
            _tool_turn("read_file", {"path": "note.txt"}, content="Reading note.txt"),
            _final_turn("File says: hello world"),
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
            _tool_turn(
                "search_text",
                {"query": "def main"},
                content="Searching for def main",
            ),
            _final_turn("Found def main in src.py"),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Where is main defined?", str(tmp_path)))
        assert result["final_answer"] == "Found def main in src.py"
        assert result["iteration_count"] == 1

    def test_tool_messages_follow_tool_calls_before_next_human_turn(self, tmp_path: Path) -> None:
        """Planner history must preserve assistant/tool adjacency across turns."""
        (tmp_path / "src.py").write_text("def main():\n    pass\n")
        cfg = _make_config(tmp_path)
        llm = _StubLLM(
            turns=[
                _tool_turn(
                    "search_text",
                    {"query": "def main"},
                    content="Searching for def main",
                ),
                _final_turn("Found def main in src.py"),
            ],
            captures=[],
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Where is main defined?", str(tmp_path)))

        assert result["final_answer"] == "Found def main in src.py"
        assert len(llm.captures) == 2
        second_turn_roles = [message.type for message in llm.captures[1]]
        assert second_turn_roles[-3:] == ["ai", "tool", "human"]

    def test_multiple_provider_tool_calls_are_collapsed_to_one_history_entry(self, tmp_path: Path) -> None:
        """Stored assistant history must not retain extra tool_call ids we never execute."""
        (tmp_path / "src.py").write_text("def main():\n    pass\n")
        cfg = _make_config(tmp_path)
        llm = _MultiToolStubLLM(
            turns=[
                _tool_turn(
                    "search_text",
                    {"query": "def main"},
                    content="Searching for def main",
                ),
                _final_turn("Found def main in src.py"),
            ],
            captures=[],
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Where is main defined?", str(tmp_path)))

        assert result["final_answer"] == "Found def main in src.py"
        assert len(llm.captures) == 2
        prior_ai = llm.captures[1][-3]
        assert isinstance(prior_ai, AIMessage)
        assert len(prior_ai.tool_calls) == 1
        assert prior_ai.tool_calls[0]["id"] == "call_0_0"

    def test_prompt_trace_listener_receives_per_turn_model_inputs(self, tmp_path: Path) -> None:
        """Prompt tracing should expose the exact message sequence sent to the LLM."""
        (tmp_path / "src.py").write_text("def main():\n    pass\n")
        cfg = _make_config(tmp_path)
        llm = _StubLLM(
            turns=[
                _tool_turn(
                    "search_text",
                    {"query": "def main"},
                    content="Searching for def main",
                ),
                _final_turn("Found def main in src.py"),
            ],
            captures=[],
        )
        traces: list[dict[str, Any]] = []
        app = build_graph(cfg, chat_model=llm, prompt_trace_listener=traces.append)

        result = app.invoke(_initial_state("Where is main defined?", str(tmp_path)))

        assert result["final_answer"] == "Found def main in src.py"
        assert len(traces) == 2
        assert traces[0]["event"] == EVENT_MODEL_INPUT
        assert [message["type"] for message in traces[0]["data"]["messages"]] == [
            "system",
            "human",
        ]
        second_roles = [message["type"] for message in traces[1]["data"]["messages"]]
        assert second_roles[-3:] == ["ai", "tool", "human"]


class TestGraphSecurityBlocking:
    def test_path_traversal_denied(self, tmp_path: Path) -> None:
        """PolicyGuard denies a path-traversal tool call."""
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            _tool_turn(
                "read_file",
                {"path": "../../../etc/passwd"},
                content="Reading /etc/passwd",
            )
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
            _tool_turn(
                "write_file",
                {"path": "evil.txt", "content": "boom"},
                content="Writing file",
            )
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
        always_list = _tool_turn("list_dir", {"path": "."}, content="Listing")
        llm = _stub_llm(always_list)
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Loop forever", str(tmp_path)))
        assert result["iteration_count"] >= cfg.max_iterations
        assert result["final_answer"] is not None
        assert "maximum iteration" in (result["final_answer"] or "")

    def test_consecutive_failures_stops_agent(self, tmp_path: Path) -> None:
        """Reflector stops when consecutive_failures reaches the limit."""
        cfg = _make_config(tmp_path, max_consecutive_failures=2)
        always_bad = _tool_turn(
            "read_file",
            {"path": "ghost.txt"},
            content="Reading ghost.txt",
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

        llm = _stub_llm(_final_turn("OK"))
        app = build_graph(cfg, chat_model=llm)
        state = _initial_state("test", str(tmp_path))
        state = AgentState(**{**state, "run_id": run_id})  # type: ignore[misc]
        app.invoke(state)

        log_file = log_dir / f"{run_id}.jsonl"
        assert log_file.exists(), "JSONL log file not created"
        events = [_json.loads(ln) for ln in log_file.read_text().splitlines() if ln]
        event_types = [e["event"] for e in events]
        assert "run_end" in event_types
