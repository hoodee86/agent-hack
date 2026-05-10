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
from linux_agent.run_store import state_path
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

    def test_run_command_then_done(self, tmp_path: Path) -> None:
        """Planner can execute run_command and preserve command metadata."""
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            _tool_turn(
                "run_command",
                {"command": "pwd", "cwd": "."},
                content="Running pwd to confirm the workspace root",
            ),
            _final_turn("Command completed."),
        )
        app = build_graph(cfg, chat_model=llm)

        result = app.invoke(_initial_state("Confirm the working directory", str(tmp_path)))

        assert result["observations"][0]["tool"] == "run_command"
        assert result["observations"][0]["ok"] is True
        command_result = result["observations"][0]["result"]
        assert command_result is not None
        assert command_result["command"] == "pwd"
        assert command_result["cwd"] == "."
        assert command_result["exit_code"] == 0
        assert "Executed commands:" in (result["final_answer"] or "")
        assert "pwd [cwd=.] -> ok" in (result["final_answer"] or "")

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

    def test_failed_command_is_preserved_for_follow_up_and_summary(self, tmp_path: Path) -> None:
        """Failed run_command observations must retain structured stderr and exit data."""
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            _tool_turn(
                "run_command",
                {"command": "find missing-dir", "cwd": "."},
                content="Running find to inspect missing-dir",
            ),
            _final_turn("The command failed; inspect missing-dir before retrying."),
        )
        events: list[dict[str, Any]] = []
        app = build_graph(cfg, chat_model=llm, event_listener=events.append)

        result = app.invoke(_initial_state("Inspect missing-dir", str(tmp_path)))

        command_result = result["observations"][0]["result"]
        assert result["observations"][0]["tool"] == "run_command"
        assert result["observations"][0]["ok"] is False
        assert command_result is not None
        assert command_result["exit_code"] not in (None, 0)
        assert command_result["stderr"] or command_result["stdout"]
        assert "Executed commands:" in (result["final_answer"] or "")
        assert "find missing-dir [cwd=.] -> failed" in (result["final_answer"] or "")
        assert any(
            event["event"] == "reflector_action"
            and event["data"].get("reason") == "command_failed"
            for event in events
        )

    def test_failed_command_history_is_available_to_next_turn_follow_up(self, tmp_path: Path) -> None:
        """Failed run_command results should flow into the next planner turn via ToolMessage."""
        (tmp_path / "test_failure.py").write_text(
            "def test_failure() -> None:\n    assert 1 == 2\n",
            encoding="utf-8",
        )
        cfg = _make_config(tmp_path)
        llm = _StubLLM(
            turns=[
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
                _final_turn("pytest fails in test_failure.py because the assertion expects 1 == 2."),
            ],
            captures=[],
        )
        app = build_graph(cfg, chat_model=llm)

        result = app.invoke(_initial_state("Diagnose the failing pytest case", str(tmp_path)))

        assert [obs["tool"] for obs in result["observations"]] == ["run_command", "read_file"]
        assert result["observations"][0]["ok"] is False
        assert result["observations"][1]["ok"] is True
        assert len(llm.captures) == 3
        tool_payload = llm.captures[1][-2].content
        assert isinstance(tool_payload, str)
        assert '"tool": "run_command"' in tool_payload
        assert '"exit_code": 1' in tool_payload
        assert "test_failure.py" in tool_payload
        assert "Executed commands:" in (result["final_answer"] or "")

    def test_command_timeout_stops_with_explicit_timeout_summary(self, tmp_path: Path) -> None:
        """Reflector should stop immediately when a command times out."""
        test_file = tmp_path / "test_sleep.py"
        test_file.write_text(
            "import time\n\n\ndef test_sleep() -> None:\n    time.sleep(2)\n",
            encoding="utf-8",
        )
        cfg = _make_config(tmp_path)
        llm = _stub_llm(
            _tool_turn(
                "run_command",
                {"command": "pytest -q", "cwd": ".", "timeout_seconds": 1},
                content="Running pytest with a short timeout",
            )
        )
        app = build_graph(cfg, chat_model=llm)

        result = app.invoke(_initial_state("Run tests quickly", str(tmp_path)))

        assert "timed out" in (result["final_answer"] or "").lower()
        assert "Executed commands:" in (result["final_answer"] or "")
        assert "pytest -q [cwd=.] -> timed out" in (result["final_answer"] or "")


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

    def test_write_tool_requires_approval(self, tmp_path: Path) -> None:
        """PolicyGuard pauses write tools for approval instead of executing them."""
        log_dir = tmp_path / "logs"
        cfg = _make_config(tmp_path, log_dir=log_dir)
        run_id = str(uuid.uuid4())
        llm = _stub_llm(
            _tool_turn(
                "write_file",
                {"path": "evil.txt", "content": "boom"},
                content="Writing file",
            )
        )
        events: list[dict[str, Any]] = []
        app = build_graph(cfg, chat_model=llm, event_listener=events.append)
        state = _initial_state("Write a file", str(tmp_path))
        state["run_id"] = run_id
        result = app.invoke(state)

        assert result["risk_decision"] == "needs_approval"
        assert result["pending_approval"] is not None
        assert result["pending_approval"]["tool"] == "write_file"
        assert "approval required" in (result["final_answer"] or "").lower()
        assert "--resume-run" in (result["final_answer"] or "")
        assert result["iteration_count"] == 0
        assert result["observations"] == []
        assert state_path(run_id, cfg).exists()
        assert any(
            event["event"] == "policy_decision"
            and event["data"].get("decision") == "needs_approval"
            and event["data"].get("reason")
            for event in events
        )
        assert any(event["event"] == "approval_requested" for event in events)

    def test_approved_resume_executes_pending_write_tool(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        cfg = _make_config(tmp_path, log_dir=log_dir)
        run_id = str(uuid.uuid4())
        (tmp_path / "test_validation.py").write_text(
            "def test_validation():\n    assert True\n",
            encoding="utf-8",
        )

        pause_llm = _stub_llm(
            _tool_turn(
                "write_file",
                {"path": "notes.txt", "content": "hello\n", "mode": "create_only"},
                content="Create notes.txt",
            )
        )
        app = build_graph(cfg, chat_model=pause_llm)
        paused_state = _initial_state("Create notes.txt", str(tmp_path))
        paused_state["run_id"] = run_id
        paused = app.invoke(paused_state)

        resume_llm = _stub_llm(
            _final_turn("Write completed."),
            _tool_turn(
                "run_command",
                {"command": "python -m pytest -q", "cwd": "."},
                content="Run tests for validation",
            ),
            _final_turn("Validated and done."),
        )
        resumed_app = build_graph(cfg, chat_model=resume_llm)
        resumed_state = AgentState(**{**paused, "resume_action": "approve"})  # type: ignore[misc]

        result = resumed_app.invoke(resumed_state)

        assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello\n"
        assert result["pending_approval"] is None
        assert result["risk_decision"] is None
        assert result["iteration_count"] == 2
        assert result["pending_verification"] is None
        assert result["last_verification"] is not None
        assert result["last_verification"]["ok"] is True
        assert result["last_rollback"] is None
        assert "Applied file changes:" in (result["final_answer"] or "")
        assert "Validation result:" in (result["final_answer"] or "")
        assert not state_path(run_id, cfg).exists()
        assert resume_llm._call_count == 3

        import json as _json

        events = [_json.loads(line) for line in (log_dir / f"{run_id}.jsonl").read_text().splitlines() if line]
        assert any(event["event"] == "approval_requested" for event in events)
        write_applied = next(event for event in events if event["event"] == "write_applied")
        assert write_applied["data"]["approval_request_id"] == paused["pending_approval"]["id"]
        assert write_applied["data"]["changed_files"] == ["notes.txt"]
        run_end = [event for event in events if event["event"] == "run_end"][-1]
        assert run_end["data"]["status"] == "verified_write_completed"
        assert run_end["data"]["verification_status"] == "passed"

    def test_write_final_answer_without_validation_is_blocked(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, log_dir=tmp_path / "logs")
        run_id = str(uuid.uuid4())
        pause_llm = _stub_llm(
            _tool_turn(
                "write_file",
                {"path": "notes.txt", "content": "hello\n", "mode": "create_only"},
                content="Create notes.txt",
            )
        )
        app = build_graph(cfg, chat_model=pause_llm)
        paused_state = _initial_state("Create notes.txt", str(tmp_path))
        paused_state["run_id"] = run_id
        paused = app.invoke(paused_state)

        resumed_app = build_graph(cfg, chat_model=_stub_llm(_final_turn("Write completed.")))
        resumed_state = AgentState(**{**paused, "resume_action": "approve"})  # type: ignore[misc]

        result = resumed_app.invoke(resumed_state)

        assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello\n"
        assert result["pending_verification"] is not None
        assert result["last_verification"] is None
        assert "stopped before validating" in (result["final_answer"] or "").lower()
        assert "Validation still required:" in (result["final_answer"] or "")

    def test_failed_validation_auto_rolls_back_recent_write(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        cfg = _make_config(tmp_path, log_dir=log_dir, auto_rollback_on_verify_failure=True)
        run_id = str(uuid.uuid4())
        (tmp_path / "test_failure.py").write_text(
            "def test_failure():\n    assert False\n",
            encoding="utf-8",
        )

        pause_llm = _stub_llm(
            _tool_turn(
                "write_file",
                {"path": "notes.txt", "content": "hello\n", "mode": "create_only"},
                content="Create notes.txt",
            )
        )
        app = build_graph(cfg, chat_model=pause_llm)
        paused_state = _initial_state("Create notes.txt", str(tmp_path))
        paused_state["run_id"] = run_id
        paused = app.invoke(paused_state)

        resume_llm = _stub_llm(
            _tool_turn(
                "run_command",
                {"command": "python -m pytest -q", "cwd": "."},
                content="Run tests for validation",
            )
        )
        resumed_app = build_graph(cfg, chat_model=resume_llm)
        resumed_state = AgentState(**{**paused, "resume_action": "approve"})  # type: ignore[misc]

        result = resumed_app.invoke(resumed_state)

        assert not (tmp_path / "notes.txt").exists()
        assert result["pending_verification"] is None
        assert result["last_verification"] is not None
        assert result["last_verification"]["ok"] is False
        assert result["last_rollback"] is not None
        assert result["last_rollback"]["ok"] is True
        assert result["last_rollback"]["trigger"] == "verify_failure"
        assert "rolled back automatically" in (result["final_answer"] or "")
        assert "Rollback result:" in (result["final_answer"] or "")

        import json as _json

        events = [_json.loads(line) for line in (log_dir / f"{run_id}.jsonl").read_text().splitlines() if line]
        write_rollback = next(event for event in events if event["event"] == "write_rollback")
        assert write_rollback["data"]["trigger"] == "verify_failure"
        run_end = [event for event in events if event["event"] == "run_end"][-1]
        assert run_end["data"]["status"] == "rolled_back_after_failed_verification"
        assert run_end["data"]["verification_status"] == "failed"

    def test_rejected_resume_skips_write_execution(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, log_dir=tmp_path / "logs")
        run_id = str(uuid.uuid4())
        pause_llm = _stub_llm(
            _tool_turn(
                "write_file",
                {"path": "notes.txt", "content": "hello\n", "mode": "create_only"},
                content="Create notes.txt",
            )
        )
        app = build_graph(cfg, chat_model=pause_llm)
        paused_state = _initial_state("Create notes.txt", str(tmp_path))
        paused_state["run_id"] = run_id
        paused = app.invoke(paused_state)

        resumed_app = build_graph(cfg, chat_model=_stub_llm(_final_turn("unused")))
        resumed_state = AgentState(**{**paused, "resume_action": "reject"})  # type: ignore[misc]

        result = resumed_app.invoke(resumed_state)

        assert not (tmp_path / "notes.txt").exists()
        assert result["iteration_count"] == 0
        assert result["observations"] == []
        assert "rejected approval request" in (result["final_answer"] or "").lower()
        assert "narrower manual change" in (result["final_answer"] or "").lower()
        assert not state_path(run_id, cfg).exists()


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

    def test_run_command_events_include_command_metadata(self, tmp_path: Path) -> None:
        """Command tool audit records should expose command-specific fields."""
        import json as _json

        log_dir = tmp_path / "logs"
        cfg = _make_config(tmp_path, log_dir=log_dir)
        run_id = str(uuid.uuid4())
        llm = _stub_llm(
            _tool_turn(
                "run_command",
                {"command": "pwd", "cwd": "."},
                content="Running pwd",
            ),
            _final_turn("Done."),
        )
        app = build_graph(cfg, chat_model=llm)
        state = _initial_state("Run pwd", str(tmp_path))
        state = AgentState(**{**state, "run_id": run_id})  # type: ignore[misc]

        app.invoke(state)

        events = [_json.loads(ln) for ln in (log_dir / f"{run_id}.jsonl").read_text().splitlines() if ln]
        by_type = {event["event"]: event for event in events if event["event"] in {"tool_proposed", "policy_decision", "tool_result", "run_end"}}
        assert by_type["tool_proposed"]["data"]["command"] == "pwd"
        assert by_type["tool_proposed"]["data"]["argv"] == ["pwd"]
        assert by_type["policy_decision"]["data"]["cwd"] == "."
        assert by_type["tool_result"]["data"]["exit_code"] == 0
        assert by_type["tool_result"]["data"]["command"] == "pwd"
        assert by_type["run_end"]["data"]["command_count"] == 1

    def test_write_tool_policy_decision_logs_approval_reason(self, tmp_path: Path) -> None:
        """Approval-gated write requests should persist structured approval metadata."""
        import json as _json

        log_dir = tmp_path / "logs"
        cfg = _make_config(tmp_path, log_dir=log_dir)
        run_id = str(uuid.uuid4())
        llm = _stub_llm(
            _tool_turn(
                "write_file",
                {"path": "notes.txt", "content": "updated"},
                content="Write notes.txt",
            )
        )
        app = build_graph(cfg, chat_model=llm)
        state = _initial_state("Write notes.txt", str(tmp_path))
        state = AgentState(**{**state, "run_id": run_id})  # type: ignore[misc]

        result = app.invoke(state)

        events = [_json.loads(ln) for ln in (log_dir / f"{run_id}.jsonl").read_text().splitlines() if ln]
        policy_event = next(event for event in events if event["event"] == "policy_decision")
        assert policy_event["data"]["decision"] == "needs_approval"
        assert policy_event["data"]["reason"] == "Write operations require explicit approval before execution."
        assert policy_event["data"]["approval_request_id"] == result["pending_approval"]["id"]
        assert "notes.txt" in policy_event["data"]["impact_summary"]
