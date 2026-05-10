"""T17 – End-to-end integration tests."""

from __future__ import annotations

import json
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared with test_graph; duplicated deliberately to keep isolation)
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(tmp_path: Path, **kwargs: Any) -> AgentConfig:
    return AgentConfig(workspace_root=tmp_path, **kwargs)  # type: ignore[arg-type]


def _initial_state(goal: str, workspace_root: str) -> AgentState:
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


class _CyclicStubLLM(BaseChatModel):
    """Cycles through a list of tool-calling turns; repeats the last one."""

    turns: list[_StubTurn]
    _call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "cyclic-stub"

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


def _stub(*turns: _StubTurn) -> _CyclicStubLLM:
    return _CyclicStubLLM(turns=list(turns))


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 – list_dir
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioListDir:
    def test_agent_lists_workspace_files(self, tmp_path: Path) -> None:
        """Agent uses list_dir to discover files and returns a final answer."""
        (tmp_path / "README.md").write_text("# Project")
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "util.py").write_text("pass")

        cfg = _make_config(tmp_path)
        llm = _stub(
            _tool_turn("list_dir", {"path": "."}, content="Listing workspace root"),
            _final_turn("Files: README.md, main.py, sub/"),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("What files are in the workspace?", str(tmp_path)))

        assert result["final_answer"] is not None
        assert result["iteration_count"] == 1
        assert len(result["observations"]) == 1
        obs = result["observations"][0]
        assert obs["tool"] == "list_dir"
        assert obs["ok"] is True
        assert obs["result"] is not None
        entry_names = {e["name"] for e in obs["result"]["entries"]}
        assert "README.md" in entry_names
        assert "main.py" in entry_names

    def test_recursive_list(self, tmp_path: Path) -> None:
        """Recursive list_dir exposes nested files."""
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("x")

        cfg = _make_config(tmp_path)
        llm = _stub(
            _tool_turn(
                "list_dir",
                {"path": ".", "recursive": True},
                content="Recursive listing",
            ),
            _final_turn("Found deep.py"),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Find all files", str(tmp_path)))

        obs = result["observations"][0]
        assert obs["ok"] is True
        paths = {e["path"] for e in obs["result"]["entries"]}  # type: ignore[index]
        assert "a/b/deep.py" in paths


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 – read_file
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioReadFile:
    def test_agent_reads_readme(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# My Project\n\nThis is a test.\n")
        cfg = _make_config(tmp_path)
        llm = _stub(
            _tool_turn("read_file", {"path": "README.md"}, content="Reading README.md"),
            _final_turn("README says: My Project"),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("What is in README.md?", str(tmp_path)))

        assert result["final_answer"] == "README says: My Project"
        obs = result["observations"][0]
        assert obs["tool"] == "read_file"
        assert obs["ok"] is True
        assert "My Project" in obs["result"]["content"]  # type: ignore[index]

    def test_read_specific_line_range(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("\n".join(f"# line {i}" for i in range(1, 21)) + "\n")
        cfg = _make_config(tmp_path)
        llm = _stub(
            _tool_turn(
                "read_file",
                {"path": "code.py", "start_line": 5, "end_line": 10},
                content="Reading lines 5-10",
            ),
            _final_turn("Seen lines 5-10"),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Read lines 5-10", str(tmp_path)))

        obs = result["observations"][0]
        assert obs["ok"] is True
        content = obs["result"]["content"]  # type: ignore[index]
        assert "# line 5" in content
        assert "# line 10" in content
        assert "# line 11" not in content


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 – search_text
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioSearchText:
    def test_find_function_definition(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def calculate():\n    return 42\n")
        (tmp_path / "utils.py").write_text("def helper():\n    pass\n")
        cfg = _make_config(tmp_path)
        llm = _stub(
            _tool_turn(
                "search_text",
                {"query": "def calculate"},
                content="Searching for def calculate",
            ),
            _final_turn("Found in main.py"),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Where is calculate defined?", str(tmp_path)))

        obs = result["observations"][0]
        assert obs["tool"] == "search_text"
        assert obs["ok"] is True
        matches = obs["result"]["matches"]  # type: ignore[index]
        assert any("main.py" in m["file"] for m in matches)

    def test_search_with_glob_filter(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("TOKEN_HERE")
        (tmp_path / "b.md").write_text("TOKEN_HERE")
        cfg = _make_config(tmp_path)
        llm = _stub(
            _tool_turn(
                "search_text",
                {"query": "TOKEN_HERE", "glob": "**/*.py"},
                content="Searching .py files",
            ),
            _final_turn("Found in .py files"),
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Find TOKEN in py files", str(tmp_path)))

        obs = result["observations"][0]
        assert obs["ok"] is True
        for m in obs["result"]["matches"]:  # type: ignore[index]
            assert m["file"].endswith(".py")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 – security block (path traversal)
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioSecurityBlock:
    def test_path_traversal_blocked_no_tool_execution(self, tmp_path: Path) -> None:
        """The agent must NOT execute any tool when PolicyGuard denies."""
        cfg = _make_config(tmp_path)
        llm = _stub(
            _tool_turn(
                "read_file",
                {"path": "../../../../etc/passwd"},
                content="Reading /etc/passwd",
            )
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Read /etc/passwd", str(tmp_path)))

        assert result["risk_decision"] == "deny"
        assert result["iteration_count"] == 0   # ToolExecutor never ran
        assert result["observations"] == []
        assert result["final_answer"] is not None
        assert "denied" in (result["final_answer"] or "").lower()

    def test_sensitive_path_blocked(self, tmp_path: Path) -> None:
        """Paths hitting sensitive_path_parts (.ssh) are denied by PolicyGuard."""
        (tmp_path / ".ssh").mkdir()
        (tmp_path / ".ssh" / "id_rsa").write_text("fake key")
        cfg = _make_config(tmp_path)
        llm = _stub(
            _tool_turn(
                "read_file",
                {"path": ".ssh/id_rsa"},
                content="Reading .ssh/id_rsa",
            )
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Read SSH key", str(tmp_path)))

        assert result["risk_decision"] == "deny"
        assert result["observations"] == []

    def test_write_tool_blocked(self, tmp_path: Path) -> None:
        """A non-read-only tool is rejected without any tool execution."""
        cfg = _make_config(tmp_path)
        llm = _stub(
            _tool_turn(
                "write_file",
                {"path": "evil.sh", "content": "rm -rf /"},
                content="Writing evil.sh",
            )
        )
        app = build_graph(cfg, chat_model=llm)
        result = app.invoke(_initial_state("Write file", str(tmp_path)))

        assert result["risk_decision"] == "deny"
        assert result["iteration_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 – iteration limit (circuit breaker)
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioIterationLimit:
    def test_agent_stops_at_max_iterations(self, tmp_path: Path) -> None:
        """Reflector must stop the agent when max_iterations is reached."""
        (tmp_path / "f.txt").write_text("x")
        cfg = _make_config(tmp_path, max_iterations=3)
        always_list = _tool_turn("list_dir", {"path": "."}, content="Listing (loop)")
        app = build_graph(cfg, chat_model=_stub(always_list))
        result = app.invoke(_initial_state("Loop forever", str(tmp_path)))

        assert result["iteration_count"] >= cfg.max_iterations
        assert result["final_answer"] is not None
        assert "maximum iteration" in (result["final_answer"] or "").lower()
        # All executed observations should be from list_dir
        assert all(o["tool"] == "list_dir" for o in result["observations"])

    def test_consecutive_failure_limit(self, tmp_path: Path) -> None:
        """Reflector stops when max_consecutive_failures is reached."""
        cfg = _make_config(tmp_path, max_consecutive_failures=2)
        always_fail = _tool_turn(
            "read_file",
            {"path": "ghost.txt"},
            content="Reading ghost.txt",
        )
        app = build_graph(cfg, chat_model=_stub(always_fail))
        result = app.invoke(_initial_state("Read ghost", str(tmp_path)))

        assert result["final_answer"] is not None
        assert "consecutive" in (result["final_answer"] or "").lower()
        # Failure count in observations
        assert all(not o["ok"] for o in result["observations"])


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 – audit log completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioAuditLog:
    def test_full_run_produces_complete_audit_trail(self, tmp_path: Path) -> None:
        """A typical run must produce run_start (via app.py) + plan_update +
        tool_proposed + policy_decision + tool_result + run_end events."""
        log_dir = tmp_path / "logs"
        cfg = _make_config(tmp_path, log_dir=log_dir)
        run_id = str(uuid.uuid4())

        (tmp_path / "hello.txt").write_text("hi")
        llm = _stub(
            _tool_turn("read_file", {"path": "hello.txt"}, content="Reading hello.txt"),
            _final_turn("Content: hi"),
        )
        app = build_graph(cfg, chat_model=llm)
        initial = AgentState(
            run_id=run_id,
            user_goal="Read hello.txt",
            workspace_root=str(tmp_path),
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
        app.invoke(initial)

        log_file = log_dir / f"{run_id}.jsonl"
        assert log_file.exists()
        events_by_type = {}
        for line in log_file.read_text().splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            events_by_type.setdefault(ev["event"], []).append(ev)

        assert "plan_update" in events_by_type
        assert "tool_proposed" in events_by_type
        assert "policy_decision" in events_by_type
        assert "tool_result" in events_by_type
        assert "run_end" in events_by_type

        # Validate schema of one tool_result event
        tr = events_by_type["tool_result"][0]
        assert tr["run_id"] == run_id
        assert "ts" in tr
        assert "ok" in tr["data"]
        assert tr["data"]["tool"] == "read_file"
