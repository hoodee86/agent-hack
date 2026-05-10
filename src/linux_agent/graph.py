"""
LangGraph state machine for the Linux Agent (T9-T14).

Nodes
-----
- planner      (T9):  LLM-powered; decides next tool call or signals completion.
- policy_guard (T10): Enforces read-only policy; denies or allows the proposed tool.
- tool_executor(T11): Dispatches to skills; records an Observation.
- reflector    (T12): Pure-logic circuit-breaker; checks iteration/failure limits.
- finalizer    (T13): Emits the final answer and writes the run_end audit event.

Graph topology
--------------
    START → planner ──► policy_guard ──► tool_executor ──► reflector ──┐
               ↑                                                        │ (loop)
               └────────────────────────────────────────────────────────┘

    planner      → finalizer   when final_answer is set
    policy_guard → finalizer   when risk_decision == "deny"
    reflector    → finalizer   when circuit-breaker trips
    finalizer    → END
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from linux_agent.audit import (
    EVENT_PLAN_UPDATE,
    EVENT_POLICY_DECISION,
    EVENT_REFLECTOR_ACTION,
    EVENT_RUN_END,
    EVENT_TOOL_PROPOSED,
    EVENT_TOOL_RESULT,
    AuditLogger,
)
from linux_agent.config import AgentConfig
from linux_agent.policy import PolicyViolation, evaluate_tool_call
from linux_agent.skills.filesystem import list_dir, read_file
from linux_agent.skills.search import search_text
from linux_agent.state import AgentState, Observation, ToolCall


# ─────────────────────────────────────────────────────────────────────────────
# T9 – Planner output schema
# ─────────────────────────────────────────────────────────────────────────────

class PlannerDecision(BaseModel):
    """Structured output produced by the Planner LLM call."""

    thought: str = Field(
        description="Step-by-step reasoning about the current state and what to do next."
    )
    plan: list[str] = Field(
        description="Updated ordered list of steps to accomplish the goal."
    )
    current_step: str = Field(
        description="Human-readable description of the current action being taken."
    )
    tool_name: str | None = Field(
        default=None,
        description=(
            "Tool to invoke: 'list_dir' | 'read_file' | 'search_text'. "
            "Set to null only when final_answer is also set."
        ),
    )
    tool_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the tool. Empty when tool_name is null.",
    )
    final_answer: str | None = Field(
        default=None,
        description=(
            "The complete answer to the user's goal. Set when finished or unable to progress. "
            "Null while still working."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a read-only Linux filesystem agent.
Your task is to answer the user's goal by exploring the filesystem with
the tools provided. You MUST NOT write, delete, or modify any files.

## Available tools

| Tool        | Required args                   | Optional args                                          |
|-------------|---------------------------------|--------------------------------------------------------|
| list_dir    | path: str (workspace-relative)  | recursive: bool (default false)                        |
| read_file   | path: str (workspace-relative)  | start_line: int (1-based), end_line: int               |
| search_text | query: str                      | path: str (default "."), glob: str, context_lines: int |

All paths must be relative to the workspace root. Use "." for the root itself.

## Rules

- Keep `final_answer` null and provide a `tool_name` while you are still gathering information.
- Set `final_answer` (non-null) when:
  1. You have enough information to fully answer the goal, OR
  2. You cannot make further progress — explain why.
- Never set both `tool_name` and `final_answer` at the same time.
"""

# Extra suffix appended to the system prompt in "prompt" mode (no tool_choice support)
_JSON_SCHEMA_PROMPT = """
## Output format

You MUST reply with a single JSON object (no prose, no markdown fences) matching:

{
  "thought": "<your reasoning>",
  "plan": ["step 1", "step 2", ...],
  "current_step": "<what you are doing right now>",
  "tool_name": "list_dir" | "read_file" | "search_text" | null,
  "tool_args": { ... } | {},
  "final_answer": "<complete answer>" | null
}

Rules for tool_args:
- list_dir:    {"path": "<rel-path>", "recursive": true|false}
- read_file:   {"path": "<rel-path>", "start_line": <int>, "end_line": <int>}
- search_text: {"query": "<text>", "path": "<rel-path>", "glob": "**/*"}

Output ONLY the JSON object. No explanation before or after it.
"""


def _fmt_observations(observations: list[Observation], n: int = 5) -> str:
    """Format the last *n* observations for inclusion in the Planner prompt."""
    if not observations:
        return "None yet."
    recent = observations[-n:]
    start_idx = len(observations) - len(recent) + 1
    parts: list[str] = []
    for i, obs in enumerate(recent, start=start_idx):
        status = "OK" if obs["ok"] else "ERROR"
        if obs["ok"] and obs["result"]:
            raw = json.dumps(obs["result"], ensure_ascii=False)
            detail = raw[:600] + " …" if len(raw) > 600 else raw
        else:
            detail = obs["error"] or ""
        parts.append(
            f"[{i}] tool={obs['tool']}  status={status}  "
            f"duration={obs['duration_ms']}ms\n{detail}"
        )
    return "\n\n".join(parts)


def _one_shot_audit(
    run_id: str,
    log_dir: Any,
    event: str,
    data: dict[str, Any],
) -> None:
    """Open, write one audit event, and close the JSONL logger."""
    with AuditLogger(run_id, log_dir) as logger:
        logger.log(event, data)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response.

    Handles three common formats:
    1. ```json\n{...}\n```
    2. ```\n{...}\n```
    3. Raw {…} anywhere in the text
    """
    # Strip fenced code block
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        return json.loads(fenced.group(1).strip())  # type: ignore[no-any-return]
    # Find first { ... } block
    brace_start = text.find("{")
    if brace_start != -1:
        # Walk to find the matching closing brace
        depth = 0
        for i, ch in enumerate(text[brace_start:], start=brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[brace_start : i + 1])  # type: ignore[no-any-return]
    raise ValueError(f"No JSON object found in model response:\n{text[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# T9 – Planner node
# ─────────────────────────────────────────────────────────────────────────────

def _make_planner(
    config: AgentConfig,
    llm: BaseChatModel,
) -> Callable[[AgentState], dict[str, Any]]:
    """Return a Planner node function closed over *config* and *llm*.

    Supports three structured-output strategies via config.llm_structured_output_method:
    - "function_calling" : tool_choice-based (deepseek-chat, GPT models)
    - "json_schema"      : OpenAI json_schema mode (GPT-4o, o-series)
    - "prompt"           : pure text + JSON extraction (deepseek-reasoner/R1, any model)
    """
    use_prompt_mode = config.llm_structured_output_method == "prompt"

    if use_prompt_mode:
        structured_llm = None
        system_prompt = _SYSTEM_PROMPT + _JSON_SCHEMA_PROMPT
    else:
        structured_llm = llm.with_structured_output(
            PlannerDecision,
            method=config.llm_structured_output_method,
        )
        system_prompt = _SYSTEM_PROMPT

    def planner(state: AgentState) -> dict[str, Any]:
        run_id = state["run_id"]

        plan_text = (
            "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(state["plan"]))
            or "  (not yet planned)"
        )
        obs_text = _fmt_observations(state["observations"])

        user_content = (
            f"**Goal:** {state['user_goal']}\n\n"
            f"**Workspace root:** {state['workspace_root']}\n\n"
            f"**Current plan:**\n{plan_text}\n\n"
            f"**Recent observations:**\n{obs_text}\n\n"
            f"**Iterations used:** {state['iteration_count']} / {config.max_iterations}\n"
            f"**Consecutive failures:** "
            f"{state['consecutive_failures']} / {config.max_consecutive_failures}\n"
        )

        lc_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        if use_prompt_mode:
            raw = llm.invoke(lc_messages)
            text = raw.content if hasattr(raw, "content") else str(raw)
            try:
                parsed = _extract_json(str(text))
                decision = PlannerDecision(**parsed)
            except Exception as exc:
                decision = PlannerDecision(
                    thought=f"JSON parse error: {exc}",
                    plan=state["plan"],
                    current_step="Parse error",
                    final_answer=f"Agent encountered a response parse error: {exc}",
                )
        else:
            decision = structured_llm.invoke(lc_messages)  # type: ignore[union-attr,assignment]

        # Guard: if neither tool nor answer was provided, force a final answer
        if not decision.tool_name and not decision.final_answer:
            decision = PlannerDecision(
                thought=decision.thought,
                plan=decision.plan,
                current_step="Unable to determine next action",
                final_answer=(
                    "Agent could not determine how to proceed. "
                    "Please refine your goal."
                ),
            )

        # Build ToolCall when the planner wants a tool (not finishing yet)
        tool_call: ToolCall | None = None
        if decision.tool_name and not decision.final_answer:
            tool_call = ToolCall(
                name=decision.tool_name,
                args=decision.tool_args,
                risk_level="low",  # all phase-1 tools are read-only
            )

        # ── Audit ────────────────────────────────────────────────────────
        _one_shot_audit(run_id, config.log_dir, EVENT_PLAN_UPDATE, {
            "plan": decision.plan,
            "current_step": decision.current_step,
            "thought": decision.thought,
        })
        if tool_call:
            _one_shot_audit(run_id, config.log_dir, EVENT_TOOL_PROPOSED, {
                "tool": tool_call["name"],
                "args": tool_call["args"],
            })

        # ── Message for history ───────────────────────────────────────────
        if decision.final_answer:
            msg_content: str = "[Planner] Final answer ready."
        else:
            msg_content = (
                f"[Planner] {decision.current_step}\n"
                f"Tool: {decision.tool_name} {json.dumps(decision.tool_args)}"
            )
        assistant_msg: dict[str, object] = {
            "role": "assistant",
            "content": msg_content,
        }

        return {
            "messages": [assistant_msg],
            "plan": decision.plan,
            "current_step": decision.current_step,
            "proposed_tool_call": tool_call,
            "final_answer": decision.final_answer,
        }

    return planner


# ─────────────────────────────────────────────────────────────────────────────
# T10 – PolicyGuard node
# ─────────────────────────────────────────────────────────────────────────────

def _make_policy_guard(
    config: AgentConfig,
) -> Callable[[AgentState], dict[str, Any]]:
    def policy_guard(state: AgentState) -> dict[str, Any]:
        tc = state["proposed_tool_call"]
        if tc is None:
            # No tool proposed (Planner set final_answer); allow trivially
            return {"risk_decision": "allow"}

        decision = evaluate_tool_call(tc, config)

        _one_shot_audit(state["run_id"], config.log_dir, EVENT_POLICY_DECISION, {
            "tool": tc["name"],
            "args": tc["args"],
            "decision": decision,
        })

        if decision == "deny":
            # Inject a message explaining the denial
            deny_msg: dict[str, object] = {
                "role": "assistant",
                "content": (
                    f"[PolicyGuard] Denied tool call '{tc['name']}' "
                    f"with args {json.dumps(tc['args'])}."
                ),
            }
            return {
                "risk_decision": "deny",
                "messages": [deny_msg],
                "final_answer": (
                    f"Operation denied by security policy: "
                    f"tool '{tc['name']}' with args {tc['args']} is not permitted."
                ),
            }

        return {"risk_decision": "allow"}

    return policy_guard


# ─────────────────────────────────────────────────────────────────────────────
# T11 – ToolExecutor node
# ─────────────────────────────────────────────────────────────────────────────

# Dispatch table: tool name → skill function
_SKILL_DISPATCH: dict[str, Any] = {
    "list_dir": list_dir,
    "read_file": read_file,
    "search_text": search_text,
}


def _make_tool_executor(
    config: AgentConfig,
) -> Callable[[AgentState], dict[str, Any]]:
    def tool_executor(state: AgentState) -> dict[str, Any]:
        tc = state["proposed_tool_call"]

        if tc is None:
            obs = Observation(
                tool="unknown",
                ok=False,
                result=None,
                error="ToolExecutor called with no proposed tool call",
                duration_ms=0,
            )
            return {
                "observations": state["observations"] + [obs],
                "iteration_count": state["iteration_count"] + 1,
                "consecutive_failures": state["consecutive_failures"] + 1,
            }

        skill_fn = _SKILL_DISPATCH.get(tc["name"])
        t0 = time.monotonic()

        if skill_fn is None:
            duration_ms = int((time.monotonic() - t0) * 1000)
            obs = Observation(
                tool=tc["name"],
                ok=False,
                result=None,
                error=f"Unknown tool: {tc['name']}",
                duration_ms=duration_ms,
            )
        else:
            try:
                # Skills: fn(primary_arg, config, **kwargs)
                # primary arg key: "path" for list_dir/read_file, "query" for search_text
                extra = dict(tc["args"])
                if tc["name"] == "search_text":
                    primary = str(extra.pop("query", ""))
                else:
                    primary = str(extra.pop("path", "."))

                result: dict[str, Any] = skill_fn(primary, config, **extra)
                duration_ms = int((time.monotonic() - t0) * 1000)
                ok = bool(result.get("ok", False))
                obs = Observation(
                    tool=tc["name"],
                    ok=ok,
                    result=result if ok else None,
                    error=str(result["error"]) if not ok and result.get("error") else None,
                    duration_ms=duration_ms,
                )
            except PolicyViolation as exc:
                duration_ms = int((time.monotonic() - t0) * 1000)
                obs = Observation(
                    tool=tc["name"],
                    ok=False,
                    result=None,
                    error=str(exc),
                    duration_ms=duration_ms,
                )
            except Exception as exc:  # noqa: BLE001
                duration_ms = int((time.monotonic() - t0) * 1000)
                obs = Observation(
                    tool=tc["name"],
                    ok=False,
                    result=None,
                    error=f"Unexpected error: {exc}",
                    duration_ms=duration_ms,
                )

        _one_shot_audit(state["run_id"], config.log_dir, EVENT_TOOL_RESULT, {
            "tool": obs["tool"],
            "ok": obs["ok"],
            "duration_ms": obs["duration_ms"],
            "error": obs["error"],
        })

        # Reset consecutive_failures on success; increment on failure
        new_failures = 0 if obs["ok"] else state["consecutive_failures"] + 1

        return {
            "observations": state["observations"] + [obs],
            "iteration_count": state["iteration_count"] + 1,
            "consecutive_failures": new_failures,
        }

    return tool_executor


# ─────────────────────────────────────────────────────────────────────────────
# T12 – Reflector node
# ─────────────────────────────────────────────────────────────────────────────

def _make_reflector(
    config: AgentConfig,
) -> Callable[[AgentState], dict[str, Any]]:
    def reflector(state: AgentState) -> dict[str, Any]:
        run_id = state["run_id"]

        # Circuit-breaker: total iteration limit
        if state["iteration_count"] >= config.max_iterations:
            summary = _fmt_observations(state["observations"], n=3)
            _one_shot_audit(run_id, config.log_dir, EVENT_REFLECTOR_ACTION, {
                "reason": "max_iterations",
                "iteration_count": state["iteration_count"],
            })
            return {
                "final_answer": (
                    f"Agent stopped: reached the maximum iteration limit "
                    f"({config.max_iterations}).\n\n"
                    f"Last observations:\n{summary}"
                )
            }

        # Circuit-breaker: consecutive failure limit
        if state["consecutive_failures"] >= config.max_consecutive_failures:
            last_err = (
                state["observations"][-1]["error"]
                if state["observations"]
                else "no observations"
            )
            _one_shot_audit(run_id, config.log_dir, EVENT_REFLECTOR_ACTION, {
                "reason": "consecutive_failures",
                "count": state["consecutive_failures"],
                "last_error": last_err,
            })
            return {
                "final_answer": (
                    f"Agent stopped: {state['consecutive_failures']} consecutive "
                    f"tool failures.\nLast error: {last_err}"
                )
            }

        # All good – continue the loop
        return {}

    return reflector


# ─────────────────────────────────────────────────────────────────────────────
# T13 – Finalizer node
# ─────────────────────────────────────────────────────────────────────────────

def _make_finalizer(
    config: AgentConfig,
) -> Callable[[AgentState], dict[str, Any]]:
    def finalizer(state: AgentState) -> dict[str, Any]:
        final = state.get("final_answer") or "(no answer produced)"

        _one_shot_audit(state["run_id"], config.log_dir, EVENT_RUN_END, {
            "final_answer": final,
            "iteration_count": state["iteration_count"],
            "observation_count": len(state["observations"]),
        })

        return {"final_answer": final}

    return finalizer


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge routing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _route_planner(state: AgentState) -> str:
    if state.get("final_answer") is not None:
        return "finalizer"
    return "policy_guard"


def _route_policy_guard(state: AgentState) -> str:
    if state.get("risk_decision") == "deny":
        return "finalizer"
    return "tool_executor"


def _route_reflector(state: AgentState) -> str:
    if state.get("final_answer") is not None:
        return "finalizer"
    return "planner"


# ─────────────────────────────────────────────────────────────────────────────
# T14 – Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    config: AgentConfig,
    chat_model: BaseChatModel | None = None,
) -> Any:
    """
    Assemble and compile the LangGraph state machine.

    Parameters
    ----------
    config:
        Agent runtime configuration (workspace limits, LLM settings, etc.).
    chat_model:
        Optional pre-built LangChain chat model.  When *None*, a
        ``ChatOpenAI`` instance is created from ``config.llm_model`` and
        ``config.llm_temperature``.

    Returns
    -------
    CompiledStateGraph
        Ready to invoke with an initial ``AgentState`` dict.

    Example
    -------
    ::

        import uuid
        from linux_agent.config import load_config
        from linux_agent.graph import build_graph

        cfg = load_config("config.yaml")
        app = build_graph(cfg)
        result = app.invoke({
            "run_id": str(uuid.uuid4()),
            "user_goal": "List all Python files in the workspace",
            "workspace_root": str(cfg.workspace_root),
            "messages": [],
            "plan": [],
            "current_step": None,
            "proposed_tool_call": None,
            "observations": [],
            "risk_decision": None,
            "iteration_count": 0,
            "consecutive_failures": 0,
            "final_answer": None,
        })
        print(result["final_answer"])
    """
    if chat_model is None:
        llm_kwargs: dict[str, Any] = {
            "model": config.llm_model,
            "temperature": config.llm_temperature,
        }
        if config.llm_base_url is not None:
            llm_kwargs["base_url"] = config.llm_base_url
        if config.llm_api_key is not None:
            llm_kwargs["api_key"] = config.llm_api_key
        chat_model = ChatOpenAI(**llm_kwargs)

    graph: Any = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("planner", _make_planner(config, chat_model))
    graph.add_node("policy_guard", _make_policy_guard(config))
    graph.add_node("tool_executor", _make_tool_executor(config))
    graph.add_node("reflector", _make_reflector(config))
    graph.add_node("finalizer", _make_finalizer(config))

    # ── Entry point ───────────────────────────────────────────────────────
    graph.add_edge(START, "planner")

    # ── Conditional edges ─────────────────────────────────────────────────
    graph.add_conditional_edges(
        "planner",
        _route_planner,
        {"policy_guard": "policy_guard", "finalizer": "finalizer"},
    )
    graph.add_conditional_edges(
        "policy_guard",
        _route_policy_guard,
        {"tool_executor": "tool_executor", "finalizer": "finalizer"},
    )
    graph.add_edge("tool_executor", "reflector")
    graph.add_conditional_edges(
        "reflector",
        _route_reflector,
        {"planner": "planner", "finalizer": "finalizer"},
    )

    # ── Terminal ──────────────────────────────────────────────────────────
    graph.add_edge("finalizer", END)

    return graph.compile()

