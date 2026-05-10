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

from datetime import datetime, timezone
import json
import time
from typing import Any, Callable, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from linux_agent.audit import (
    AuditEventListener,
    EVENT_MODEL_INPUT,
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
# Tool schemas exposed to the model
# ─────────────────────────────────────────────────────────────────────────────

@tool("list_dir")
def list_dir_tool(path: str, recursive: bool = False) -> str:
    """List entries under a workspace-relative directory path. Use '.' for the workspace root."""
    raise RuntimeError("list_dir is executed by linux_agent.graph, not by LangChain.")


@tool("read_file")
def read_file_tool(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read a workspace-relative text file. Provide start_line/end_line only when you need a slice."""
    raise RuntimeError("read_file is executed by linux_agent.graph, not by LangChain.")


@tool("search_text")
def search_text_tool(
    query: str,
    path: str = ".",
    glob: str = "**/*",
    context_lines: int = 2,
) -> str:
    """Search for a literal string inside workspace files. Use glob to narrow the search when helpful."""
    raise RuntimeError("search_text is executed by linux_agent.graph, not by LangChain.")


_MODEL_TOOLS = [list_dir_tool, read_file_tool, search_text_tool]


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

- Use tool calling directly when you need more evidence.
- Call at most one tool at a time.
- Answer in plain text when you already have enough information.
- Do not invent files, directories, or file contents.
- Keep tool arguments minimal and precise.
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
    listener: AuditEventListener | None = None,
) -> None:
    """Open, write one audit event, and close the JSONL logger."""
    with AuditLogger(run_id, log_dir, listener=listener) as logger:
        logger.log(event, data)


def _emit_runtime_event(
    run_id: str,
    event: str,
    data: dict[str, Any],
    listener: AuditEventListener | None = None,
) -> None:
    if listener is None:
        return

    listener(
        {
            "run_id": run_id,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "event": event,
            "data": data,
        }
    )


def _serialize_tool_payload(obs: Observation, limit: int = 4000) -> str:
    payload: dict[str, Any] = {
        "tool": obs["tool"],
        "ok": obs["ok"],
        "result": obs["result"],
        "error": obs["error"],
        "duration_ms": obs["duration_ms"],
    }
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > limit:
        return text[:limit] + " …"
    return text


def _serialize_trace_message(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }

    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = message.tool_calls

    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id

    return payload


def _format_planner_prompt(state: AgentState, config: AgentConfig) -> str:
    plan_text = (
        "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(state["plan"]))
        or "  (not yet planned)"
    )
    obs_text = _fmt_observations(state["observations"])
    return (
        f"Goal: {state['user_goal']}\n\n"
        f"Workspace root: {state['workspace_root']}\n\n"
        f"Current plan:\n{plan_text}\n\n"
        f"Recent observations:\n{obs_text}\n\n"
        f"Iterations used: {state['iteration_count']} / {config.max_iterations}\n"
        f"Consecutive failures: {state['consecutive_failures']} / "
        f"{config.max_consecutive_failures}"
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    parts.append(part.strip())
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _bind_model_tools(llm: BaseChatModel) -> Any:
    try:
        return llm.bind_tools(_MODEL_TOOLS, parallel_tool_calls=False)
    except TypeError:
        return llm.bind_tools(_MODEL_TOOLS)


def _coerce_ai_message(response: Any) -> AIMessage:
    if isinstance(response, AIMessage):
        return _normalize_ai_message(response)

    content = getattr(response, "content", response)
    tool_calls = getattr(response, "tool_calls", None) or []
    return _normalize_ai_message(AIMessage(content=content, tool_calls=tool_calls))


def _normalize_ai_message(message: AIMessage) -> AIMessage:
    if len(message.tool_calls) <= 1:
        return message

    # The graph executes one tool per turn. If the provider emits multiple
    # tool calls anyway, keep only the first one so the stored assistant
    # history stays consistent with the single ToolMessage we will append.
    return cast(
        AIMessage,
        message.model_copy(
            update={
                "tool_calls": [message.tool_calls[0]],
                "invalid_tool_calls": [],
            }
        ),
    )


def _coerce_tool_args(raw_args: Any) -> dict[str, object]:
    if isinstance(raw_args, dict):
        return {str(key): cast(object, value) for key, value in raw_args.items()}
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(key): cast(object, value) for key, value in parsed.items()}
    return {}


def _build_tool_call(message: AIMessage, state: AgentState) -> ToolCall | None:
    tool_calls = message.tool_calls or []
    if not tool_calls:
        return None

    raw_call = tool_calls[0]
    raw_name = raw_call.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None

    raw_id = raw_call.get("id")
    call_id = str(raw_id) if raw_id else f"{raw_name}_{state['iteration_count'] + 1}"
    return ToolCall(
        id=call_id,
        name=raw_name,
        args=_coerce_tool_args(raw_call.get("args", {})),
        risk_level="low",
    )


def _tool_step(tool_call: ToolCall) -> str:
    args_json = json.dumps(tool_call["args"], ensure_ascii=False)
    return f"Calling {tool_call['name']} {args_json}"


def _summarize_step(message: AIMessage, tool_call: ToolCall | None) -> str:
    content = _content_to_text(message.content)
    if content:
        first_line = content.splitlines()[0].strip()
        if first_line:
            return first_line[:240]
    if tool_call is not None:
        return _tool_step(tool_call)
    return "Preparing final answer"


def _append_plan_step(plan: list[str], step: str) -> list[str]:
    normalized = step.strip()
    if not normalized:
        return plan
    if plan and plan[-1] == normalized:
        return plan
    return [*plan, normalized]


def _is_deepseek_model(model_name: str) -> bool:
    return model_name.strip().lower().startswith("deepseek")


# ─────────────────────────────────────────────────────────────────────────────
# T9 – Planner node
# ─────────────────────────────────────────────────────────────────────────────

def _make_planner(
    config: AgentConfig,
    llm: BaseChatModel,
    event_listener: AuditEventListener | None = None,
    prompt_trace_listener: AuditEventListener | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    """Return a Planner node function closed over *config* and *llm*."""
    llm_with_tools = _bind_model_tools(llm)

    def planner(state: AgentState) -> dict[str, Any]:
        run_id = state["run_id"]
        prompt_message = HumanMessage(content=_format_planner_prompt(state, config))
        history: list[BaseMessage] = state["messages"]
        request_messages = [SystemMessage(content=_SYSTEM_PROMPT), *history, prompt_message]
        _emit_runtime_event(
            run_id,
            EVENT_MODEL_INPUT,
            {
                "message_count": len(request_messages),
                "messages": [_serialize_trace_message(message) for message in request_messages],
            },
            prompt_trace_listener,
        )
        raw = llm_with_tools.invoke(request_messages)
        response = _coerce_ai_message(raw)
        tool_call = _build_tool_call(response, state)
        assistant_content = _content_to_text(response.content)
        current_step = _summarize_step(response, tool_call)
        plan = _append_plan_step(state["plan"], current_step)

        final_answer: str | None = None
        if tool_call is None:
            final_answer = _content_to_text(response.content) or (
                "Agent returned neither a tool call nor a final answer."
            )

        # ── Audit ────────────────────────────────────────────────────────
        _one_shot_audit(
            run_id,
            config.log_dir,
            EVENT_PLAN_UPDATE,
            {
                "plan": plan,
                "current_step": current_step,
                "assistant_content": assistant_content or None,
                "final_answer": final_answer,
            },
            event_listener,
        )
        if tool_call:
            _one_shot_audit(
                run_id,
                config.log_dir,
                EVENT_TOOL_PROPOSED,
                {
                    "tool": tool_call["name"],
                    "tool_call_id": tool_call["id"],
                    "args": tool_call["args"],
                },
                event_listener,
            )

        return {
            "messages": [prompt_message, response],
            "plan": plan,
            "current_step": current_step,
            "proposed_tool_call": tool_call,
            "final_answer": final_answer,
        }

    return planner


# ─────────────────────────────────────────────────────────────────────────────
# T10 – PolicyGuard node
# ─────────────────────────────────────────────────────────────────────────────

def _make_policy_guard(
    config: AgentConfig,
    event_listener: AuditEventListener | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    def policy_guard(state: AgentState) -> dict[str, Any]:
        tc = state["proposed_tool_call"]
        if tc is None:
            # No tool proposed (Planner set final_answer); allow trivially
            return {"risk_decision": "allow"}

        decision = evaluate_tool_call(tc, config)

        _one_shot_audit(
            state["run_id"],
            config.log_dir,
            EVENT_POLICY_DECISION,
            {
                "tool": tc["name"],
                "tool_call_id": tc["id"],
                "args": tc["args"],
                "decision": decision,
            },
            event_listener,
        )

        if decision == "deny":
            return {
                "risk_decision": "deny",
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
    event_listener: AuditEventListener | None = None,
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
                "proposed_tool_call": None,
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

        _one_shot_audit(
            state["run_id"],
            config.log_dir,
            EVENT_TOOL_RESULT,
            {
                "tool": obs["tool"],
                "tool_call_id": tc["id"],
                "ok": obs["ok"],
                "duration_ms": obs["duration_ms"],
                "error": obs["error"],
                "result": obs["result"],
            },
            event_listener,
        )

        # Reset consecutive_failures on success; increment on failure
        new_failures = 0 if obs["ok"] else state["consecutive_failures"] + 1

        return {
            "messages": [
                ToolMessage(
                    content=_serialize_tool_payload(obs),
                    tool_call_id=tc["id"],
                )
            ],
            "observations": state["observations"] + [obs],
            "iteration_count": state["iteration_count"] + 1,
            "consecutive_failures": new_failures,
            "proposed_tool_call": None,
        }

    return tool_executor


# ─────────────────────────────────────────────────────────────────────────────
# T12 – Reflector node
# ─────────────────────────────────────────────────────────────────────────────

def _make_reflector(
    config: AgentConfig,
    event_listener: AuditEventListener | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    def reflector(state: AgentState) -> dict[str, Any]:
        run_id = state["run_id"]

        # Circuit-breaker: total iteration limit
        if state["iteration_count"] >= config.max_iterations:
            summary = _fmt_observations(state["observations"], n=3)
            _one_shot_audit(
                run_id,
                config.log_dir,
                EVENT_REFLECTOR_ACTION,
                {
                    "reason": "max_iterations",
                    "iteration_count": state["iteration_count"],
                },
                event_listener,
            )
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
            _one_shot_audit(
                run_id,
                config.log_dir,
                EVENT_REFLECTOR_ACTION,
                {
                    "reason": "consecutive_failures",
                    "count": state["consecutive_failures"],
                    "last_error": last_err,
                },
                event_listener,
            )
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
    event_listener: AuditEventListener | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    def finalizer(state: AgentState) -> dict[str, Any]:
        final = state.get("final_answer") or "(no answer produced)"

        _one_shot_audit(
            state["run_id"],
            config.log_dir,
            EVENT_RUN_END,
            {
                "final_answer": final,
                "iteration_count": state["iteration_count"],
                "observation_count": len(state["observations"]),
            },
            event_listener,
        )

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
    event_listener: AuditEventListener | None = None,
    prompt_trace_listener: AuditEventListener | None = None,
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
        if _is_deepseek_model(config.llm_model):
            # ChatOpenAI does not round-trip DeepSeek reasoning_content across
            # tool-calling turns, so disable thinking mode when using the
            # OpenAI-compatible client.
            llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if config.llm_base_url is not None:
            llm_kwargs["base_url"] = config.llm_base_url
        if config.llm_api_key is not None:
            llm_kwargs["api_key"] = config.llm_api_key
        chat_model = ChatOpenAI(**llm_kwargs)

    graph: Any = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node(
        "planner",
        _make_planner(config, chat_model, event_listener, prompt_trace_listener),
    )
    graph.add_node("policy_guard", _make_policy_guard(config, event_listener))
    graph.add_node("tool_executor", _make_tool_executor(config, event_listener))
    graph.add_node("reflector", _make_reflector(config, event_listener))
    graph.add_node("finalizer", _make_finalizer(config, event_listener))

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

