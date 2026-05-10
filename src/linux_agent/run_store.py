"""Persistence helpers for resumable Linux Agent runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from linux_agent.config import AgentConfig
from linux_agent.state import AgentState

_STATE_DIRNAME = "state"
_STATE_VERSION = 2


def _default_plan_steps(
    plan: list[str],
    current_step: str | None,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, title in enumerate(plan, start=1):
        status = "in_progress" if current_step and title == current_step else "pending"
        steps.append(
            {
                "id": f"step_{index}",
                "title": title,
                "status": status,
                "rationale": None,
                "evidence_refs": [],
            }
        )
    return steps


def _normalize_budget_status(
    raw_budget_status: Any,
    *,
    iteration_count: int,
    command_count: int,
) -> dict[str, Any]:
    budget_status = {
        "iteration_count": iteration_count,
        "command_count": command_count,
        "elapsed_seconds": 0,
        "warning_triggered": False,
    }
    if isinstance(raw_budget_status, dict):
        budget_status.update(
            {
                "iteration_count": int(
                    raw_budget_status.get("iteration_count", iteration_count)
                ),
                "command_count": int(
                    raw_budget_status.get("command_count", command_count)
                ),
                "elapsed_seconds": int(raw_budget_status.get("elapsed_seconds", 0)),
                "warning_triggered": bool(
                    raw_budget_status.get("warning_triggered", False)
                ),
            }
        )
    return budget_status


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _deserialize_message(payload: dict[str, Any]) -> BaseMessage:
    message_type = str(payload.get("type", "human"))
    content = payload.get("content", "")
    if message_type == "system":
        return SystemMessage(content=content)
    if message_type == "ai":
        tool_calls = payload.get("tool_calls")
        return AIMessage(content=content, tool_calls=tool_calls or [])
    if message_type == "tool":
        return ToolMessage(content=cast(str, content), tool_call_id=str(payload.get("tool_call_id", "")))
    return HumanMessage(content=content)


def state_dir(config: AgentConfig) -> Path:
    return config.log_dir / _STATE_DIRNAME


def state_path(run_id: str, config: AgentConfig) -> Path:
    return state_dir(config) / f"{run_id}.json"


def save_run_state(state: AgentState, config: AgentConfig) -> Path:
    path = state_path(state["run_id"], config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _STATE_VERSION,
        "state": {
            **state,
            "messages": [_serialize_message(message) for message in state["messages"]],
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_run_state(run_id: str, config: AgentConfig) -> AgentState:
    path = state_path(run_id, config)
    if not path.exists():
        raise FileNotFoundError(f"Run state not found for {run_id}: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_state = payload.get("state")
    if not isinstance(raw_state, dict):
        raise ValueError(f"Run state file is invalid: {path}")

    messages = raw_state.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError(f"Run state messages are invalid: {path}")

    plan = [str(step) for step in cast(list[Any], raw_state.get("plan", []))]
    current_step = cast(str | None, raw_state.get("current_step"))
    iteration_count = int(raw_state.get("iteration_count", 0))
    command_count = int(raw_state.get("command_count", 0))

    raw_plan_steps = raw_state.get("plan_steps")
    plan_steps = (
        cast(list[dict[str, Any]], raw_plan_steps)
        if isinstance(raw_plan_steps, list)
        else _default_plan_steps(plan, current_step)
    )
    budget_status = _normalize_budget_status(
        raw_state.get("budget_status"),
        iteration_count=iteration_count,
        command_count=command_count,
    )

    return AgentState(
        run_id=str(raw_state["run_id"]),
        user_goal=str(raw_state["user_goal"]),
        workspace_root=str(raw_state["workspace_root"]),
        started_at=cast(str | None, raw_state.get("started_at")),
        messages=[_deserialize_message(cast(dict[str, Any], message)) for message in messages],
        plan=plan,
        command_count=command_count,
        plan_version=int(raw_state.get("plan_version", 1 if plan else 0)),
        plan_revision_count=int(raw_state.get("plan_revision_count", 0)),
        plan_steps=plan_steps,
        last_reflection=cast(dict[str, Any] | None, raw_state.get("last_reflection")),
        recovery_state=cast(dict[str, Any] | None, raw_state.get("recovery_state")),
        budget_status=budget_status,
        budget_stop_reason=cast(str | None, raw_state.get("budget_stop_reason")),
        current_step=current_step,
        proposed_tool_call=cast(dict[str, Any] | None, raw_state.get("proposed_tool_call")),
        observations=cast(list[dict[str, Any]], raw_state.get("observations", [])),
        risk_decision=cast(str | None, raw_state.get("risk_decision")),
        pending_approval=cast(dict[str, Any] | None, raw_state.get("pending_approval")),
        resume_action=cast(str | None, raw_state.get("resume_action")),
        pending_verification=cast(dict[str, Any] | None, raw_state.get("pending_verification")),
        last_write=cast(dict[str, Any] | None, raw_state.get("last_write")),
        last_verification=cast(dict[str, Any] | None, raw_state.get("last_verification")),
        last_rollback=cast(dict[str, Any] | None, raw_state.get("last_rollback")),
        iteration_count=iteration_count,
        consecutive_failures=int(raw_state.get("consecutive_failures", 0)),
        final_answer=cast(str | None, raw_state.get("final_answer")),
    )


def delete_run_state(run_id: str, config: AgentConfig) -> None:
    path = state_path(run_id, config)
    if path.exists():
        path.unlink()