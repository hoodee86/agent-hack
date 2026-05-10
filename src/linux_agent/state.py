"""
AgentState and related schema definitions for the Linux Agent.

All state types are defined as TypedDict so they work natively with
LangGraph's StateGraph. Field-level reducers (e.g. add_messages) are
declared via Annotated to let the graph merge partial updates correctly.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ToolCall(TypedDict):
    """A single tool invocation proposed by the Planner node."""

    # Provider-generated tool-call id so ToolMessage can be correlated
    id: str
    # Tool name – one of the current tool schemas; future approval-only
    # proposals may also include write_file or apply_patch.
    name: str
    # Raw arguments forwarded to the skill function unchanged
    args: dict[str, object]
    # Pre-classified risk level; read-only tools default to "low"
    risk_level: Literal["low", "medium", "high"]


class Observation(TypedDict):
    """The structured result produced by the Tool Executor node."""

    # Name of the tool that was executed
    tool: str
    # True when the skill returned without error
    ok: bool
    # Structured payload returned by the skill, including failure metadata
    # when a tool reports a structured error result.
    result: dict[str, object] | None
    # Human-readable error description (None when ok=True)
    error: str | None
    # Wall-clock time for the skill call in milliseconds
    duration_ms: int


class ApprovalRequest(TypedDict):
    """Structured approval metadata produced by Policy Guard."""

    id: str
    tool: str
    args: dict[str, object]
    reason: str
    impact_summary: str
    diff_preview: str | None
    backup_plan: str | None


class AgentState(TypedDict):
    """
    Full mutable state threaded through the LangGraph state machine.

    Fields marked with Annotated[..., reducer] are merged by the graph
    engine on every step instead of being overwritten outright.
    """

    # Unique identifier for this agent run (UUID4 string)
    run_id: str

    # The user-supplied goal that the agent is trying to accomplish
    user_goal: str

    # Absolute path to the sandboxed workspace directory
    workspace_root: str

    # Chat-style message history; add_messages appends rather than replaces
    messages: Annotated[list[BaseMessage], add_messages]

    # Ordered list of steps the Planner believes are needed to reach the goal
    plan: list[str]

    # Human-readable description of the step being executed right now
    current_step: str | None

    # Tool invocation proposed by the Planner, pending Policy Guard review
    proposed_tool_call: ToolCall | None

    # Accumulated results from every tool execution in this run
    observations: list[Observation]

    # Latest decision from the Policy Guard node.
    risk_decision: Literal["allow", "deny", "needs_approval"] | None

    # Populated when Policy Guard pauses a write operation for approval.
    pending_approval: ApprovalRequest | None

    # Set only when a paused run is resumed from the CLI.
    resume_action: Literal["approve", "reject"] | None

    # Total number of tool executions so far (incremented by Tool Executor)
    iteration_count: int

    # Consecutive failure counter; reset to 0 on any successful observation
    consecutive_failures: int

    # Set by the Planner (task done) or the Reflector (circuit-breaker).
    # Non-None value signals the graph to route to the Finalizer.
    final_answer: str | None
