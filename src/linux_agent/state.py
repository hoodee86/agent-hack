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
from typing_extensions import NotRequired, TypedDict


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
    affected_files: NotRequired[list[str]]
    risk_level: NotRequired[Literal["low", "medium", "high"]]
    suggested_verification_command: NotRequired[str | None]
    rollback_command: NotRequired[str | None]


class WriteSummary(TypedDict):
    """Metadata for the latest successfully applied write operation."""

    tool: Literal["apply_patch", "write_file"]
    changed_files: list[str]
    added_lines: int
    removed_lines: int
    backup_root: str | None
    manifest_path: str | None
    approval_request_id: str | None


class VerificationSummary(TypedDict):
    """Summary of the command used to verify a recent write."""

    command: str
    cwd: str
    ok: bool
    exit_code: int | None
    timed_out: bool
    truncated: bool
    stdout_preview: str | None
    stderr_preview: str | None


class RollbackSummary(TypedDict):
    """Result of restoring files from the per-run backup manifest."""

    ok: bool
    run_id: str
    manifest_path: str | None
    backup_root: str | None
    restored_files: list[str]
    removed_files: list[str]
    error: str | None
    trigger: Literal["manual", "verify_failure"]


PlanStepStatus = Literal["pending", "in_progress", "completed", "blocked", "skipped"]
ReflectionOutcome = Literal["continue", "replan", "retry", "pause", "stop"]
BudgetStopReason = Literal[
    "max_iterations",
    "max_command_count",
    "max_runtime_seconds",
    "max_plan_revisions",
    "max_recovery_attempts",
]


class PlanStep(TypedDict):
    """Structured plan step used by the phase-4 planning lifecycle."""

    id: str
    title: str
    status: PlanStepStatus
    rationale: str | None
    evidence_refs: list[int]


class ReflectionResult(TypedDict):
    """Structured reflector output for scoring and next-action guidance."""

    score: int
    outcome: ReflectionOutcome
    reason: str
    retryable: bool
    recommended_next_action: str | None


class RecoveryState(TypedDict):
    """Bounded recovery metadata for repeated failures."""

    issue_type: str
    fingerprint: str
    attempt_count: int
    last_action: str | None
    can_retry: bool


class BudgetStatus(TypedDict):
    """Mutable budget counters tracked across a single run."""

    iteration_count: int
    command_count: int
    elapsed_seconds: int
    warning_triggered: bool


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

    # ISO-8601 timestamp for when the run was first created.
    started_at: NotRequired[str | None]

    # Chat-style message history; add_messages appends rather than replaces
    messages: Annotated[list[BaseMessage], add_messages]

    # Ordered list of steps the Planner believes are needed to reach the goal
    plan: list[str]

    # Phase-4 structured planning metadata; defaults keep stage 1-3 compatible.
    command_count: NotRequired[int]
    plan_version: NotRequired[int]
    plan_revision_count: NotRequired[int]
    plan_steps: NotRequired[list[PlanStep]]
    last_reflection: NotRequired[ReflectionResult | None]
    recovery_state: NotRequired[RecoveryState | None]
    recovery_attempt_total: NotRequired[int]
    budget_status: NotRequired[BudgetStatus]
    budget_stop_reason: NotRequired[BudgetStopReason | None]

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

    # Optional reviewer note recorded with the latest approval response.
    approval_response_note: NotRequired[str | None]

    # Populated after a successful write until a verification command runs.
    pending_verification: WriteSummary | None

    # The latest successfully applied write, regardless of verification outcome.
    last_write: WriteSummary | None

    # Summary of the latest verification command run after a write.
    last_verification: VerificationSummary | None

    # Populated when a write is rolled back manually or after failed validation.
    last_rollback: RollbackSummary | None

    # Total number of tool executions so far (incremented by Tool Executor)
    iteration_count: int

    # Consecutive failure counter; reset to 0 on any successful observation
    consecutive_failures: int

    # Set by the Planner (task done) or the Reflector (circuit-breaker).
    # Non-None value signals the graph to route to the Finalizer.
    final_answer: str | None
