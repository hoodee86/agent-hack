"""
LangGraph state machine for the Linux Agent.

Nodes
-----
- planner      (T9):  LLM-powered; decides next tool call or signals completion.
- policy_guard (T10): Enforces the security policy; allows, denies, or pauses for approval.
- tool_executor(T11): Dispatches to skills; records an Observation.
- reflector    (T12): Pure-logic circuit-breaker; checks iteration/failure limits.
- finalizer    (T13): Emits the final answer and writes the run_end audit event.

Graph topology
--------------
        START → planner ──► policy_guard ──► tool_executor ──► reflector ──┐
            │          │                          ▲                           │ (loop)
            │          └──────────────► finalizer │                           │
            │                                     │                           │
            └────────► resume_gate ───────────────┘                           │
                                                                │                                       │
                                                                └──────────────────────► finalizer      │
                             └────────────────────────────────────────────────────────┘

    planner      → finalizer   when final_answer is set
        policy_guard → approval_pause when risk_decision is "needs_approval"
        policy_guard → finalizer   when risk_decision is "deny"
    reflector    → finalizer   when circuit-breaker trips
        approval_pause → finalizer after persisting the pending approval snapshot
    finalizer    → END
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
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

from linux_agent.approval_ui import build_approval_view
from linux_agent.audit import (
    AuditEventListener,
    EVENT_APPROVAL_PRESENTED,
    EVENT_APPROVAL_REQUESTED,
    EVENT_APPROVAL_RESPONSE,
    EVENT_BUDGET_EXHAUSTED,
    EVENT_BUDGET_WARNING,
    EVENT_MODEL_INPUT,
    EVENT_PLAN_UPDATE,
    EVENT_PLAN_REVISED,
    EVENT_POLICY_DECISION,
    EVENT_RECOVERY_ATTEMPTED,
    EVENT_RECOVERY_CLEARED,
    EVENT_RECOVERY_EXHAUSTED,
    EVENT_REFLECTION_SCORED,
    EVENT_REFLECTOR_ACTION,
    EVENT_RUN_END,
    EVENT_TOOL_PROPOSED,
    EVENT_TOOL_RESULT,
    EVENT_WRITE_APPLIED,
    EVENT_WRITE_ROLLBACK,
    AuditLogger,
)
from linux_agent.config import AgentConfig
from linux_agent.policy import (
    PolicyViolation,
    assess_tool_call,
    classify_command,
    parse_command,
)
from linux_agent.run_store import delete_run_state, save_run_state
from linux_agent.skills.filesystem import list_dir, read_file
from linux_agent.skills.search import search_text
from linux_agent.skills.shell import run_command
from linux_agent.skills.write import apply_patch, rollback_run, write_file
from linux_agent.state import (
    AgentState,
    Observation,
    RollbackSummary,
    ToolCall,
    VerificationSummary,
    WriteSummary,
)


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


@tool("run_command")
def run_command_tool(
    command: str,
    cwd: str = ".",
    timeout_seconds: int | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a safe developer command inside the workspace, such as tests, lint, type checks, or git status."""
    raise RuntimeError("run_command is executed by linux_agent.graph, not by LangChain.")


@tool("apply_patch")
def apply_patch_tool(patch: str) -> str:
    """Apply a constrained patch payload to workspace text files. This tool always requires explicit approval."""
    raise RuntimeError("apply_patch is executed by linux_agent.graph, not by LangChain.")


@tool("write_file")
def write_file_tool(path: str, content: str, mode: str = "overwrite") -> str:
    """Write text content to a workspace file. Modes: overwrite, create_only, append. This tool always requires explicit approval."""
    raise RuntimeError("write_file is executed by linux_agent.graph, not by LangChain.")


_MODEL_TOOLS = [
    list_dir_tool,
    read_file_tool,
    search_text_tool,
    run_command_tool,
    apply_patch_tool,
    write_file_tool,
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a controlled Linux workspace agent.
Your task is to answer the user's goal by inspecting the workspace and,
when useful, running tightly constrained developer commands and proposing
bounded text-only file changes that require explicit approval before execution.

## Available tools

| Tool        | Required args                   | Optional args                                          |
|-------------|---------------------------------|--------------------------------------------------------|
| list_dir    | path: str (workspace-relative)  | recursive: bool (default false)                        |
| read_file   | path: str (workspace-relative)  | start_line: int (1-based), end_line: int               |
| search_text | query: str                      | path: str (default "."), glob: str, context_lines: int |
| run_command | command: str                    | cwd: str (default "."), timeout_seconds: int, env: dict |
| apply_patch | patch: str                      | none                                                   |
| write_file  | path: str, content: str         | mode: str (overwrite/create_only/append)               |

All paths and working directories must stay within the workspace root.

## Rules

- Use list_dir, read_file, and search_text to inspect files and source code.
- Use run_command only for safe developer commands such as tests, lint, type checks,
    build diagnostics, or git status/diff.
- Use apply_patch and write_file only when a file change is necessary to complete the goal.
- Keep write requests narrow, text-only, and limited to the workspace.
- Before proposing a write, gather enough evidence from read_file/search_text/list_dir to justify the exact change.
- Never request shell control syntax such as pipes, redirection, chaining, background
    execution, or shell wrappers.
- Call at most one tool at a time.
- If a write tool is required, explain the intended change clearly so the approval summary is useful.
- After any successful write, you MUST call run_command with a narrow validation command before giving a final answer.
- If validation fails, inspect the failure output and either propose another narrow fix or explain the rollback path; never claim success.
- After a command fails, inspect the referenced files or error locations before retrying.
- Use the planner prompt's Last reflection and Recovery state sections to decide whether to continue,
    replan, take one bounded recovery step, or stop.
- If the last reflection says outcome=retry, take the recommended recovery action instead of repeating the
    same failing command or path access.
- If the last reflection says outcome=replan, update the plan before choosing the next tool call.
- If the last reflection says outcome=stop, do not call another tool; provide a concise final answer.
- Answer in plain text when you already have enough information.
- Do not invent files, directories, file contents, command outputs, or test results.
- Keep tool arguments minimal and precise.
"""

_WRITE_TOOL_NAMES: frozenset[str] = frozenset({"apply_patch", "write_file"})
_VALIDATION_COMMAND_KEYWORDS: tuple[str, ...] = (
    "pytest",
    "mypy",
    "ruff",
    "test",
    "check",
    "build",
    "compile",
)
_NON_VALIDATING_COMMAND_PREFIXES: tuple[str, ...] = ("git", "ls", "pwd", "find", "rg")
_RECOVERY_LOCATION_RE = re.compile(r"([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)(?::(\d+))?")


def _fmt_observations(observations: list[Observation], n: int = 5) -> str:
    """Format the last *n* observations for inclusion in the Planner prompt."""
    if not observations:
        return "None yet."
    recent = observations[-n:]
    start_idx = len(observations) - len(recent) + 1
    parts: list[str] = []
    for i, obs in enumerate(recent, start=start_idx):
        status = "OK" if obs["ok"] else "ERROR"
        detail = _format_observation_detail(obs)
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


def _audit_with_legacy_reflector(
    run_id: str,
    log_dir: Any,
    event: str,
    data: dict[str, Any],
    listener: AuditEventListener | None = None,
    *,
    legacy_reason: str | None = None,
) -> None:
    _one_shot_audit(run_id, log_dir, event, data, listener)
    if legacy_reason is None:
        return
    legacy_payload = dict(data)
    legacy_payload.setdefault("reason", legacy_reason)
    _one_shot_audit(run_id, log_dir, EVENT_REFLECTOR_ACTION, legacy_payload, listener)


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


def _preview_text(value: Any, *, limit: int = 320) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    clipped = "\n".join(line.rstrip() for line in text.splitlines()[:12]).strip()
    if len(clipped) > limit:
        return clipped[:limit] + " …"
    return clipped


def _command_follow_up_hint(result: dict[str, Any]) -> str:
    command = str(result.get("command", "")).lower()
    if "pytest" in command:
        return (
            "Read the failing test or source files mentioned in the output before "
            "retrying the test command."
        )
    if "mypy" in command or "ruff" in command:
        return (
            "Inspect the reported file and line range with read_file before rerunning "
            "the command."
        )
    return (
        "Use read_file or search_text to inspect the files or errors referenced by "
        "the command output before retrying."
    )


def _format_observation_detail(obs: Observation) -> str:
    result = obs.get("result")
    if isinstance(result, dict):
        if obs["tool"] == "run_command":
            lines = [
                f"command={result.get('command', '(unknown)')}  cwd={result.get('cwd', '.')}",
                (
                    f"exit_code={result.get('exit_code')}  "
                    f"timed_out={bool(result.get('timed_out', False))}  "
                    f"truncated={bool(result.get('truncated', False))}"
                ),
            ]
            stderr_preview = _preview_text(result.get("stderr"), limit=500)
            stdout_preview = _preview_text(result.get("stdout"), limit=500)
            if stderr_preview is not None:
                lines.append(f"stderr:\n{stderr_preview}")
            if stdout_preview is not None:
                lines.append(f"stdout:\n{stdout_preview}")
            if result.get("timed_out") or result.get("truncated") or not obs["ok"]:
                lines.append(f"follow_up_hint: {_command_follow_up_hint(result)}")
            return "\n".join(lines)

        raw = json.dumps(result, ensure_ascii=False)
        return raw[:600] + " …" if len(raw) > 600 else raw

    return obs["error"] or ""


def _command_fields_from_args(args: dict[str, object]) -> dict[str, Any]:
    payload: dict[str, Any] = {"cwd": str(args.get("cwd", "."))}
    raw_command = args.get("command")
    if isinstance(raw_command, str) and raw_command.strip():
        payload["command"] = raw_command.strip()
        try:
            payload["argv"] = parse_command(raw_command)
        except PolicyViolation:
            pass
    timeout_seconds = args.get("timeout_seconds")
    if timeout_seconds is not None:
        payload["timeout_seconds"] = timeout_seconds
    raw_env = args.get("env")
    if isinstance(raw_env, dict):
        payload["env_keys"] = sorted(str(key) for key in raw_env)
    return payload


def _command_fields_from_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    payload: dict[str, Any] = {}
    for key in ("command", "argv", "cwd", "exit_code", "timed_out", "truncated"):
        if key in result and result.get(key) is not None:
            payload[key] = result.get(key)

    stdout_preview = _preview_text(result.get("stdout"), limit=600)
    stderr_preview = _preview_text(result.get("stderr"), limit=600)
    if stdout_preview is not None:
        payload["stdout_preview"] = stdout_preview
    if stderr_preview is not None:
        payload["stderr_preview"] = stderr_preview
    return payload


def _write_fields_from_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    payload: dict[str, Any] = {}
    for key in (
        "path",
        "mode",
        "changed_files",
        "added_lines",
        "removed_lines",
        "backup_paths",
        "backup_root",
        "manifest_path",
        "created",
        "rolled_back",
        "restored_files",
        "removed_files",
    ):
        if key in result and result.get(key) is not None:
            payload[key] = result.get(key)

    diff_preview = _preview_text(result.get("diff"), limit=800)
    if diff_preview is not None:
        payload["diff_preview"] = diff_preview
    return payload


def _tool_call_audit_payload(tool_call: ToolCall) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool": tool_call["name"],
        "tool_call_id": tool_call["id"],
        "risk_level": tool_call["risk_level"],
        "args": tool_call["args"],
    }
    if tool_call["name"] == "run_command":
        payload.update(_command_fields_from_args(tool_call["args"]))
    if tool_call["name"] == "apply_patch":
        payload["diff_preview"] = _preview_text(
            tool_call["args"].get("patch", tool_call["args"].get("diff"))
        )
    if tool_call["name"] == "write_file":
        payload["path"] = tool_call["args"].get("path")
        payload["mode"] = tool_call["args"].get("mode", "overwrite")
        payload["diff_preview"] = _preview_text(tool_call["args"].get("content"))
    return payload


def _tool_result_audit_payload(tool_call: ToolCall, obs: Observation) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool": obs["tool"],
        "tool_call_id": tool_call["id"],
        "risk_level": tool_call["risk_level"],
        "ok": obs["ok"],
        "duration_ms": obs["duration_ms"],
        "error": obs["error"],
        "result": obs["result"],
    }
    if obs["tool"] == "run_command":
        payload.update(_command_fields_from_result(obs["result"]))
    if obs["tool"] in {"apply_patch", "write_file"}:
        payload.update(_write_fields_from_result(cast(dict[str, Any] | None, obs["result"])))
    return payload


def _build_write_summary(
    tool_name: str,
    result: dict[str, Any],
    *,
    approval_request_id: str | None,
) -> WriteSummary:
    raw_changed_files = result.get("changed_files")
    changed_files = (
        [str(path) for path in raw_changed_files]
        if isinstance(raw_changed_files, list)
        else []
    )
    return {
        "tool": cast("Literal['apply_patch', 'write_file']", tool_name),
        "changed_files": changed_files,
        "added_lines": int(result.get("added_lines", 0) or 0),
        "removed_lines": int(result.get("removed_lines", 0) or 0),
        "backup_root": cast(str | None, result.get("backup_root")),
        "manifest_path": cast(str | None, result.get("manifest_path")),
        "approval_request_id": approval_request_id,
    }


def _build_verification_summary(result: dict[str, Any], *, ok: bool) -> VerificationSummary:
    return {
        "command": str(result.get("command", "(unknown)")),
        "cwd": str(result.get("cwd", ".")),
        "ok": ok,
        "exit_code": cast(int | None, result.get("exit_code")),
        "timed_out": bool(result.get("timed_out", False)),
        "truncated": bool(result.get("truncated", False)),
        "stdout_preview": _preview_text(result.get("stdout"), limit=400),
        "stderr_preview": _preview_text(result.get("stderr"), limit=400),
    }


def _build_rollback_summary(
    rollback_result: dict[str, Any],
    *,
    trigger: str,
) -> RollbackSummary:
    raw_restored = rollback_result.get("restored_files")
    raw_removed = rollback_result.get("removed_files")
    restored_files = [str(path) for path in raw_restored] if isinstance(raw_restored, list) else []
    removed_files = [str(path) for path in raw_removed] if isinstance(raw_removed, list) else []
    return {
        "ok": bool(rollback_result.get("ok", False)),
        "run_id": str(rollback_result.get("run_id", "")),
        "manifest_path": cast(str | None, rollback_result.get("manifest_path")),
        "backup_root": cast(str | None, rollback_result.get("backup_root")),
        "restored_files": restored_files,
        "removed_files": removed_files,
        "error": cast(str | None, rollback_result.get("error")),
        "trigger": cast("Literal['manual', 'verify_failure']", trigger),
    }


def _is_validation_command(result: dict[str, Any]) -> bool:
    raw_command = result.get("command")
    if not isinstance(raw_command, str) or not raw_command.strip():
        return False

    try:
        argv = [part.lower() for part in parse_command(raw_command)]
    except PolicyViolation:
        argv = [part.lower() for part in raw_command.split() if part.strip()]

    if not argv:
        return False

    prefix = argv[0]
    if prefix in _NON_VALIDATING_COMMAND_PREFIXES:
        return False

    joined = " ".join(argv)
    return any(keyword in joined for keyword in _VALIDATION_COMMAND_KEYWORDS)


def _format_pending_verification(state: AgentState) -> str:
    pending = state.get("pending_verification")
    if pending is not None:
        changed = ", ".join(pending["changed_files"][:5]) or "(unknown files)"
        lines = [
            "A recent write was applied but has not been verified yet.",
            f"Changed files: {changed}",
            f"Added/removed lines: +{pending['added_lines']}/-{pending['removed_lines']}",
            "You MUST call run_command next with a narrow validation command such as tests, lint, type checks, or build verification.",
            "Do not answer in plain text until validation finishes.",
        ]
        if pending.get("backup_root"):
            lines.append(f"Backup root: {pending['backup_root']}")
        if pending.get("manifest_path"):
            lines.append(f"Manifest path: {pending['manifest_path']}")
        return "\n".join(lines)

    last_verification = state.get("last_verification")
    last_write = state.get("last_write")
    last_rollback = state.get("last_rollback")
    if last_verification is not None and last_write is not None:
        changed = ", ".join(last_write["changed_files"][:5]) or "(unknown files)"
        lines = [f"Latest write touched: {changed}."]
        if last_verification["ok"]:
            lines.append(
                f"Latest validation passed via '{last_verification['command']}' (exit {last_verification['exit_code']})."
            )
            return "\n".join(lines)

        lines.append(
            f"Latest validation failed via '{last_verification['command']}' (exit {last_verification['exit_code']})."
        )
        if last_rollback is not None and last_rollback.get("ok"):
            lines.append("The failed write has already been rolled back.")
        else:
            lines.append(
                "Inspect the failure output and either propose another narrow fix or explicitly explain the rollback path before claiming success."
            )
            if last_write.get("manifest_path"):
                lines.append(f"Rollback manifest: {last_write['manifest_path']}")
        return "\n".join(lines)

    return "No pending write verification."


def _verification_retry_message(write_summary: WriteSummary) -> str:
    changed = ", ".join(write_summary["changed_files"][:5]) or "(unknown files)"
    return (
        "Recent file changes are still unverified.\n"
        f"Changed files: {changed}\n"
        "Call run_command next with a narrow validation command such as tests, lint, type checks, or build verification.\n"
        "Do not answer in plain text until validation completes."
    )


def _build_unverified_write_answer(write_summary: WriteSummary) -> str:
    changed = ", ".join(write_summary["changed_files"][:5]) or "(unknown files)"
    lines = [
        "Agent stopped before validating the latest file changes.",
        f"Changed files: {changed}",
        "Run a validation command before treating this write as complete.",
    ]
    if write_summary.get("backup_root"):
        lines.append(f"Backup root: {write_summary['backup_root']}")
    if write_summary.get("manifest_path"):
        lines.append(f"Manifest path: {write_summary['manifest_path']}")
    return "\n".join(lines)


def _build_validation_failure_answer(
    write_summary: WriteSummary,
    verification: VerificationSummary,
    rollback_summary: RollbackSummary | None,
) -> str:
    changed = ", ".join(write_summary["changed_files"][:5]) or "(unknown files)"
    lines = [
        "Validation failed after applying file changes.",
        f"Changed files: {changed}",
        f"Validation command: {verification['command']} [cwd={verification['cwd']}]",
    ]
    if verification["exit_code"] is not None:
        lines.append(f"Exit code: {verification['exit_code']}")
    if verification.get("stderr_preview"):
        lines.append(f"stderr preview: {verification['stderr_preview']}")
    elif verification.get("stdout_preview"):
        lines.append(f"stdout preview: {verification['stdout_preview']}")

    if rollback_summary is not None and rollback_summary["ok"]:
        lines.append("The write was rolled back automatically after the failed validation command.")
        if rollback_summary["restored_files"]:
            lines.append(
                "Restored files: " + ", ".join(rollback_summary["restored_files"])
            )
        if rollback_summary["removed_files"]:
            lines.append(
                "Removed files: " + ", ".join(rollback_summary["removed_files"])
            )
    else:
        lines.append("Rollback is still available if you want to restore the previous state.")
        if write_summary.get("manifest_path"):
            lines.append(f"Rollback manifest: {write_summary['manifest_path']}")
    return "\n".join(lines)


def _classify_tool_risk(
    name: str,
    args: dict[str, object],
    config: AgentConfig,
) -> str:
    if name in {"apply_patch", "write_file"}:
        return "high"
    if name != "run_command":
        return "low"

    raw_command = args.get("command")
    if not isinstance(raw_command, str):
        return "high"

    try:
        argv = parse_command(raw_command)
    except PolicyViolation:
        return "high"
    return classify_command(argv, config)


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
    plan_steps = _effective_plan_steps(state)
    plan_version = _effective_plan_version(state, plan_steps=plan_steps)
    plan_revision_count = _effective_plan_revision_count(state)
    recovery_attempt_count = _current_recovery_attempt_count(state)
    budget_status = _budget_status_snapshot(state)
    remaining_budget = _budget_remaining(
        config,
        budget_status,
        plan_revision_count=plan_revision_count,
        recovery_attempt_count=recovery_attempt_count,
    )
    obs_text = _fmt_observations(state["observations"])
    write_text = _format_pending_verification(state)
    plan_text = _format_structured_plan(plan_steps)
    last_reflection = _format_last_reflection(state)
    recovery_text = _format_recovery_state(state)
    budget_text = _format_budget_for_prompt(
        config,
        budget_status,
        remaining_budget,
        plan_revision_count=plan_revision_count,
        recovery_attempt_count=recovery_attempt_count,
    )
    return (
        f"Goal: {state['user_goal']}\n\n"
        f"Workspace root: {state['workspace_root']}\n\n"
        f"Current plan (version {plan_version}, revisions used {plan_revision_count}/{config.max_plan_revisions}):\n"
        f"{plan_text}\n\n"
        f"Recent observations:\n{obs_text}\n\n"
        f"Last reflection:\n{last_reflection}\n\n"
        f"Recovery state:\n{recovery_text}\n\n"
        f"Write verification status:\n{write_text}\n\n"
        f"Remaining budget:\n{budget_text}\n\n"
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


def _build_tool_call(
    message: AIMessage,
    state: AgentState,
    config: AgentConfig,
) -> ToolCall | None:
    tool_calls = message.tool_calls or []
    if not tool_calls:
        return None

    raw_call = tool_calls[0]
    raw_name = raw_call.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None

    raw_id = raw_call.get("id")
    call_id = str(raw_id) if raw_id else f"{raw_name}_{state['iteration_count'] + 1}"
    args = _coerce_tool_args(raw_call.get("args", {}))
    return ToolCall(
        id=call_id,
        name=raw_name,
        args=args,
        risk_level=cast(str, _classify_tool_risk(raw_name, args, config)),
    )


def _tool_step(tool_call: ToolCall) -> str:
    if tool_call["name"] == "run_command":
        command = str(tool_call["args"].get("command", ""))
        cwd = str(tool_call["args"].get("cwd", "."))
        return f"Running command {command} (cwd={cwd})"
    args_json = json.dumps(tool_call["args"], ensure_ascii=False)
    return f"Calling {tool_call['name']} {args_json}"


def _command_reflection_payload(obs: Observation) -> dict[str, Any] | None:
    result = obs.get("result")
    if obs["tool"] != "run_command" or not isinstance(result, dict):
        return None

    payload = _command_fields_from_result(result)
    payload["tool"] = obs["tool"]
    if result.get("timed_out"):
        payload.update(
            {
                "reason": "command_timeout",
                "action": "stop_after_timeout",
                "guidance": (
                    "The command timed out. Report the timeout and suggest a narrower "
                    "command or direct file inspection instead of blindly retrying."
                ),
            }
        )
        return payload

    if result.get("truncated"):
        payload.update(
            {
                "reason": "command_output_truncated",
                "action": "narrow_scope",
                "guidance": (
                    "The command output was truncated. Narrow the command scope or "
                    "inspect the referenced files directly before continuing."
                ),
            }
        )
        if obs["ok"]:
            return payload

    if not obs["ok"]:
        payload.update(
            {
                "reason": "command_failed",
                "action": "inspect_failure_context",
                "guidance": _command_follow_up_hint(result),
            }
        )
        return payload

    return payload if payload.get("reason") else None


def _build_command_timeout_answer(obs: Observation) -> str:
    result = cast(dict[str, Any], obs["result"])
    lines = [
        "Agent stopped after a command timed out.",
        f"Command: {result.get('command', '(unknown)')}",
        f"Working directory: {result.get('cwd', '.')}",
    ]
    stdout_preview = _preview_text(result.get("stdout"), limit=400)
    stderr_preview = _preview_text(result.get("stderr"), limit=400)
    if stderr_preview is not None:
        lines.append(f"stderr preview: {stderr_preview}")
    if stdout_preview is not None:
        lines.append(f"stdout preview: {stdout_preview}")
    lines.append(
        "Suggested next step: rerun a narrower command or inspect the referenced "
        "files directly with read_file or search_text."
    )
    return "\n".join(lines)


def _command_summary_lines(observations: list[Observation]) -> list[str]:
    lines: list[str] = []
    for index, obs in enumerate(observations, start=1):
        result = obs.get("result")
        if obs["tool"] != "run_command" or not isinstance(result, dict):
            continue

        command = str(result.get("command", "(unknown)"))
        cwd = str(result.get("cwd", "."))
        exit_code = result.get("exit_code")
        if result.get("timed_out"):
            status = "timed out"
        elif obs["ok"]:
            status = f"ok (exit {exit_code})"
        elif exit_code is not None:
            status = f"failed (exit {exit_code})"
        else:
            status = "failed"

        evidence = _preview_text(result.get("stderr"), limit=180) or _preview_text(
            result.get("stdout"),
            limit=180,
        )
        line = f"{len(lines) + 1}. {command} [cwd={cwd}] -> {status}"
        if evidence is not None:
            line += f" | evidence: {evidence.splitlines()[0]}"
        lines.append(line)
    return lines


def _write_summary_lines(observations: list[Observation]) -> list[str]:
    lines: list[str] = []
    for obs in observations:
        result = obs.get("result")
        if obs["tool"] not in {"apply_patch", "write_file"} or not isinstance(result, dict):
            continue

        changed_files = result.get("changed_files")
        if isinstance(changed_files, list) and changed_files:
            target_summary = ", ".join(str(path) for path in changed_files[:3])
            if len(changed_files) > 3:
                target_summary += f" (+{len(changed_files) - 3} more)"
        else:
            target_summary = str(result.get("path", "(unknown)"))

        added_lines = int(result.get("added_lines", 0) or 0)
        removed_lines = int(result.get("removed_lines", 0) or 0)
        status = "ok" if obs["ok"] else "failed"
        line = (
            f"{len(lines) + 1}. {obs['tool']} -> {status} | files: {target_summary} | "
            f"+{added_lines}/-{removed_lines}"
        )
        if result.get("rolled_back"):
            line += " | rollback applied"
        elif result.get("backup_paths"):
            line += " | backup created"
        lines.append(line)
    return lines


def _summarize_step(message: AIMessage, tool_call: ToolCall | None) -> str:
    content = _content_to_text(message.content)
    if content:
        first_line = content.splitlines()[0].strip()
        if first_line:
            return first_line[:240]
    if tool_call is not None:
        return _tool_step(tool_call)
    return "Preparing final answer"


def _parse_started_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ensure_started_at(state: AgentState) -> str:
    parsed = _parse_started_at(state.get("started_at"))
    if parsed is None:
        return datetime.now(tz=timezone.utc).isoformat()
    return parsed.isoformat()


def _elapsed_seconds(started_at: str | None) -> int:
    parsed = _parse_started_at(started_at)
    if parsed is None:
        return 0
    return max(0, int((datetime.now(tz=timezone.utc) - parsed).total_seconds()))


def _default_plan_steps(plan: list[str], current_step: str | None) -> list[dict[str, Any]]:
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


def _effective_plan_steps(state: AgentState) -> list[dict[str, Any]]:
    raw_steps = state.get("plan_steps")
    if isinstance(raw_steps, list):
        normalized_steps: list[dict[str, Any]] = []
        for index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, dict):
                continue
            title = str(raw_step.get("title", "")).strip()
            if not title:
                continue
            raw_status = str(raw_step.get("status", "pending")).strip()
            status = (
                raw_status
                if raw_status in {"pending", "in_progress", "completed", "blocked", "skipped"}
                else "pending"
            )
            evidence_refs = raw_step.get("evidence_refs")
            normalized_steps.append(
                {
                    "id": str(raw_step.get("id") or f"step_{index}"),
                    "title": title,
                    "status": status,
                    "rationale": cast(str | None, raw_step.get("rationale")),
                    "evidence_refs": (
                        [int(value) for value in evidence_refs if isinstance(value, int)]
                        if isinstance(evidence_refs, list)
                        else []
                    ),
                }
            )
        if normalized_steps or not state["plan"]:
            return normalized_steps
    return _default_plan_steps(state["plan"], state.get("current_step"))


def _effective_plan_version(state: AgentState, *, plan_steps: list[dict[str, Any]] | None = None) -> int:
    raw_version = state.get("plan_version")
    if isinstance(raw_version, int) and raw_version >= 0:
        return raw_version
    steps = plan_steps if plan_steps is not None else _effective_plan_steps(state)
    return 1 if steps else 0


def _effective_plan_revision_count(state: AgentState) -> int:
    raw_count = state.get("plan_revision_count")
    if isinstance(raw_count, int) and raw_count >= 0:
        return raw_count
    return 0


def _effective_command_count(state: AgentState) -> int:
    raw_count = state.get("command_count")
    if isinstance(raw_count, int) and raw_count >= 0:
        return raw_count
    return sum(1 for obs in state["observations"] if obs["tool"] == "run_command")


def _current_recovery_attempt_count(state: AgentState) -> int:
    recovery_state = state.get("recovery_state")
    if not isinstance(recovery_state, dict):
        return 0
    raw_attempts = recovery_state.get("attempt_count")
    if isinstance(raw_attempts, int) and raw_attempts >= 0:
        return raw_attempts
    return 0


def _effective_recovery_attempt_total(state: AgentState) -> int:
    raw_total = state.get("recovery_attempt_total")
    if isinstance(raw_total, int) and raw_total >= 0:
        return raw_total
    return 0


def _budget_status_snapshot(
    state: AgentState,
    *,
    started_at: str | None = None,
    iteration_count: int | None = None,
    command_count: int | None = None,
    warning_triggered: bool | None = None,
) -> dict[str, Any]:
    raw_budget = state.get("budget_status")
    warning_value = False
    if isinstance(raw_budget, dict):
        warning_value = bool(raw_budget.get("warning_triggered", False))
    if warning_triggered is not None:
        warning_value = warning_triggered

    effective_started_at = started_at if started_at is not None else cast(str | None, state.get("started_at"))
    return {
        "iteration_count": (
            iteration_count if iteration_count is not None else int(state.get("iteration_count", 0))
        ),
        "command_count": (
            command_count if command_count is not None else _effective_command_count(state)
        ),
        "elapsed_seconds": _elapsed_seconds(effective_started_at),
        "warning_triggered": warning_value,
    }


def _budget_remaining(
    config: AgentConfig,
    budget_status: dict[str, Any],
    *,
    plan_revision_count: int,
    recovery_attempt_count: int,
) -> dict[str, Any]:
    return {
        "iterations_remaining": max(0, config.max_iterations - int(budget_status["iteration_count"])),
        "commands_remaining": max(0, config.max_command_count - int(budget_status["command_count"])),
        "runtime_remaining_seconds": max(0, config.max_runtime_seconds - int(budget_status["elapsed_seconds"])),
        "plan_revisions_remaining": max(0, config.max_plan_revisions - plan_revision_count),
        "recovery_attempts_remaining": max(
            0,
            config.max_recovery_attempts_per_issue - recovery_attempt_count,
        ),
    }


def _format_structured_plan(plan_steps: list[dict[str, Any]]) -> str:
    if not plan_steps:
        return "  (not yet planned)"

    lines: list[str] = []
    for index, step in enumerate(plan_steps, start=1):
        line = f"  {index}. [{step['status']}] {step['title']}"
        rationale = cast(str | None, step.get("rationale"))
        if rationale:
            line += f" — {rationale}"
        lines.append(line)
    return "\n".join(lines)


def _format_last_reflection(state: AgentState) -> str:
    reflection = state.get("last_reflection")
    if not isinstance(reflection, dict):
        return "None yet."
    lines = [
        f"score={reflection.get('score', '(unknown)')}",
        f"outcome={reflection.get('outcome', '(unknown)')}",
        f"reason={reflection.get('reason', '(none)')}",
    ]
    next_action = reflection.get("recommended_next_action")
    if next_action:
        lines.append(f"recommended_next_action={next_action}")
    return "\n".join(lines)


def _format_recovery_state(state: AgentState) -> str:
    recovery_state = state.get("recovery_state")
    if not isinstance(recovery_state, dict):
        return "Not currently recovering from a repeated issue."
    return "\n".join(
        [
            f"issue_type={recovery_state.get('issue_type', '(unknown)')}",
            f"fingerprint={recovery_state.get('fingerprint', '(unknown)')}",
            f"attempt_count={recovery_state.get('attempt_count', 0)}",
            f"last_action={recovery_state.get('last_action', '(none)')}",
            f"can_retry={bool(recovery_state.get('can_retry', False))}",
        ]
    )


def _format_budget_for_prompt(
    config: AgentConfig,
    budget_status: dict[str, Any],
    remaining_budget: dict[str, Any],
    *,
    plan_revision_count: int,
    recovery_attempt_count: int,
) -> str:
    return "\n".join(
        [
            (
                f"- iterations: {budget_status['iteration_count']} / {config.max_iterations} "
                f"(remaining {remaining_budget['iterations_remaining']})"
            ),
            (
                f"- commands: {budget_status['command_count']} / {config.max_command_count} "
                f"(remaining {remaining_budget['commands_remaining']})"
            ),
            (
                f"- runtime: {budget_status['elapsed_seconds']} / {config.max_runtime_seconds}s "
                f"(remaining {remaining_budget['runtime_remaining_seconds']}s)"
            ),
            (
                f"- plan revisions: {plan_revision_count} / {config.max_plan_revisions} "
                f"(remaining {remaining_budget['plan_revisions_remaining']})"
            ),
            (
                f"- recovery attempts for current issue: {recovery_attempt_count} / "
                f"{config.max_recovery_attempts_per_issue} "
                f"(remaining {remaining_budget['recovery_attempts_remaining']})"
            ),
        ]
    )


def _find_plan_step_index(plan_steps: list[dict[str, Any]], title: str | None) -> int | None:
    if title is None:
        return None
    for index, step in enumerate(plan_steps):
        if step["title"] == title:
            return index
    return None


def _plan_evidence_refs(state: AgentState) -> list[int]:
    if not state["observations"]:
        return []
    return [len(state["observations"])]


def _plan_step_rationale(
    next_step: str,
    assistant_content: str,
    last_obs: Observation | None,
) -> str | None:
    first_line = assistant_content.splitlines()[0].strip() if assistant_content.strip() else ""
    if first_line and first_line != next_step:
        return first_line[:240]
    if last_obs is not None and not last_obs["ok"]:
        return f"Follow-up after {last_obs['tool']} failed."
    if last_obs is not None and last_obs["tool"] == "run_command":
        return f"Follow-up after {last_obs['tool']} output."
    return None


def _derive_plan_update(
    state: AgentState,
    next_step: str | None,
    *,
    assistant_content: str,
    final_answer: str | None,
) -> dict[str, Any]:
    plan_steps = [dict(step) for step in _effective_plan_steps(state)]
    plan_version = _effective_plan_version(state, plan_steps=plan_steps)
    plan_revision_count = _effective_plan_revision_count(state)
    revision_reason: str | None = None
    last_obs = state["observations"][-1] if state["observations"] else None
    previous_step = cast(str | None, state.get("current_step"))
    previous_idx = _find_plan_step_index(plan_steps, previous_step)
    if previous_idx is None:
        previous_idx = next(
            (index for index, step in enumerate(plan_steps) if step["status"] == "in_progress"),
            None,
        )

    if previous_idx is not None and (final_answer is not None or next_step != previous_step):
        previous_status = "completed"
        if last_obs is not None and not last_obs["ok"]:
            previous_status = "blocked"
        plan_steps[previous_idx] = {
            **plan_steps[previous_idx],
            "status": previous_status,
        }

    current_idx: int | None = None
    if next_step is not None:
        current_idx = _find_plan_step_index(plan_steps, next_step)
        if current_idx is None:
            plan_steps.append(
                {
                    "id": f"step_{len(plan_steps) + 1}",
                    "title": next_step,
                    "status": "in_progress",
                    "rationale": _plan_step_rationale(next_step, assistant_content, last_obs),
                    "evidence_refs": _plan_evidence_refs(state),
                }
            )
            current_idx = len(plan_steps) - 1
            if len(plan_steps) == 1:
                plan_version = max(plan_version, 1)
                revision_reason = "initial_plan_created"
            else:
                plan_version = max(plan_version, 1) + 1
                plan_revision_count += 1
                revision_reason = (
                    "appended_step_after_failed_observation"
                    if last_obs is not None and not last_obs["ok"]
                    else "appended_step"
                )
        else:
            plan_steps[current_idx] = {
                **plan_steps[current_idx],
                "status": "in_progress",
            }

    if current_idx is not None:
        for index, step in enumerate(plan_steps):
            if index == current_idx:
                continue
            if step["status"] == "in_progress":
                fallback_status = "blocked" if last_obs is not None and not last_obs["ok"] else "completed"
                plan_steps[index] = {**step, "status": fallback_status}

    return {
        "plan": [str(step["title"]) for step in plan_steps],
        "current_step": next_step,
        "plan_steps": plan_steps,
        "plan_version": plan_version,
        "plan_revision_count": plan_revision_count,
        "plan_revision_reason": revision_reason,
    }


def _budget_warning_dimensions(
    config: AgentConfig,
    budget_status: dict[str, Any],
    *,
    plan_revision_count: int,
    recovery_attempt_count: int,
) -> list[str]:
    dimensions: list[str] = []
    ratio = config.budget_warning_ratio
    if config.max_iterations > 0 and budget_status["iteration_count"] / config.max_iterations >= ratio:
        dimensions.append("max_iterations")
    if config.max_command_count > 0 and budget_status["command_count"] / config.max_command_count >= ratio:
        dimensions.append("max_command_count")
    if config.max_runtime_seconds > 0 and budget_status["elapsed_seconds"] / config.max_runtime_seconds >= ratio:
        dimensions.append("max_runtime_seconds")
    if config.max_plan_revisions > 0 and plan_revision_count / config.max_plan_revisions >= ratio:
        dimensions.append("max_plan_revisions")
    if (
        config.max_recovery_attempts_per_issue > 0
        and recovery_attempt_count / config.max_recovery_attempts_per_issue >= ratio
    ):
        dimensions.append("max_recovery_attempts")
    return dimensions


def _build_budget_stop_answer(
    reason: str,
    state: AgentState,
    config: AgentConfig,
    budget_status: dict[str, Any],
    *,
    plan_steps: list[dict[str, Any]] | None = None,
    current_step: str | None = None,
    plan_revision_count: int | None = None,
    recovery_attempt_count: int | None = None,
) -> str:
    effective_plan_steps = plan_steps if plan_steps is not None else _effective_plan_steps(state)
    effective_current_step = current_step if current_step is not None else cast(str | None, state.get("current_step"))
    effective_plan_revision_count = (
        plan_revision_count if plan_revision_count is not None else _effective_plan_revision_count(state)
    )
    effective_recovery_attempt_count = (
        recovery_attempt_count
        if recovery_attempt_count is not None
        else _current_recovery_attempt_count(state)
    )
    completed_steps = [step["title"] for step in effective_plan_steps if step["status"] == "completed"]
    blocked_steps = [step["title"] for step in effective_plan_steps if step["status"] == "blocked"]
    reason_line_map = {
        "max_iterations": (
            f"Agent stopped: reached the maximum iteration budget "
            f"({budget_status['iteration_count']} / {config.max_iterations})."
        ),
        "max_command_count": (
            f"Agent stopped: command budget exhausted "
            f"({budget_status['command_count']} / {config.max_command_count})."
        ),
        "max_runtime_seconds": (
            f"Agent stopped: runtime budget exhausted "
            f"({budget_status['elapsed_seconds']} / {config.max_runtime_seconds}s)."
        ),
        "max_plan_revisions": (
            f"Agent stopped: plan revision budget exhausted "
            f"({effective_plan_revision_count} / {config.max_plan_revisions})."
        ),
        "max_recovery_attempts": (
            f"Agent stopped: recovery budget exhausted "
            f"({effective_recovery_attempt_count} / {config.max_recovery_attempts_per_issue})."
        ),
    }
    lines = [reason_line_map.get(reason, "Agent stopped after exhausting a run budget.")]
    if completed_steps:
        lines.append("Completed steps: " + "; ".join(completed_steps[:5]))
    if effective_current_step is not None:
        lines.append(f"Current step: {effective_current_step}")
    if blocked_steps:
        lines.append("Blocked steps: " + "; ".join(blocked_steps[:5]))
    lines.extend(
        [
            "Budget usage:",
            f"- iterations: {budget_status['iteration_count']} / {config.max_iterations}",
            f"- commands: {budget_status['command_count']} / {config.max_command_count}",
            f"- runtime: {budget_status['elapsed_seconds']} / {config.max_runtime_seconds}s",
            f"- plan revisions: {effective_plan_revision_count} / {config.max_plan_revisions}",
            (
                f"- recovery attempts: {effective_recovery_attempt_count} / "
                f"{config.max_recovery_attempts_per_issue}"
            ),
        ]
    )
    if state["observations"]:
        lines.append("Last observations:")
        lines.append(_fmt_observations(state["observations"], n=3))
    return "\n".join(lines)


def _fingerprint_text(value: Any, *, limit: int = 120) -> str:
    text = " ".join(str(value).split()).strip()
    if not text:
        return "unknown"
    if len(text) > limit:
        return text[:limit]
    return text


def _extract_issue_location(*texts: str | None) -> str | None:
    for text in texts:
        if not text:
            continue
        match = _RECOVERY_LOCATION_RE.search(text)
        if match is None:
            continue
        path = match.group(1)
        line = match.group(2)
        return f"{path}:{line}" if line else path
    return None


def _observation_action_label(obs: Observation) -> str:
    result = obs.get("result")
    if obs["tool"] == "run_command" and isinstance(result, dict):
        return f"run_command {result.get('command', '(unknown)')}"
    if obs["tool"] == "read_file" and isinstance(result, dict):
        return f"read_file {result.get('path', '(unknown)')}"
    if obs["tool"] == "search_text" and isinstance(result, dict):
        return f"search_text {result.get('query', '(unknown)')}"
    if obs["tool"] == "list_dir" and isinstance(result, dict):
        return f"list_dir {result.get('path', '(unknown)')}"
    return obs["tool"]


def _observation_produced_new_information(obs: Observation) -> bool:
    result = obs.get("result")
    if isinstance(result, dict):
        if obs["tool"] == "read_file":
            return bool(str(result.get("content", "")).strip())
        if obs["tool"] == "search_text":
            return int(result.get("total_matches", 0) or 0) > 0
        if obs["tool"] == "list_dir":
            entries = result.get("entries")
            return isinstance(entries, list) and len(entries) > 0
        if obs["tool"] == "run_command":
            return any(
                bool(str(result.get(key, "")).strip())
                for key in ("stdout", "stderr")
            ) or result.get("exit_code") is not None
        if obs["tool"] in _WRITE_TOOL_NAMES:
            changed_files = result.get("changed_files")
            return isinstance(changed_files, list) and len(changed_files) > 0
        return bool(result)
    return bool(obs.get("error"))


def _classify_recovery_issue(
    state: AgentState,
    obs: Observation,
) -> dict[str, Any] | None:
    result = obs.get("result")
    if obs["tool"] == "run_command" and isinstance(result, dict):
        command = str(result.get("command", "(unknown)"))
        stdout_preview = _preview_text(result.get("stdout"), limit=220)
        stderr_preview = _preview_text(result.get("stderr"), limit=220)
        issue_location = _extract_issue_location(stderr_preview, stdout_preview)
        if bool(result.get("timed_out", False)):
            return {
                "issue_type": "command_timeout",
                "fingerprint": f"command_timeout::{command}::{issue_location or 'timeout'}",
                "retryable": False,
                "reason": f"Command '{command}' timed out before completing.",
                "recommended_next_action": (
                    "Report the timeout and recommend a narrower command or direct file inspection instead of retrying blindly."
                ),
                "last_action": _observation_action_label(obs),
            }

        pending_verification = state.get("pending_verification")
        if (
            pending_verification is not None
            and not obs["ok"]
            and _is_validation_command(result)
        ):
            changed = ",".join(pending_verification["changed_files"][:3]) or "unknown"
            return {
                "issue_type": "verification_failed",
                "fingerprint": (
                    f"verification_failed::{command}::{changed}::"
                    f"{issue_location or _fingerprint_text(result.get('error') or stderr_preview or stdout_preview)}"
                ),
                "retryable": True,
                "reason": (
                    f"Validation command '{command}' failed after a write and needs a narrower follow-up fix."
                ),
                "recommended_next_action": (
                    "Inspect the validation output and referenced files before proposing another narrow change."
                ),
                "last_action": _observation_action_label(obs),
            }

        if not obs["ok"]:
            return {
                "issue_type": "command_failure",
                "fingerprint": (
                    f"command_failure::{command}::"
                    f"{issue_location or _fingerprint_text(result.get('error') or stderr_preview or stdout_preview)}"
                ),
                "retryable": True,
                "reason": f"Command '{command}' failed and requires nearby context before retrying.",
                "recommended_next_action": _command_follow_up_hint(result),
                "last_action": _observation_action_label(obs),
            }

        return None

    if obs["tool"] == "search_text" and isinstance(result, dict):
        if obs["ok"] and int(result.get("total_matches", 0) or 0) == 0:
            query = str(result.get("query", "")).strip() or "(unknown)"
            return {
                "issue_type": "search_no_results",
                "fingerprint": f"search_no_results::{query}",
                "retryable": True,
                "reason": f"Search returned no matches for '{query}'.",
                "recommended_next_action": (
                    "Try a narrower path or an alternate symbol or file name before repeating the same search."
                ),
                "last_action": _observation_action_label(obs),
            }
        return None

    if obs["tool"] == "read_file" and isinstance(result, dict) and not obs["ok"]:
        path = str(result.get("path", "(unknown)"))
        error_text = str(result.get("error", obs.get("error") or ""))
        issue_type = "file_missing" if "file not found" in error_text else "file_read_error"
        return {
            "issue_type": issue_type,
            "fingerprint": f"{issue_type}::{path}::{_fingerprint_text(error_text)}",
            "retryable": True,
            "reason": f"Reading '{path}' failed: {error_text or 'unknown error'}.",
            "recommended_next_action": (
                "List the parent directory or search for similarly named files before retrying this read."
            ),
            "last_action": _observation_action_label(obs),
        }

    if obs["tool"] == "list_dir" and isinstance(result, dict) and not obs["ok"]:
        path = str(result.get("path", "(unknown)"))
        error_text = str(result.get("error", obs.get("error") or ""))
        issue_type = "path_missing" if "not found" in error_text else "path_access_error"
        return {
            "issue_type": issue_type,
            "fingerprint": f"{issue_type}::{path}::{_fingerprint_text(error_text)}",
            "retryable": True,
            "reason": f"Listing '{path}' failed: {error_text or 'unknown error'}.",
            "recommended_next_action": (
                "Inspect the parent directory or search for nearby paths before repeating the same listing request."
            ),
            "last_action": _observation_action_label(obs),
        }

    return None


def _next_recovery_state(
    config: AgentConfig,
    state: AgentState,
    issue: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if issue is None:
        return None

    previous = state.get("recovery_state")
    previous_fingerprint = (
        str(previous.get("fingerprint"))
        if isinstance(previous, dict) and previous.get("fingerprint") is not None
        else None
    )
    previous_attempt_count = (
        int(previous.get("attempt_count", 0))
        if isinstance(previous, dict)
        else 0
    )
    attempt_count = previous_attempt_count + 1 if previous_fingerprint == issue["fingerprint"] else 1
    can_retry = bool(issue.get("retryable", False)) and attempt_count <= config.max_recovery_attempts_per_issue
    return {
        "issue_type": str(issue["issue_type"]),
        "fingerprint": str(issue["fingerprint"]),
        "attempt_count": attempt_count,
        "last_action": cast(str | None, issue.get("last_action")),
        "can_retry": can_retry,
    }


def _budget_pressure(
    config: AgentConfig,
    budget_status: dict[str, Any],
    *,
    plan_revision_count: int,
    recovery_attempt_count: int,
) -> float:
    ratios: list[float] = []
    if config.max_iterations > 0:
        ratios.append(float(budget_status["iteration_count"]) / float(config.max_iterations))
    if config.max_command_count > 0:
        ratios.append(float(budget_status["command_count"]) / float(config.max_command_count))
    if config.max_runtime_seconds > 0:
        ratios.append(float(budget_status["elapsed_seconds"]) / float(config.max_runtime_seconds))
    if config.max_plan_revisions > 0:
        ratios.append(float(plan_revision_count) / float(config.max_plan_revisions))
    if config.max_recovery_attempts_per_issue > 0:
        ratios.append(
            float(max(0, recovery_attempt_count - 1))
            / float(config.max_recovery_attempts_per_issue)
        )
    return max(ratios, default=0.0)


def _build_reflection_result(
    config: AgentConfig,
    state: AgentState,
    obs: Observation,
    *,
    issue: dict[str, Any] | None,
    recovery_state: dict[str, Any] | None,
    budget_status: dict[str, Any],
    plan_revision_count: int,
) -> dict[str, Any]:
    new_information = _observation_produced_new_information(obs)
    recovery_attempt_count = (
        int(recovery_state.get("attempt_count", 0)) if isinstance(recovery_state, dict) else 0
    )
    budget_pressure = _budget_pressure(
        config,
        budget_status,
        plan_revision_count=plan_revision_count,
        recovery_attempt_count=recovery_attempt_count,
    )
    score = 100
    if not new_information:
        score -= 12
    if not obs["ok"]:
        score -= 30
    if issue is not None:
        issue_type = str(issue["issue_type"])
        if issue_type == "command_timeout":
            score -= 30
        elif issue_type == "verification_failed":
            score -= 22
        elif issue_type in {"search_no_results", "file_missing", "path_missing"}:
            score -= 18
        else:
            score -= 15
    if recovery_attempt_count > 1:
        score -= 15 * (recovery_attempt_count - 1)
    score -= min(plan_revision_count, config.max_plan_revisions) * 5
    score -= int(min(1.0, budget_pressure) * 35)
    score = max(0, min(100, score))

    recommended_next_action: str | None
    reason: str
    retryable = bool(issue.get("retryable", False)) if issue is not None else False
    if issue is None:
        reason = (
            "Latest observation produced new information and does not require bounded recovery."
            if new_information
            else "Latest observation added little information; continue cautiously."
        )
        recommended_next_action = (
            "Proceed to the next planned step using the newest observation as evidence."
        )
        if score <= config.reflection_stop_threshold:
            outcome = "stop"
            recommended_next_action = "Summarize the current evidence instead of continuing."
        elif score <= config.reflection_replan_threshold:
            outcome = "replan"
            recommended_next_action = "Revise the next step before calling another tool."
        else:
            outcome = "continue"
    else:
        issue_type = str(issue["issue_type"])
        reason = str(issue["reason"])
        recommended_next_action = cast(str | None, issue.get("recommended_next_action"))
        if issue_type == "command_timeout":
            outcome = "stop"
            retryable = False
        elif isinstance(recovery_state, dict) and not bool(recovery_state.get("can_retry", False)):
            outcome = "stop"
            retryable = False
            if recommended_next_action is None:
                recommended_next_action = (
                    "Stop automatic recovery and summarize the current state for manual follow-up."
                )
        elif issue_type in {"search_no_results", "file_missing", "path_missing", "verification_failed"}:
            outcome = "replan"
        elif issue_type == "command_failure" and isinstance(recovery_state, dict):
            if int(recovery_state.get("attempt_count", 0)) <= 1:
                outcome = "retry"
            else:
                outcome = "replan"
        elif isinstance(recovery_state, dict) and int(recovery_state.get("attempt_count", 0)) > 1:
            outcome = "replan"
        elif score <= config.reflection_replan_threshold:
            outcome = "replan"
        else:
            outcome = "retry"

    return {
        "score": score,
        "outcome": outcome,
        "reason": reason,
        "retryable": retryable,
        "recommended_next_action": recommended_next_action,
    }


def _build_reflection_stop_answer(
    reflection: dict[str, Any],
    state: AgentState,
    recovery_state: dict[str, Any] | None,
) -> str:
    lines = [
        "Agent stopped after reflector judged that continuing would have low value.",
        f"Reflection score: {reflection.get('score', '(unknown)')}",
        f"Outcome: {reflection.get('outcome', '(unknown)')}",
        f"Reason: {reflection.get('reason', '(none)')}",
    ]
    next_action = reflection.get("recommended_next_action")
    if next_action:
        lines.append(f"Recommended next action: {next_action}")
    if isinstance(recovery_state, dict):
        lines.append(
            "Recovery state: "
            f"{recovery_state.get('issue_type', '(unknown)')} | "
            f"fingerprint={recovery_state.get('fingerprint', '(unknown)')} | "
            f"attempts={recovery_state.get('attempt_count', 0)}"
        )
    if state["observations"]:
        lines.append("Last observations:")
        lines.append(_fmt_observations(state["observations"], n=3))
    return "\n".join(lines)


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

    def _invoke_planner_messages(
        run_id: str,
        messages: list[BaseMessage],
        *,
        retry_reason: str | None = None,
    ) -> AIMessage:
        payload: dict[str, Any] = {
            "message_count": len(messages),
            "messages": [_serialize_trace_message(message) for message in messages],
        }
        if retry_reason is not None:
            payload["retry_reason"] = retry_reason
        _emit_runtime_event(
            run_id,
            EVENT_MODEL_INPUT,
            payload,
            prompt_trace_listener,
        )
        return _coerce_ai_message(llm_with_tools.invoke(messages))

    def planner(state: AgentState) -> dict[str, Any]:
        run_id = state["run_id"]
        started_at = _ensure_started_at(state)
        plan_steps = _effective_plan_steps(state)
        plan_version = _effective_plan_version(state, plan_steps=plan_steps)
        plan_revision_count = _effective_plan_revision_count(state)
        command_count = _effective_command_count(state)
        recovery_attempt_count = _current_recovery_attempt_count(state)
        budget_status = _budget_status_snapshot(
            state,
            started_at=started_at,
            command_count=command_count,
        )

        if budget_status["elapsed_seconds"] >= config.max_runtime_seconds:
            final_answer = _build_budget_stop_answer(
                "max_runtime_seconds",
                state,
                config,
                budget_status,
                plan_steps=plan_steps,
                plan_revision_count=plan_revision_count,
                recovery_attempt_count=recovery_attempt_count,
            )
            _audit_with_legacy_reflector(
                run_id,
                config.log_dir,
                EVENT_BUDGET_EXHAUSTED,
                {
                    "reason": "budget_exhausted",
                    "budget_stop_reason": "max_runtime_seconds",
                    "budget_status": budget_status,
                    "budget_remaining": _budget_remaining(
                        config,
                        budget_status,
                        plan_revision_count=plan_revision_count,
                        recovery_attempt_count=recovery_attempt_count,
                    ),
                },
                event_listener,
                legacy_reason="budget_exhausted",
            )
            return {
                "started_at": started_at,
                "plan": [step["title"] for step in plan_steps],
                "plan_steps": plan_steps,
                "plan_version": plan_version,
                "plan_revision_count": plan_revision_count,
                "command_count": command_count,
                "budget_status": budget_status,
                "budget_stop_reason": "max_runtime_seconds",
                "proposed_tool_call": None,
                "final_answer": final_answer,
            }

        prompt_message = HumanMessage(content=_format_planner_prompt(state, config))
        history: list[BaseMessage] = state["messages"]
        request_messages = [SystemMessage(content=_SYSTEM_PROMPT), *history, prompt_message]
        response = _invoke_planner_messages(run_id, request_messages)
        tool_call = _build_tool_call(response, state, config)

        pending_verification = state.get("pending_verification")
        if pending_verification is not None and tool_call is None:
            reminder = HumanMessage(content=_verification_retry_message(pending_verification))
            response = _invoke_planner_messages(
                run_id,
                [*request_messages, response, reminder],
                retry_reason="pending_write_verification",
            )
            tool_call = _build_tool_call(response, state, config)

        assistant_content = _content_to_text(response.content)

        final_answer: str | None = None
        if tool_call is None:
            if pending_verification is not None:
                final_answer = _build_unverified_write_answer(pending_verification)
            else:
                final_answer = _content_to_text(response.content) or (
                    "Agent returned neither a tool call nor a final answer."
                )

        next_step = None if tool_call is None else _summarize_step(response, tool_call)
        plan_update = _derive_plan_update(
            state,
            next_step,
            assistant_content=assistant_content,
            final_answer=final_answer,
        )
        budget_stop_reason: str | None = None
        if tool_call is not None and tool_call["name"] == "run_command" and command_count >= config.max_command_count:
            budget_stop_reason = "max_command_count"
        elif plan_update["plan_revision_count"] > config.max_plan_revisions:
            budget_stop_reason = "max_plan_revisions"
        elif recovery_attempt_count > config.max_recovery_attempts_per_issue:
            budget_stop_reason = "max_recovery_attempts"

        warning_dimensions = _budget_warning_dimensions(
            config,
            budget_status,
            plan_revision_count=int(plan_update["plan_revision_count"]),
            recovery_attempt_count=recovery_attempt_count,
        )
        if warning_dimensions and not bool(budget_status.get("warning_triggered", False)):
            budget_status = _budget_status_snapshot(
                state,
                started_at=started_at,
                command_count=command_count,
                warning_triggered=True,
            )
            _audit_with_legacy_reflector(
                run_id,
                config.log_dir,
                EVENT_BUDGET_WARNING,
                {
                    "reason": "budget_warning",
                    "dimensions": warning_dimensions,
                    "budget_status": budget_status,
                    "budget_remaining": _budget_remaining(
                        config,
                        budget_status,
                        plan_revision_count=int(plan_update["plan_revision_count"]),
                        recovery_attempt_count=recovery_attempt_count,
                    ),
                },
                event_listener,
                legacy_reason="budget_warning",
            )

        if budget_stop_reason is not None:
            final_answer = _build_budget_stop_answer(
                budget_stop_reason,
                state,
                config,
                budget_status,
                plan_steps=cast(list[dict[str, Any]], plan_update["plan_steps"]),
                current_step=cast(str | None, plan_update["current_step"]),
                plan_revision_count=int(plan_update["plan_revision_count"]),
                recovery_attempt_count=recovery_attempt_count,
            )
            _audit_with_legacy_reflector(
                run_id,
                config.log_dir,
                EVENT_BUDGET_EXHAUSTED,
                {
                    "reason": "budget_exhausted",
                    "budget_stop_reason": budget_stop_reason,
                    "attempted_tool": tool_call["name"] if tool_call is not None else None,
                    "attempted_step": plan_update["current_step"],
                    "budget_status": budget_status,
                    "budget_remaining": _budget_remaining(
                        config,
                        budget_status,
                        plan_revision_count=int(plan_update["plan_revision_count"]),
                        recovery_attempt_count=recovery_attempt_count,
                    ),
                },
                event_listener,
                legacy_reason="budget_exhausted",
            )

        updated_recovery_state = state.get("recovery_state")
        if (
            isinstance(updated_recovery_state, dict)
            and next_step is not None
            and budget_stop_reason is None
        ):
            last_reflection = state.get("last_reflection")
            if isinstance(last_reflection, dict) and str(last_reflection.get("outcome")) in {"retry", "replan"}:
                updated_recovery_state = {
                    **updated_recovery_state,
                    "last_action": next_step,
                }

        # ── Audit ────────────────────────────────────────────────────────
        _one_shot_audit(
            run_id,
            config.log_dir,
            EVENT_PLAN_UPDATE,
            {
                "plan": plan_update["plan"],
                "current_step": plan_update["current_step"],
                "plan_steps": plan_update["plan_steps"],
                "plan_version": plan_update["plan_version"],
                "plan_revision_count": plan_update["plan_revision_count"],
                "plan_revision_reason": plan_update["plan_revision_reason"],
                "assistant_content": assistant_content or None,
                "final_answer": final_answer,
                "budget_status": budget_status,
                "budget_remaining": _budget_remaining(
                    config,
                    budget_status,
                    plan_revision_count=int(plan_update["plan_revision_count"]),
                    recovery_attempt_count=recovery_attempt_count,
                ),
                "last_reflection": state.get("last_reflection"),
                "recovery_state": updated_recovery_state,
            },
            event_listener,
        )
        if plan_update["plan_revision_reason"] is not None:
            _one_shot_audit(
                run_id,
                config.log_dir,
                EVENT_PLAN_REVISED,
                {
                    "plan": plan_update["plan"],
                    "current_step": plan_update["current_step"],
                    "plan_steps": plan_update["plan_steps"],
                    "plan_version": plan_update["plan_version"],
                    "plan_revision_count": plan_update["plan_revision_count"],
                    "plan_revision_reason": plan_update["plan_revision_reason"],
                    "budget_status": budget_status,
                    "budget_remaining": _budget_remaining(
                        config,
                        budget_status,
                        plan_revision_count=int(plan_update["plan_revision_count"]),
                        recovery_attempt_count=recovery_attempt_count,
                    ),
                },
                event_listener,
            )
        if tool_call:
            _one_shot_audit(
                run_id,
                config.log_dir,
                EVENT_TOOL_PROPOSED,
                _tool_call_audit_payload(tool_call),
                event_listener,
            )

        return {
            "messages": [prompt_message, response],
            "started_at": started_at,
            "plan": plan_update["plan"],
            "current_step": plan_update["current_step"],
            "plan_steps": plan_update["plan_steps"],
            "plan_version": plan_update["plan_version"],
            "plan_revision_count": plan_update["plan_revision_count"],
            "command_count": command_count,
            "budget_status": budget_status,
            "budget_stop_reason": budget_stop_reason,
            "recovery_attempt_total": _effective_recovery_attempt_total(state),
            "recovery_state": updated_recovery_state,
            "proposed_tool_call": None if budget_stop_reason is not None else tool_call,
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
            return {"risk_decision": "allow", "pending_approval": None}

        assessment = assess_tool_call(tc, config, run_id=state["run_id"])
        decision = assessment["decision"]
        approval_request = assessment["approval_request"]

        audit_payload: dict[str, Any] = {
            **_tool_call_audit_payload(tc),
            "decision": decision,
        }
        if assessment["reason"] is not None:
            audit_payload["reason"] = assessment["reason"]
        if approval_request is not None:
            audit_payload["approval_request_id"] = approval_request["id"]
            audit_payload["impact_summary"] = approval_request["impact_summary"]
            audit_payload["diff_preview"] = approval_request["diff_preview"]
            audit_payload["backup_plan"] = approval_request["backup_plan"]
            audit_payload["affected_files"] = approval_request.get("affected_files")
            audit_payload["rollback_command"] = approval_request.get("rollback_command")
            audit_payload["suggested_verification_command"] = approval_request.get("suggested_verification_command")

        _one_shot_audit(
            state["run_id"],
            config.log_dir,
            EVENT_POLICY_DECISION,
            audit_payload,
            event_listener,
        )

        if decision == "deny":
            return {
                "risk_decision": "deny",
                "pending_approval": None,
                "final_answer": (
                    f"Operation denied by security policy: "
                    f"tool '{tc['name']}' with args {tc['args']} is not permitted."
                    + (
                        f" Reason: {assessment['reason']}."
                        if assessment["reason"] is not None
                        else ""
                    )
                ),
            }

        if decision == "needs_approval" and approval_request is not None:
            return {
                "risk_decision": "needs_approval",
                "pending_approval": approval_request,
                "final_answer": (
                    f"Approval required before executing tool '{tc['name']}'. "
                    f"{approval_request['impact_summary']} "
                    f"Reason: {approval_request['reason']}"
                ),
            }

        return {"risk_decision": "allow", "pending_approval": None}

    return policy_guard


def _make_approval_pause(
    config: AgentConfig,
    event_listener: AuditEventListener | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    def approval_pause(state: AgentState) -> dict[str, Any]:
        approval_request = state.get("pending_approval")
        tool_call = state.get("proposed_tool_call")
        if approval_request is None or tool_call is None:
            return {}

        snapshot_path = save_run_state(state, config)
        approval_view = build_approval_view(state, config, state_path=snapshot_path)
        _one_shot_audit(
            state["run_id"],
            config.log_dir,
            EVENT_APPROVAL_REQUESTED,
            {
                **_tool_call_audit_payload(tool_call),
                "approval_request_id": approval_request["id"],
                "reason": approval_request["reason"],
                "impact_summary": approval_request["impact_summary"],
                "diff_preview": approval_request["diff_preview"],
                "backup_plan": approval_request["backup_plan"],
                "state_path": str(snapshot_path),
                "resume_approve_command": f"--resume-run {state['run_id']} --approve",
                "resume_reject_command": f"--resume-run {state['run_id']} --reject",
            },
            event_listener,
        )
        _one_shot_audit(
            state["run_id"],
            config.log_dir,
            EVENT_APPROVAL_PRESENTED,
            {
                **approval_view,
                "source": "approval_pause",
            },
            event_listener,
        )

        message = state.get("final_answer") or (
            f"Approval required before executing tool '{tool_call['name']}'."
        )
        message = (
            f"{message}\n\n"
            f"Review with: --show-pending-run {state['run_id']}\n"
            f"Resume with: --resume-run {state['run_id']} --approve\n"
            f"Reject with: --resume-run {state['run_id']} --reject\n"
            "Use --decision-note \"...\" to record why you approved or rejected."
        )
        return {"final_answer": message}

    return approval_pause


def _make_resume_gate(
    config: AgentConfig,
    event_listener: AuditEventListener | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    def resume_gate(state: AgentState) -> dict[str, Any]:
        action = state.get("resume_action")
        approval_request = state.get("pending_approval")
        tool_call = state.get("proposed_tool_call")
        response_note = cast(str | None, state.get("approval_response_note"))

        if action not in {"approve", "reject"}:
            return {
                "final_answer": "Resume requested without a valid approval decision.",
            }

        if approval_request is None or tool_call is None:
            delete_run_state(state["run_id"], config)
            return {
                "resume_action": None,
                "approval_response_note": None,
                "final_answer": "No pending approval request was found for this run.",
            }

        _one_shot_audit(
            state["run_id"],
            config.log_dir,
            EVENT_APPROVAL_RESPONSE,
            {
                "approval_request_id": approval_request["id"],
                "tool": approval_request["tool"],
                "action": action,
                "note": response_note,
                "affected_files": approval_request.get("affected_files"),
                "source": "resume_gate",
            },
            event_listener,
        )
        delete_run_state(state["run_id"], config)
        if action == "approve":
            return {
                "resume_action": None,
                "approval_response_note": None,
                "risk_decision": "allow",
                "final_answer": None,
            }

        note_suffix = "" if not response_note else f" Reviewer note: {response_note}"
        return {
            "resume_action": None,
            "approval_response_note": None,
            "risk_decision": "deny",
            "pending_approval": None,
            "proposed_tool_call": None,
            "final_answer": (
                f"User rejected approval request '{approval_request['id']}' for tool "
                f"'{tool_call['name']}'. No changes were applied. "
                "Inspect additional context and propose a narrower manual change if needed."
                f"{note_suffix}"
            ),
        }

    return resume_gate


# ─────────────────────────────────────────────────────────────────────────────
# T11 – ToolExecutor node
# ─────────────────────────────────────────────────────────────────────────────

# Dispatch table: tool name → skill function
_SKILL_DISPATCH: dict[str, Any] = {
    "list_dir": list_dir,
    "read_file": read_file,
    "search_text": search_text,
    "run_command": run_command,
    "apply_patch": apply_patch,
    "write_file": write_file,
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
                # primary arg key: "path" for list_dir/read_file,
                # "query" for search_text, "command" for run_command
                extra = dict(tc["args"])
                skill_kwargs: dict[str, Any] = {}
                if tc["name"] == "search_text":
                    primary = str(extra.pop("query", ""))
                    result = skill_fn(primary, config, **extra)
                elif tc["name"] == "run_command":
                    primary = str(extra.pop("command", ""))
                    result = skill_fn(primary, config, **extra)
                elif tc["name"] == "apply_patch":
                    primary = str(extra.pop("patch", extra.pop("diff", "")))
                    skill_kwargs["run_id"] = state["run_id"]
                    result = skill_fn(primary, config, **extra, **skill_kwargs)
                elif tc["name"] == "write_file":
                    primary = str(extra.pop("path", "."))
                    content = str(extra.pop("content", ""))
                    skill_kwargs["run_id"] = state["run_id"]
                    result = skill_fn(primary, content, config, **extra, **skill_kwargs)
                else:
                    primary = str(extra.pop("path", "."))
                    result = skill_fn(primary, config, **extra)
                duration_ms = int((time.monotonic() - t0) * 1000)
                ok = bool(result.get("ok", False))
                obs = Observation(
                    tool=tc["name"],
                    ok=ok,
                    result=result,
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
            _tool_result_audit_payload(tc, obs),
            event_listener,
        )
        result = obs.get("result")
        if obs["tool"] in {"apply_patch", "write_file"} and isinstance(result, dict):
            write_payload = {
                **_tool_result_audit_payload(tc, obs),
                "approval_request_id": (
                    state["pending_approval"]["id"]
                    if state.get("pending_approval") is not None
                    else None
                ),
            }
            if result.get("rolled_back"):
                _one_shot_audit(
                    state["run_id"],
                    config.log_dir,
                    EVENT_WRITE_ROLLBACK,
                    write_payload,
                    event_listener,
                )
            elif obs["ok"]:
                _one_shot_audit(
                    state["run_id"],
                    config.log_dir,
                    EVENT_WRITE_APPLIED,
                    write_payload,
                    event_listener,
                )

        # Reset consecutive_failures on success; increment on failure
        new_failures = 0 if obs["ok"] else state["consecutive_failures"] + 1

        state_update: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    content=_serialize_tool_payload(obs),
                    tool_call_id=tc["id"],
                )
            ],
            "observations": state["observations"] + [obs],
            "iteration_count": state["iteration_count"] + 1,
            "command_count": _effective_command_count(state) + (1 if tc["name"] == "run_command" else 0),
            "consecutive_failures": new_failures,
            "risk_decision": None,
            "pending_approval": None,
            "resume_action": None,
            "proposed_tool_call": None,
            "budget_status": _budget_status_snapshot(
                state,
                started_at=cast(str | None, state.get("started_at")),
                iteration_count=state["iteration_count"] + 1,
                command_count=_effective_command_count(state) + (1 if tc["name"] == "run_command" else 0),
            ),
            "budget_stop_reason": None,
        }

        if obs["tool"] in _WRITE_TOOL_NAMES and obs["ok"] and isinstance(result, dict):
            state_update.update(
                {
                    "last_write": _build_write_summary(
                        obs["tool"],
                        result,
                        approval_request_id=(
                            state["pending_approval"]["id"]
                            if state.get("pending_approval") is not None
                            else None
                        ),
                    ),
                    "pending_verification": _build_write_summary(
                        obs["tool"],
                        result,
                        approval_request_id=(
                            state["pending_approval"]["id"]
                            if state.get("pending_approval") is not None
                            else None
                        ),
                    ),
                    "last_verification": None,
                    "last_rollback": None,
                }
            )

        return state_update

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
        last_obs = state["observations"][-1] if state["observations"] else None
        budget_status = _budget_status_snapshot(state)
        plan_revision_count = _effective_plan_revision_count(state)
        recovery_attempt_count = _current_recovery_attempt_count(state)
        recovery_attempt_total = _effective_recovery_attempt_total(state)
        reflection_result: dict[str, Any] | None = None
        next_recovery_state: dict[str, Any] | None = cast(
            dict[str, Any] | None,
            state.get("recovery_state"),
        )

        # Circuit-breaker: total iteration limit
        if state["iteration_count"] >= config.max_iterations:
            _audit_with_legacy_reflector(
                run_id,
                config.log_dir,
                EVENT_BUDGET_EXHAUSTED,
                {
                    "reason": "budget_exhausted",
                    "budget_stop_reason": "max_iterations",
                    "iteration_count": state["iteration_count"],
                    "budget_status": budget_status,
                    "budget_remaining": _budget_remaining(
                        config,
                        budget_status,
                        plan_revision_count=plan_revision_count,
                        recovery_attempt_count=recovery_attempt_count,
                    ),
                },
                event_listener,
                legacy_reason="budget_exhausted",
            )
            return {
                "budget_status": budget_status,
                "budget_stop_reason": "max_iterations",
                "final_answer": _build_budget_stop_answer(
                    "max_iterations",
                    state,
                    config,
                    budget_status,
                    plan_revision_count=plan_revision_count,
                    recovery_attempt_count=recovery_attempt_count,
                ),
            }

        if budget_status["elapsed_seconds"] >= config.max_runtime_seconds:
            _audit_with_legacy_reflector(
                run_id,
                config.log_dir,
                EVENT_BUDGET_EXHAUSTED,
                {
                    "reason": "budget_exhausted",
                    "budget_stop_reason": "max_runtime_seconds",
                    "budget_status": budget_status,
                    "budget_remaining": _budget_remaining(
                        config,
                        budget_status,
                        plan_revision_count=plan_revision_count,
                        recovery_attempt_count=recovery_attempt_count,
                    ),
                },
                event_listener,
                legacy_reason="budget_exhausted",
            )
            return {
                "budget_status": budget_status,
                "budget_stop_reason": "max_runtime_seconds",
                "final_answer": _build_budget_stop_answer(
                    "max_runtime_seconds",
                    state,
                    config,
                    budget_status,
                    plan_revision_count=plan_revision_count,
                    recovery_attempt_count=recovery_attempt_count,
                ),
            }

        if recovery_attempt_count > config.max_recovery_attempts_per_issue:
            _audit_with_legacy_reflector(
                run_id,
                config.log_dir,
                EVENT_BUDGET_EXHAUSTED,
                {
                    "reason": "budget_exhausted",
                    "budget_stop_reason": "max_recovery_attempts",
                    "budget_status": budget_status,
                    "budget_remaining": _budget_remaining(
                        config,
                        budget_status,
                        plan_revision_count=plan_revision_count,
                        recovery_attempt_count=recovery_attempt_count,
                    ),
                },
                event_listener,
                legacy_reason="budget_exhausted",
            )
            return {
                "budget_status": budget_status,
                "budget_stop_reason": "max_recovery_attempts",
                "final_answer": _build_budget_stop_answer(
                    "max_recovery_attempts",
                    state,
                    config,
                    budget_status,
                    plan_revision_count=plan_revision_count,
                    recovery_attempt_count=recovery_attempt_count,
                ),
            }

        if last_obs is not None:
            issue = _classify_recovery_issue(state, last_obs)
            next_recovery_state = _next_recovery_state(config, state, issue)
            reflection_result = _build_reflection_result(
                config,
                state,
                last_obs,
                issue=issue,
                recovery_state=next_recovery_state,
                budget_status=budget_status,
                plan_revision_count=plan_revision_count,
            )
            next_recovery_attempt_count = (
                int(next_recovery_state.get("attempt_count", 0))
                if isinstance(next_recovery_state, dict)
                else 0
            )
            next_recovery_attempt_total = (
                recovery_attempt_total + 1
                if isinstance(next_recovery_state, dict)
                else recovery_attempt_total
            )
            _audit_with_legacy_reflector(
                run_id,
                config.log_dir,
                EVENT_REFLECTION_SCORED,
                {
                    "score": reflection_result["score"],
                    "outcome": reflection_result["outcome"],
                    "retryable": reflection_result["retryable"],
                    "reflection_reason": reflection_result["reason"],
                    "recommended_next_action": reflection_result["recommended_next_action"],
                    "tool": last_obs["tool"],
                    "issue_type": None if issue is None else issue["issue_type"],
                    "recovery_attempt_count": next_recovery_attempt_count,
                    "recovery_attempt_total": next_recovery_attempt_total,
                    "budget_status": budget_status,
                    "budget_pressure": _budget_pressure(
                        config,
                        budget_status,
                        plan_revision_count=plan_revision_count,
                        recovery_attempt_count=next_recovery_attempt_count,
                    ),
                    "new_information": _observation_produced_new_information(last_obs),
                },
                event_listener,
                legacy_reason="reflection_scored",
            )
            if isinstance(next_recovery_state, dict):
                _audit_with_legacy_reflector(
                    run_id,
                    config.log_dir,
                    EVENT_RECOVERY_ATTEMPTED,
                    {
                        **next_recovery_state,
                        "recovery_attempt_total": next_recovery_attempt_total,
                        "recommended_next_action": reflection_result["recommended_next_action"],
                    },
                    event_listener,
                    legacy_reason="recovery_attempted",
                )
                if not bool(next_recovery_state.get("can_retry", False)):
                    _audit_with_legacy_reflector(
                        run_id,
                        config.log_dir,
                        EVENT_RECOVERY_EXHAUSTED,
                        {
                            **next_recovery_state,
                            "recovery_attempt_total": next_recovery_attempt_total,
                            "recommended_next_action": reflection_result["recommended_next_action"],
                        },
                        event_listener,
                        legacy_reason="recovery_exhausted",
                    )
            elif isinstance(state.get("recovery_state"), dict):
                previous_recovery_state = cast(dict[str, Any], state["recovery_state"])
                _audit_with_legacy_reflector(
                    run_id,
                    config.log_dir,
                    EVENT_RECOVERY_CLEARED,
                    {
                        "issue_type": previous_recovery_state.get("issue_type"),
                        "fingerprint": previous_recovery_state.get("fingerprint"),
                        "recovery_attempt_total": recovery_attempt_total,
                    },
                    event_listener,
                    legacy_reason="recovery_cleared",
                )
            else:
                next_recovery_attempt_total = recovery_attempt_total

            command_reflection = _command_reflection_payload(last_obs)
            if command_reflection is not None:
                _one_shot_audit(
                    run_id,
                    config.log_dir,
                    EVENT_REFLECTOR_ACTION,
                    command_reflection,
                    event_listener,
                )
                result = last_obs.get("result")
                if (
                    last_obs["tool"] == "run_command"
                    and isinstance(result, dict)
                    and result.get("timed_out")
                ):
                    return {
                        "last_reflection": reflection_result,
                        "recovery_state": next_recovery_state,
                        "recovery_attempt_total": next_recovery_attempt_total,
                        "final_answer": _build_command_timeout_answer(last_obs),
                    }

        pending_verification = state.get("pending_verification")
        if last_obs is not None and pending_verification is not None:
            if last_obs["tool"] in _WRITE_TOOL_NAMES and last_obs["ok"]:
                _one_shot_audit(
                    run_id,
                    config.log_dir,
                    EVENT_REFLECTOR_ACTION,
                    {
                        "reason": "write_requires_verification",
                        "action": "run_validation_command",
                        "changed_files": pending_verification["changed_files"],
                        "backup_root": pending_verification["backup_root"],
                        "manifest_path": pending_verification["manifest_path"],
                        "guidance": (
                            "Run a narrow validation command before finalizing any answer about the recent file change."
                        ),
                    },
                    event_listener,
                )
            elif last_obs["tool"] == "run_command" and isinstance(last_obs.get("result"), dict):
                command_result = cast(dict[str, Any], last_obs["result"])
                if _is_validation_command(command_result):
                    verification = _build_verification_summary(command_result, ok=last_obs["ok"])
                    if verification["ok"]:
                        _one_shot_audit(
                            run_id,
                            config.log_dir,
                            EVENT_REFLECTOR_ACTION,
                            {
                                "reason": "write_validation_passed",
                                "action": "finalize_after_verified_write",
                                "command": verification["command"],
                                "changed_files": pending_verification["changed_files"],
                            },
                            event_listener,
                        )
                        return {
                            "pending_verification": None,
                            "last_verification": verification,
                            "last_rollback": None,
                            "last_reflection": reflection_result,
                            "recovery_state": next_recovery_state,
                            "recovery_attempt_total": next_recovery_attempt_total,
                        }

                    rollback_summary: RollbackSummary | None = None
                    if config.auto_rollback_on_verify_failure:
                        rollback_result = rollback_run(state["run_id"], config)
                        rollback_summary = _build_rollback_summary(
                            rollback_result,
                            trigger="verify_failure",
                        )
                        _one_shot_audit(
                            run_id,
                            config.log_dir,
                            EVENT_WRITE_ROLLBACK,
                            {
                                "tool": "rollback_run",
                                "trigger": "verify_failure",
                                "approval_request_id": pending_verification["approval_request_id"],
                                **rollback_summary,
                            },
                            event_listener,
                        )

                    _one_shot_audit(
                        run_id,
                        config.log_dir,
                        EVENT_REFLECTOR_ACTION,
                        {
                            "reason": "write_validation_failed",
                            "action": (
                                "auto_rollback"
                                if config.auto_rollback_on_verify_failure
                                else "inspect_failure_context"
                            ),
                            "command": verification["command"],
                            "exit_code": verification["exit_code"],
                            "stderr_preview": verification["stderr_preview"],
                            "stdout_preview": verification["stdout_preview"],
                            "changed_files": pending_verification["changed_files"],
                        },
                        event_listener,
                    )

                    state_update: dict[str, Any] = {
                        "pending_verification": None,
                        "last_verification": verification,
                        "last_rollback": rollback_summary,
                        "last_reflection": reflection_result,
                        "recovery_state": next_recovery_state,
                        "recovery_attempt_total": next_recovery_attempt_total,
                    }
                    if rollback_summary is not None:
                        state_update["final_answer"] = _build_validation_failure_answer(
                            pending_verification,
                            verification,
                            rollback_summary,
                        )
                    return state_update

                _one_shot_audit(
                    run_id,
                    config.log_dir,
                    EVENT_REFLECTOR_ACTION,
                    {
                        "reason": "write_validation_still_required",
                        "action": "run_validation_command",
                        "command": command_result.get("command"),
                        "guidance": (
                            "The latest command does not look like a validation step. Run tests, lint, or type checks before finalizing the recent write."
                        ),
                    },
                    event_listener,
                )

        if reflection_result is not None and reflection_result["outcome"] == "stop":
            state_update: dict[str, Any] = {
                "last_reflection": reflection_result,
                "recovery_state": next_recovery_state,
                "recovery_attempt_total": next_recovery_attempt_total,
            }
            if isinstance(next_recovery_state, dict) and not bool(next_recovery_state.get("can_retry", False)):
                state_update.update(
                    {
                        "budget_stop_reason": "max_recovery_attempts",
                        "final_answer": _build_budget_stop_answer(
                            "max_recovery_attempts",
                            state,
                            config,
                            budget_status,
                            plan_revision_count=plan_revision_count,
                            recovery_attempt_count=int(next_recovery_state.get("attempt_count", 0)),
                        ),
                    }
                )
            else:
                state_update["final_answer"] = _build_reflection_stop_answer(
                    reflection_result,
                    state,
                    next_recovery_state,
                )
            return state_update

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
                "last_reflection": reflection_result,
                "recovery_state": next_recovery_state,
                "recovery_attempt_total": next_recovery_attempt_total,
                "final_answer": (
                    f"Agent stopped: {state['consecutive_failures']} consecutive "
                    f"tool failures.\nLast error: {last_err}"
                )
            }

        # All good – continue the loop
        return {
            "last_reflection": reflection_result,
            "recovery_state": next_recovery_state,
            "recovery_attempt_total": (
                next_recovery_attempt_total
                if last_obs is not None
                else recovery_attempt_total
            ),
        }

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
        command_summaries = _command_summary_lines(state["observations"])
        write_summaries = _write_summary_lines(state["observations"])
        last_verification = state.get("last_verification")
        last_rollback = state.get("last_rollback")
        pending_verification = state.get("pending_verification")
        last_reflection = cast(dict[str, Any] | None, state.get("last_reflection"))
        recovery_state = cast(dict[str, Any] | None, state.get("recovery_state"))
        plan_steps = _effective_plan_steps(state)
        plan_revision_count = _effective_plan_revision_count(state)
        recovery_attempt_count = _current_recovery_attempt_count(state)
        recovery_attempt_total = _effective_recovery_attempt_total(state)
        budget_status = _budget_status_snapshot(state)
        budget_remaining = _budget_remaining(
            config,
            budget_status,
            plan_revision_count=plan_revision_count,
            recovery_attempt_count=recovery_attempt_count,
        )
        budget_stop_reason = cast(str | None, state.get("budget_stop_reason"))
        if command_summaries and "Executed commands:" not in final:
            final = f"{final}\n\nExecuted commands:\n" + "\n".join(command_summaries)
        if write_summaries and "Applied file changes:" not in final:
            final = f"{final}\n\nApplied file changes:\n" + "\n".join(write_summaries)
        if pending_verification is not None and "Validation still required:" not in final:
            changed = ", ".join(pending_verification["changed_files"][:5]) or "(unknown files)"
            validation_lines = [
                f"Recent write remains unverified for: {changed}",
            ]
            if pending_verification.get("backup_root"):
                validation_lines.append(f"Backup root: {pending_verification['backup_root']}")
            if pending_verification.get("manifest_path"):
                validation_lines.append(f"Manifest path: {pending_verification['manifest_path']}")
            final = f"{final}\n\nValidation still required:\n" + "\n".join(validation_lines)
        if last_verification is not None and "Validation result:" not in final:
            validation_lines = [
                f"Command: {last_verification['command']} [cwd={last_verification['cwd']}]",
                f"Status: {'passed' if last_verification['ok'] else 'failed'}",
            ]
            if last_verification["exit_code"] is not None:
                validation_lines.append(f"Exit code: {last_verification['exit_code']}")
            if last_verification.get("stderr_preview"):
                validation_lines.append(f"stderr preview: {last_verification['stderr_preview']}")
            elif last_verification.get("stdout_preview"):
                validation_lines.append(f"stdout preview: {last_verification['stdout_preview']}")
            final = f"{final}\n\nValidation result:\n" + "\n".join(validation_lines)
        if last_rollback is not None and "Rollback result:" not in final:
            rollback_lines = [
                f"Trigger: {last_rollback['trigger']}",
                f"Status: {'ok' if last_rollback['ok'] else 'failed'}",
            ]
            if last_rollback["restored_files"]:
                rollback_lines.append(
                    "Restored files: " + ", ".join(last_rollback["restored_files"])
                )
            if last_rollback["removed_files"]:
                rollback_lines.append(
                    "Removed files: " + ", ".join(last_rollback["removed_files"])
                )
            if last_rollback.get("manifest_path"):
                rollback_lines.append(f"Manifest path: {last_rollback['manifest_path']}")
            if last_rollback.get("error"):
                rollback_lines.append(f"Error: {last_rollback['error']}")
            final = f"{final}\n\nRollback result:\n" + "\n".join(rollback_lines)
        if budget_stop_reason is not None and "Budget usage:" not in final:
            budget_lines = [
                f"Stop reason: {budget_stop_reason}",
                f"Iterations: {budget_status['iteration_count']} / {config.max_iterations}",
                f"Commands: {budget_status['command_count']} / {config.max_command_count}",
                f"Runtime: {budget_status['elapsed_seconds']} / {config.max_runtime_seconds}s",
                f"Plan revisions: {plan_revision_count} / {config.max_plan_revisions}",
            ]
            if state.get("current_step") is not None:
                budget_lines.append(f"Current step: {state['current_step']}")
            final = f"{final}\n\nBudget usage:\n" + "\n".join(budget_lines)
        if (
            last_reflection is not None
            and last_reflection.get("outcome") != "continue"
            and "Last reflection:" not in final
        ):
            reflection_lines = [
                f"Score: {last_reflection['score']}",
                f"Outcome: {last_reflection['outcome']}",
                f"Reason: {last_reflection['reason']}",
            ]
            if last_reflection.get("recommended_next_action"):
                reflection_lines.append(
                    f"Recommended next action: {last_reflection['recommended_next_action']}"
                )
            final = f"{final}\n\nLast reflection:\n" + "\n".join(reflection_lines)
        if recovery_state is not None and "Recovery state:" not in final:
            recovery_lines = [
                f"Issue type: {recovery_state['issue_type']}",
                f"Fingerprint: {recovery_state['fingerprint']}",
                f"Attempt count: {recovery_state['attempt_count']}",
                f"Can retry: {recovery_state['can_retry']}",
            ]
            if recovery_state.get("last_action"):
                recovery_lines.append(f"Last action: {recovery_state['last_action']}")
            final = f"{final}\n\nRecovery state:\n" + "\n".join(recovery_lines)

        if budget_stop_reason is not None:
            status = "budget_exhausted"
        elif last_reflection is not None and last_reflection.get("outcome") == "stop":
            status = "stopped_after_reflection"
        elif state.get("risk_decision") == "needs_approval":
            status = "paused_for_approval"
        elif final.startswith("User rejected approval request"):
            status = "approval_rejected"
        elif state.get("risk_decision") == "deny":
            status = "denied"
        elif last_rollback is not None and last_verification is not None and not last_verification["ok"]:
            status = "rolled_back_after_failed_verification"
        elif last_verification is not None and not last_verification["ok"]:
            status = "verification_failed"
        elif last_verification is not None and last_verification["ok"]:
            status = "verified_write_completed"
        elif pending_verification is not None:
            status = "completed_with_unverified_write"
        else:
            status = "completed"

        _one_shot_audit(
            state["run_id"],
            config.log_dir,
            EVENT_RUN_END,
            {
                "status": status,
                "final_answer": final,
                "iteration_count": state["iteration_count"],
                "observation_count": len(state["observations"]),
                "command_count": budget_status["command_count"],
                "command_summaries": command_summaries,
                "write_count": len(write_summaries),
                "write_summaries": write_summaries,
                "plan_version": _effective_plan_version(state, plan_steps=plan_steps),
                "plan_revision_count": plan_revision_count,
                "plan_steps": plan_steps,
                "budget_status": budget_status,
                "budget_remaining": budget_remaining,
                "budget_usage": {
                    "iterations_used": budget_status["iteration_count"],
                    "iteration_limit": config.max_iterations,
                    "commands_used": budget_status["command_count"],
                    "command_limit": config.max_command_count,
                    "runtime_seconds": budget_status["elapsed_seconds"],
                    "runtime_limit_seconds": config.max_runtime_seconds,
                    "plan_revisions_used": plan_revision_count,
                    "plan_revision_limit": config.max_plan_revisions,
                    "recovery_attempts_used": recovery_attempt_total,
                    "recovery_attempt_limit_per_issue": config.max_recovery_attempts_per_issue,
                },
                "budget_stop_reason": budget_stop_reason,
                "runtime_seconds": budget_status["elapsed_seconds"],
                "last_reflection": last_reflection,
                "recovery_state": recovery_state,
                "recovery_attempt_total": recovery_attempt_total,
                "verification_status": (
                    None
                    if last_verification is None
                    else ("passed" if last_verification["ok"] else "failed")
                ),
                "verification_command": (
                    last_verification["command"] if last_verification is not None else None
                ),
                "verification_exit_code": (
                    last_verification["exit_code"] if last_verification is not None else None
                ),
                "rollback_result": last_rollback,
                "approval_request_id": (
                    state["pending_approval"]["id"]
                    if state.get("pending_approval") is not None
                    else None
                ),
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


def _route_start(state: AgentState) -> str:
    if state.get("resume_action") in {"approve", "reject"}:
        return "resume_gate"
    return "planner"


def _route_resume_gate(state: AgentState) -> str:
    if state.get("final_answer") is not None:
        return "finalizer"
    return "tool_executor"


def _route_policy_guard(state: AgentState) -> str:
    if state.get("risk_decision") == "needs_approval":
        return "approval_pause"
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
            "pending_approval": None,
            "resume_action": None,
            "pending_verification": None,
            "last_write": None,
            "last_verification": None,
            "last_rollback": None,
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
    graph.add_node("resume_gate", _make_resume_gate(config, event_listener))
    graph.add_node("policy_guard", _make_policy_guard(config, event_listener))
    graph.add_node("approval_pause", _make_approval_pause(config, event_listener))
    graph.add_node("tool_executor", _make_tool_executor(config, event_listener))
    graph.add_node("reflector", _make_reflector(config, event_listener))
    graph.add_node("finalizer", _make_finalizer(config, event_listener))

    # ── Entry point ───────────────────────────────────────────────────────
    graph.add_conditional_edges(
        START,
        _route_start,
        {"planner": "planner", "resume_gate": "resume_gate"},
    )
    graph.add_conditional_edges(
        "resume_gate",
        _route_resume_gate,
        {"tool_executor": "tool_executor", "finalizer": "finalizer"},
    )

    # ── Conditional edges ─────────────────────────────────────────────────
    graph.add_conditional_edges(
        "planner",
        _route_planner,
        {"policy_guard": "policy_guard", "finalizer": "finalizer"},
    )
    graph.add_conditional_edges(
        "policy_guard",
        _route_policy_guard,
        {
            "tool_executor": "tool_executor",
            "approval_pause": "approval_pause",
            "finalizer": "finalizer",
        },
    )
    graph.add_edge("approval_pause", "finalizer")
    graph.add_edge("tool_executor", "reflector")
    graph.add_conditional_edges(
        "reflector",
        _route_reflector,
        {"planner": "planner", "finalizer": "finalizer"},
    )

    # ── Terminal ──────────────────────────────────────────────────────────
    graph.add_edge("finalizer", END)

    return graph.compile()

