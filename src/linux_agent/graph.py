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
    EVENT_APPROVAL_REQUESTED,
    EVENT_MODEL_INPUT,
    EVENT_PLAN_UPDATE,
    EVENT_POLICY_DECISION,
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
    plan_text = (
        "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(state["plan"]))
        or "  (not yet planned)"
    )
    obs_text = _fmt_observations(state["observations"])
    write_text = _format_pending_verification(state)
    return (
        f"Goal: {state['user_goal']}\n\n"
        f"Workspace root: {state['workspace_root']}\n\n"
        f"Current plan:\n{plan_text}\n\n"
        f"Recent observations:\n{obs_text}\n\n"
        f"Write verification status:\n{write_text}\n\n"
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
        current_step = _summarize_step(response, tool_call)
        plan = _append_plan_step(state["plan"], current_step)

        final_answer: str | None = None
        if tool_call is None:
            if pending_verification is not None:
                final_answer = _build_unverified_write_answer(pending_verification)
            else:
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
                _tool_call_audit_payload(tool_call),
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

        message = state.get("final_answer") or (
            f"Approval required before executing tool '{tool_call['name']}'."
        )
        message = (
            f"{message}\n\n"
            f"Resume with: --resume-run {state['run_id']} --approve\n"
            f"Reject with: --resume-run {state['run_id']} --reject"
        )
        return {"final_answer": message}

    return approval_pause


def _make_resume_gate(config: AgentConfig) -> Callable[[AgentState], dict[str, Any]]:
    def resume_gate(state: AgentState) -> dict[str, Any]:
        action = state.get("resume_action")
        approval_request = state.get("pending_approval")
        tool_call = state.get("proposed_tool_call")

        if action not in {"approve", "reject"}:
            return {
                "final_answer": "Resume requested without a valid approval decision.",
            }

        if approval_request is None or tool_call is None:
            delete_run_state(state["run_id"], config)
            return {
                "resume_action": None,
                "final_answer": "No pending approval request was found for this run.",
            }

        delete_run_state(state["run_id"], config)
        if action == "approve":
            return {
                "resume_action": None,
                "risk_decision": "allow",
                "final_answer": None,
            }

        return {
            "resume_action": None,
            "risk_decision": "deny",
            "pending_approval": None,
            "proposed_tool_call": None,
            "final_answer": (
                f"User rejected approval request '{approval_request['id']}' for tool "
                f"'{tool_call['name']}'. No changes were applied. "
                "Inspect additional context and propose a narrower manual change if needed."
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
            "consecutive_failures": new_failures,
            "risk_decision": None,
            "pending_approval": None,
            "resume_action": None,
            "proposed_tool_call": None,
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

        if last_obs is not None:
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
                    return {"final_answer": _build_command_timeout_answer(last_obs)}

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
        command_summaries = _command_summary_lines(state["observations"])
        write_summaries = _write_summary_lines(state["observations"])
        last_verification = state.get("last_verification")
        last_rollback = state.get("last_rollback")
        pending_verification = state.get("pending_verification")
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

        if state.get("risk_decision") == "needs_approval":
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
                "command_count": len(command_summaries),
                "command_summaries": command_summaries,
                "write_count": len(write_summaries),
                "write_summaries": write_summaries,
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
    graph.add_node("resume_gate", _make_resume_gate(config))
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

