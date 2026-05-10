"""Reusable approval payload and CLI rendering helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from linux_agent.config import AgentConfig
from linux_agent.state import AgentState


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _default_budget_status(state: AgentState) -> dict[str, Any]:
    raw = state.get("budget_status")
    if isinstance(raw, dict):
        return {
            "iteration_count": _safe_int(raw.get("iteration_count"), _safe_int(state.get("iteration_count"), 0)),
            "command_count": _safe_int(raw.get("command_count"), _safe_int(state.get("command_count"), 0)),
            "elapsed_seconds": _safe_int(raw.get("elapsed_seconds"), 0),
            "warning_triggered": bool(raw.get("warning_triggered", False)),
        }
    return {
        "iteration_count": _safe_int(state.get("iteration_count"), 0),
        "command_count": _safe_int(state.get("command_count"), 0),
        "elapsed_seconds": 0,
        "warning_triggered": False,
    }


def _normalize_affected_files(request: dict[str, Any]) -> list[str]:
    raw_files = request.get("affected_files")
    if isinstance(raw_files, list):
        normalized = [str(path) for path in raw_files if str(path).strip()]
        if normalized:
            return normalized

    args = cast(dict[str, Any], request.get("args") or {})
    raw_path = args.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        return [raw_path.strip()]
    return []


def _sanitized_args(request: dict[str, Any]) -> dict[str, Any]:
    args = dict(cast(dict[str, Any], request.get("args") or {}))
    tool = str(request.get("tool", ""))
    if tool == "write_file" and "content" in args:
        args["content"] = "(omitted; see diff_preview)"
    if tool == "apply_patch":
        if "patch" in args:
            args["patch"] = "(omitted; see diff_preview)"
        if "diff" in args:
            args["diff"] = "(omitted; see diff_preview)"
    return args


def _tool_summary(request: dict[str, Any]) -> str:
    tool = str(request.get("tool", "(unknown)"))
    affected_files = _normalize_affected_files(request)
    args = cast(dict[str, Any], request.get("args") or {})

    if tool == "write_file":
        path = affected_files[0] if affected_files else str(args.get("path", "(unknown)"))
        mode = str(args.get("mode", "overwrite") or "overwrite")
        return f"{tool} {path} [{mode}]"
    if tool == "apply_patch":
        if affected_files:
            preview = ", ".join(affected_files[:3])
            extra = "" if len(affected_files) <= 3 else f" (+{len(affected_files) - 3} more)"
            return f"{tool} on {preview}{extra}"
        return tool
    return tool


def _budget_remaining(
    config: AgentConfig,
    state: AgentState,
    budget_status: dict[str, Any],
) -> dict[str, int]:
    recovery_state = state.get("recovery_state")
    current_recovery_attempts = (
        _safe_int(cast(dict[str, Any], recovery_state).get("attempt_count"), 0)
        if isinstance(recovery_state, dict)
        else 0
    )
    return {
        "iterations_remaining": max(
            _safe_int(getattr(config, "max_iterations", 12))
            - _safe_int(state.get("iteration_count"), budget_status["iteration_count"]),
            0,
        ),
        "commands_remaining": max(
            _safe_int(getattr(config, "max_command_count", 8))
            - _safe_int(state.get("command_count"), budget_status["command_count"]),
            0,
        ),
        "runtime_remaining_seconds": max(
            _safe_int(getattr(config, "max_runtime_seconds", 900))
            - _safe_int(budget_status.get("elapsed_seconds"), 0),
            0,
        ),
        "plan_revisions_remaining": max(
            _safe_int(getattr(config, "max_plan_revisions", 3))
            - _safe_int(state.get("plan_revision_count"), 0),
            0,
        ),
        "recovery_attempts_remaining": max(
            _safe_int(getattr(config, "max_recovery_attempts_per_issue", 2))
            - current_recovery_attempts,
            0,
        ),
    }


def build_approval_view(
    state: AgentState,
    config: AgentConfig,
    *,
    state_path: Path | str | None = None,
) -> dict[str, Any]:
    request = cast(dict[str, Any] | None, state.get("pending_approval"))
    if request is None:
        raise ValueError("pending_approval is required to build an approval view")

    run_id = str(state["run_id"])
    budget_status = _default_budget_status(state)
    affected_files = _normalize_affected_files(request)
    snapshot_path = None if state_path is None else str(state_path)

    return {
        "run_id": run_id,
        "workspace_root": str(state.get("workspace_root", "")),
        "approval_request_id": str(request.get("id", "")),
        "tool": str(request.get("tool", "")),
        "tool_summary": _tool_summary(request),
        "risk_level": request.get("risk_level"),
        "args": _sanitized_args(request),
        "reason": str(request.get("reason", "")),
        "impact_summary": str(request.get("impact_summary", "")),
        "affected_files": affected_files,
        "diff_preview": request.get("diff_preview"),
        "backup_plan": request.get("backup_plan"),
        "rollback_command": request.get("rollback_command") or (f"--rollback-run {run_id}" if run_id else None),
        "suggested_verification_command": request.get("suggested_verification_command"),
        "budget_status": budget_status,
        "budget_remaining": _budget_remaining(config, state, budget_status),
        "recovery_state": state.get("recovery_state") if isinstance(state.get("recovery_state"), dict) else None,
        "plan_version": _safe_int(state.get("plan_version"), 0),
        "plan_steps": list(cast(list[dict[str, Any]], state.get("plan_steps", []))),
        "state_path": snapshot_path,
        "resume_approve_command": f"--resume-run {run_id} --approve",
        "resume_reject_command": f"--resume-run {run_id} --reject",
        "show_pending_command": f"--show-pending-run {run_id}",
        "approval_ui_mode": str(getattr(config, "approval_ui_mode", "compact")),
    }


def format_approval_view(view: dict[str, Any], *, mode: str = "compact") -> str:
    compact = mode != "detailed"
    lines = [
        "Approval Review",
        "===============",
        f"Run ID: {view.get('run_id')}",
        f"Approval ID: {view.get('approval_request_id')}",
        f"Tool: {view.get('tool_summary')}",
    ]
    if view.get("risk_level") is not None:
        lines.append(f"Risk Level: {view.get('risk_level')}")
    lines.extend(
        [
            f"Reason: {view.get('reason')}",
            f"Impact: {view.get('impact_summary')}",
        ]
    )

    affected_files = cast(list[str], view.get("affected_files") or [])
    if affected_files:
        lines.append("Affected Files: " + ", ".join(affected_files))

    diff_preview = view.get("diff_preview")
    if diff_preview:
        lines.append("Diff Preview:")
        lines.extend(f"  {line}" for line in str(diff_preview).splitlines())

    if view.get("backup_plan"):
        lines.append(f"Backup Plan: {view.get('backup_plan')}")
    if view.get("rollback_command"):
        lines.append(f"Rollback Command: {view.get('rollback_command')}")
    if view.get("suggested_verification_command"):
        lines.append(
            "Suggested Verification: " + str(view.get("suggested_verification_command"))
        )

    budget_remaining = cast(dict[str, Any], view.get("budget_remaining") or {})
    if budget_remaining:
        lines.append("Budget Remaining:")
        lines.append(
            "  iterations={iterations_remaining}, commands={commands_remaining}, runtime={runtime_remaining_seconds}s, "
            "plan_revisions={plan_revisions_remaining}, recovery_attempts={recovery_attempts_remaining}".format(
                **budget_remaining,
            )
        )

    recovery_state = cast(dict[str, Any] | None, view.get("recovery_state"))
    if recovery_state is not None:
        lines.append("Recovery State:")
        lines.append(
            "  issue_type={issue_type}, fingerprint={fingerprint}, attempts={attempt_count}, can_retry={can_retry}".format(
                **recovery_state,
            )
        )
        if recovery_state.get("last_action"):
            lines.append(f"  last_action={recovery_state['last_action']}")

    if not compact:
        args = view.get("args")
        if args is not None:
            lines.append("Args:")
            lines.extend(
                f"  {line}"
                for line in json.dumps(args, ensure_ascii=False, indent=2, default=str).splitlines()
            )
        plan_steps = cast(list[dict[str, Any]], view.get("plan_steps") or [])
        if plan_steps:
            lines.append(f"Plan Version: {view.get('plan_version')}")
            lines.append("Plan Steps:")
            for idx, step in enumerate(plan_steps, start=1):
                lines.append(
                    f"  {idx}. [{step.get('status', 'pending')}] {step.get('title', '(untitled)')}"
                )
        if view.get("state_path"):
            lines.append(f"State Path: {view.get('state_path')}")

    lines.append("Decision Commands:")
    lines.append(f"  Review Again: {view.get('show_pending_command')}")
    lines.append(
        "  Approve: "
        + str(view.get("resume_approve_command"))
        + " [--decision-note \"why this is safe\"]"
    )
    lines.append(
        "  Reject: "
        + str(view.get("resume_reject_command"))
        + " [--decision-note \"why this is rejected\"]"
    )

    return "\n".join(lines)