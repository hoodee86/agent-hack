"""
CLI entry point for the Linux Agent (T15).

Usage
-----
    # Use config.yaml (workspace_root must be set inside)
    uv run python -m linux_agent.app "List all Python files" --config config.yaml

    # Override workspace via flag (no config.yaml needed)
    uv run python -m linux_agent.app "What is in README.md?" --workspace ./my_repo

    # Both flags: config.yaml provides defaults, --workspace overrides workspace_root
    uv run python -m linux_agent.app "Find TODO comments" \
        --config config.yaml --workspace /tmp/project --verbose

    # Dump the exact message transcript sent to the model each iteration
    uv run python -m linux_agent.app "Find TODO comments" \
        --config config.yaml --verbose --show-prompts

Exit codes
----------
    0  Agent finished and produced a final answer.
    1  Configuration error or unexpected exception.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any, TextIO

from linux_agent.audit import (
    AuditEvent,
    AuditEventListener,
    AuditLogger,
    EVENT_MODEL_INPUT,
    EVENT_PLAN_UPDATE,
    EVENT_POLICY_DECISION,
    EVENT_REFLECTOR_ACTION,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_TOOL_PROPOSED,
    EVENT_TOOL_RESULT,
)
from linux_agent.config import AgentConfig, load_config
from linux_agent.graph import build_graph
from linux_agent.state import AgentState


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linux-agent",
        description="Controlled Linux workspace agent powered by LangGraph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "goal",
        help="Natural-language goal for the agent (e.g. 'List all Python files').",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a YAML config file. When omitted, built-in defaults are used.",
    )
    parser.add_argument(
        "--workspace",
        metavar="PATH",
        default=None,
        help=(
            "Override workspace_root. Must be an existing directory. "
            "Takes precedence over the value in --config."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed per-step model, policy, and tool events to stderr.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        default=False,
        help=(
            "Also print each planner round's full system/history/user message "
            "sequence to stderr. Implies --verbose."
        ),
    )
    return parser


class _Ansi:
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def _supports_color(stream: TextIO) -> bool:
    is_tty = getattr(stream, "isatty", lambda: False)()
    return bool(
        is_tty
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM") not in {None, "", "dumb"}
    )


def _style(text: str, *codes: str, enabled: bool) -> str:
    if not enabled or not codes:
        return text
    return f"{''.join(codes)}{text}{_Ansi.RESET}"


def _render_value(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "(empty)"
        if stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return value
            return json.dumps(parsed, ensure_ascii=False, indent=2, default=str)
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _print_block(stream: TextIO, text: str, indent: str = "  ") -> None:
    for line in textwrap.indent(text, indent).splitlines() or [indent.rstrip()]:
        print(line, file=stream)


def _print_field(
    stream: TextIO,
    label: str,
    value: Any,
    *,
    use_color: bool,
    inline: bool = False,
) -> None:
    rendered = _render_value(value)
    label_text = _style(f"{label}:", _Ansi.BOLD, enabled=use_color)
    if inline and "\n" not in rendered:
        print(f"{label_text} {rendered}", file=stream)
        return
    print(label_text, file=stream)
    _print_block(stream, rendered)


def _print_plan(stream: TextIO, plan: list[str], *, use_color: bool) -> None:
    label_text = _style("Plan:", _Ansi.BOLD, enabled=use_color)
    print(label_text, file=stream)
    if not plan:
        print("  (empty)", file=stream)
        return
    for idx, step in enumerate(plan, start=1):
        print(f"  {idx}. {step}", file=stream)


def _print_section(stream: TextIO, title: str, *, color: str, use_color: bool) -> None:
    header = f"[linux-agent] {title}"
    line = "=" * max(0, 79 - len(header))
    print(
        _style(f"{header} {line}".rstrip(), _Ansi.BOLD, color, enabled=use_color),
        file=stream,
    )


def _format_role(role: str) -> str:
    return {
        "ai": "assistant",
        "human": "human",
        "system": "system",
        "tool": "tool",
    }.get(role, role)


def _make_verbose_event_printer(stream: TextIO) -> AuditEventListener:
    use_color = _supports_color(stream)
    counters = {"iteration": 0}

    def _iteration_for(record: AuditEvent) -> int | None:
        event = record["event"]
        if event == EVENT_MODEL_INPUT:
            return counters["iteration"] + 1
        if event == EVENT_PLAN_UPDATE:
            counters["iteration"] += 1
            return counters["iteration"]
        if counters["iteration"] and event in {
            EVENT_TOOL_PROPOSED,
            EVENT_POLICY_DECISION,
            EVENT_TOOL_RESULT,
            EVENT_REFLECTOR_ACTION,
        }:
            return counters["iteration"]
        return None

    def _title(record: AuditEvent, iteration: int | None) -> str:
        event = record["event"]
        if event == EVENT_RUN_START:
            return "Run Start"
        if event == EVENT_RUN_END:
            return "Run End"
        if event == EVENT_MODEL_INPUT:
            return f"Iteration {iteration} | Model Input"
        if event == EVENT_PLAN_UPDATE:
            return f"Iteration {iteration} | Planner"
        if event == EVENT_TOOL_PROPOSED:
            return f"Iteration {iteration} | Tool Proposal"
        if event == EVENT_POLICY_DECISION:
            return f"Iteration {iteration} | Policy Guard"
        if event == EVENT_TOOL_RESULT:
            return f"Iteration {iteration} | Tool Result"
        if event == EVENT_REFLECTOR_ACTION:
            return f"Iteration {iteration} | Reflector"
        return record["event"].replace("_", " ").title()

    def _color(record: AuditEvent) -> str:
        event = record["event"]
        if event == EVENT_MODEL_INPUT:
            return _Ansi.MAGENTA
        if event == EVENT_PLAN_UPDATE:
            return _Ansi.BLUE
        if event == EVENT_TOOL_PROPOSED:
            return _Ansi.YELLOW
        if event == EVENT_POLICY_DECISION:
            decision = record["data"].get("decision")
            if decision == "allow":
                return _Ansi.GREEN
            if decision == "needs_approval":
                return _Ansi.YELLOW
            return _Ansi.RED
        if event == EVENT_TOOL_RESULT:
            return _Ansi.GREEN if record["data"].get("ok") else _Ansi.RED
        if event == EVENT_REFLECTOR_ACTION:
            return _Ansi.YELLOW
        if event == EVENT_RUN_END:
            return _Ansi.GREEN
        return _Ansi.CYAN

    def _print_messages(messages: list[dict[str, Any]]) -> None:
        for idx, message in enumerate(messages, start=1):
            role_header = _style(
                f"[{idx}] {_format_role(str(message.get('type', 'message')))}",
                _Ansi.BOLD,
                enabled=use_color,
            )
            print(role_header, file=stream)
            if "content" in message:
                _print_field(stream, "Content", message["content"], use_color=use_color)
            if "tool_calls" in message:
                _print_field(stream, "Tool Calls", message["tool_calls"], use_color=use_color)
            if "tool_call_id" in message:
                _print_field(
                    stream,
                    "Tool Call ID",
                    message["tool_call_id"],
                    use_color=use_color,
                    inline=True,
                )
            if idx != len(messages):
                print(file=stream)

    def _print_command_fields(data: dict[str, Any]) -> None:
        _print_field(
            stream,
            "Command",
            data.get("command"),
            use_color=use_color,
            inline=True,
        )
        _print_field(
            stream,
            "Working Directory",
            data.get("cwd", "."),
            use_color=use_color,
            inline=True,
        )
        if data.get("argv") is not None:
            _print_field(stream, "Argv", data.get("argv"), use_color=use_color)
        if data.get("timeout_seconds") is not None:
            _print_field(
                stream,
                "Timeout",
                f"{data.get('timeout_seconds')}s",
                use_color=use_color,
                inline=True,
            )
        if data.get("env_keys"):
            _print_field(stream, "Env Keys", data.get("env_keys"), use_color=use_color)

    def _print_record(record: AuditEvent, iteration: int | None) -> None:
        data = record["data"]
        event = record["event"]
        _print_section(stream, _title(record, iteration), color=_color(record), use_color=use_color)

        if event == EVENT_RUN_START:
            _print_field(stream, "Run ID", record["run_id"], use_color=use_color, inline=True)
            _print_field(stream, "Goal", data.get("user_goal"), use_color=use_color)
            _print_field(stream, "Workspace", data.get("workspace_root"), use_color=use_color, inline=True)
            _print_field(stream, "Config", data.get("config"), use_color=use_color)
        elif event == EVENT_MODEL_INPUT:
            _print_field(stream, "Message Count", data.get("message_count"), use_color=use_color, inline=True)
            _print_messages(list(data.get("messages", [])))
        elif event == EVENT_PLAN_UPDATE:
            _print_field(stream, "Step", data.get("current_step"), use_color=use_color)
            _print_plan(stream, list(data.get("plan", [])), use_color=use_color)
            _print_field(stream, "Assistant", data.get("assistant_content"), use_color=use_color)
            if data.get("final_answer") is not None:
                _print_field(stream, "Final Answer", data.get("final_answer"), use_color=use_color)
        elif event == EVENT_TOOL_PROPOSED:
            _print_field(stream, "Tool", data.get("tool"), use_color=use_color, inline=True)
            if data.get("risk_level") is not None:
                _print_field(stream, "Risk Level", data.get("risk_level"), use_color=use_color, inline=True)
            _print_field(stream, "Call ID", data.get("tool_call_id"), use_color=use_color, inline=True)
            if data.get("tool") == "run_command":
                _print_command_fields(data)
            else:
                _print_field(stream, "Args", data.get("args"), use_color=use_color)
        elif event == EVENT_POLICY_DECISION:
            _print_field(stream, "Tool", data.get("tool"), use_color=use_color, inline=True)
            _print_field(stream, "Decision", data.get("decision"), use_color=use_color, inline=True)
            if data.get("risk_level") is not None:
                _print_field(stream, "Risk Level", data.get("risk_level"), use_color=use_color, inline=True)
            _print_field(stream, "Call ID", data.get("tool_call_id"), use_color=use_color, inline=True)
            if data.get("tool") == "run_command":
                _print_command_fields(data)
            else:
                _print_field(stream, "Args", data.get("args"), use_color=use_color)
            if data.get("reason") is not None:
                _print_field(stream, "Reason", data.get("reason"), use_color=use_color)
            if data.get("impact_summary") is not None:
                _print_field(stream, "Impact Summary", data.get("impact_summary"), use_color=use_color)
            if data.get("backup_plan") is not None:
                _print_field(stream, "Backup Plan", data.get("backup_plan"), use_color=use_color)
        elif event == EVENT_TOOL_RESULT:
            status = "ok" if data.get("ok") else "error"
            _print_field(stream, "Tool", data.get("tool"), use_color=use_color, inline=True)
            _print_field(stream, "Status", status, use_color=use_color, inline=True)
            if data.get("risk_level") is not None:
                _print_field(stream, "Risk Level", data.get("risk_level"), use_color=use_color, inline=True)
            _print_field(stream, "Call ID", data.get("tool_call_id"), use_color=use_color, inline=True)
            _print_field(stream, "Duration", f"{data.get('duration_ms', 0)}ms", use_color=use_color, inline=True)
            if data.get("tool") == "run_command":
                _print_command_fields(data)
                if data.get("exit_code") is not None:
                    _print_field(stream, "Exit Code", data.get("exit_code"), use_color=use_color, inline=True)
                if data.get("timed_out") is not None:
                    _print_field(stream, "Timed Out", data.get("timed_out"), use_color=use_color, inline=True)
                if data.get("truncated") is not None:
                    _print_field(stream, "Truncated", data.get("truncated"), use_color=use_color, inline=True)
                if data.get("stderr_preview") is not None:
                    _print_field(stream, "Stderr", data.get("stderr_preview"), use_color=use_color)
                if data.get("stdout_preview") is not None:
                    _print_field(stream, "Stdout", data.get("stdout_preview"), use_color=use_color)
            if data.get("error") is not None:
                _print_field(stream, "Error", data.get("error"), use_color=use_color)
            _print_field(stream, "Result", data.get("result"), use_color=use_color)
        elif event == EVENT_REFLECTOR_ACTION:
            for key, value in data.items():
                _print_field(stream, key.replace("_", " ").title(), value, use_color=use_color)
        elif event == EVENT_RUN_END:
            _print_field(stream, "Iterations", data.get("iteration_count"), use_color=use_color, inline=True)
            _print_field(stream, "Observations", data.get("observation_count"), use_color=use_color, inline=True)
            if data.get("command_count") is not None:
                _print_field(stream, "Commands", data.get("command_count"), use_color=use_color, inline=True)
            if data.get("command_summaries"):
                _print_field(stream, "Command Summaries", data.get("command_summaries"), use_color=use_color)
            _print_field(stream, "Final Answer", data.get("final_answer"), use_color=use_color)
        else:
            _print_field(stream, "Data", data, use_color=use_color)

        print(file=stream)

    def _emit(record: AuditEvent) -> None:
        iteration = _iteration_for(record)
        _print_record(record, iteration)

    return _emit


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Load config ──────────────────────────────────────────────────────────
    try:
        config: AgentConfig = load_config(args.config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[linux-agent] Configuration error: {exc}", file=sys.stderr)
        return 1

    # ── Override workspace_root if --workspace was given ─────────────────────
    if args.workspace is not None:
        ws_path = Path(args.workspace).expanduser().resolve()
        if not ws_path.exists() or not ws_path.is_dir():
            print(
                f"[linux-agent] --workspace '{args.workspace}' does not exist "
                "or is not a directory.",
                file=sys.stderr,
            )
            return 1
        # Re-create config with the overridden workspace_root
        config = AgentConfig(
            **{
                **config.model_dump(),
                "workspace_root": ws_path,
            }
        )

    verbose_enabled = args.verbose or args.show_prompts
    verbose_listener = _make_verbose_event_printer(sys.stderr) if verbose_enabled else None
    prompt_trace_listener = verbose_listener if args.show_prompts else None

    # ── Run ──────────────────────────────────────────────────────────────────
    run_id = str(uuid.uuid4())

    # Write run_start audit event
    with AuditLogger(run_id, config.log_dir, listener=verbose_listener) as audit:
        audit.log(
            EVENT_RUN_START,
            {
                "user_goal": args.goal,
                "workspace_root": str(config.workspace_root),
                "config": {
                    "max_iterations": config.max_iterations,
                    "max_consecutive_failures": config.max_consecutive_failures,
                    "llm_model": config.llm_model,
                },
            },
        )

    initial_state = AgentState(
        run_id=run_id,
        user_goal=args.goal,
        workspace_root=str(config.workspace_root),
        messages=[],
        plan=[],
        current_step=None,
        proposed_tool_call=None,
        observations=[],
        risk_decision=None,
        pending_approval=None,
        iteration_count=0,
        consecutive_failures=0,
        final_answer=None,
    )

    try:
        app = build_graph(
            config,
            event_listener=verbose_listener,
            prompt_trace_listener=prompt_trace_listener,
        )
        final_state: AgentState = app.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        print(f"[linux-agent] Runtime error: {exc}", file=sys.stderr)
        return 1

    print(final_state.get("final_answer") or "(no answer produced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

