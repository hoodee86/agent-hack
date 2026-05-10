"""
CLI entry point for the Linux Agent (T15).

Usage
-----
    # Use config.yaml (workspace_root must be set inside)
    uv run python -m linux_agent.app "List all Python files" --config config.yaml

    # Override workspace via flag (no config.yaml needed)
    uv run python -m linux_agent.app "What is in README.md?" --workspace ./my_repo

    # Both flags: config.yaml provides defaults, --workspace overrides workspace_root
    uv run python -m linux_agent.app "Find TODO comments" \\
        --config config.yaml --workspace /tmp/project --verbose

Exit codes
----------
    0  Agent finished and produced a final answer.
    1  Configuration error or unexpected exception.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from linux_agent.audit import EVENT_RUN_START, AuditLogger
from linux_agent.config import AgentConfig, load_config
from linux_agent.graph import build_graph
from linux_agent.state import AgentState


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linux-agent",
        description="Read-only Linux filesystem agent powered by LangGraph.",
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
        help="Print each step's summary to stderr as the agent runs.",
    )
    return parser


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

    # ── Run ──────────────────────────────────────────────────────────────────
    run_id = str(uuid.uuid4())

    if args.verbose:
        print(
            f"[linux-agent] run_id={run_id}  workspace={config.workspace_root}",
            file=sys.stderr,
        )

    # Write run_start audit event
    with AuditLogger(run_id, config.log_dir) as audit:
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
        iteration_count=0,
        consecutive_failures=0,
        final_answer=None,
    )

    try:
        app = build_graph(config)
        final_state: AgentState = app.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        print(f"[linux-agent] Runtime error: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(
            f"[linux-agent] Done after {final_state['iteration_count']} iteration(s).",
            file=sys.stderr,
        )

    print(final_state.get("final_answer") or "(no answer produced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

