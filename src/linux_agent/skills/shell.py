"""
Command-execution skill for the Linux Agent (phase 2 foundation).

The skill executes a tightly restricted command inside the workspace using
``subprocess.run(..., shell=False)``. Policy errors raise ``PolicyViolation``
so the graph can distinguish them from runtime failures.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import subprocess
import time
from typing import Any

from linux_agent.config import AgentConfig
from linux_agent.policy import (
    filter_command_env,
    parse_command,
    resolve_command_cwd,
)


def _truncate_output(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0:
        return "", bool(text)

    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False

    truncated = encoded[:limit].decode("utf-8", errors="ignore")
    return truncated, True


def _coerce_stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _display_cwd(config: AgentConfig, resolved_cwd: os.PathLike[str] | str) -> str:
    cwd_path = resolved_cwd if isinstance(resolved_cwd, str) else os.fspath(resolved_cwd)
    try:
        rel = os.path.relpath(cwd_path, config.workspace_root)
    except ValueError:
        return str(cwd_path)
    return "." if rel == "." else rel


def run_command(
    command: str,
    config: AgentConfig,
    *,
    cwd: str = ".",
    timeout_seconds: int | None = None,
    env: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Execute a policy-vetted command inside the workspace."""
    argv = parse_command(command)
    safe_cwd = resolve_command_cwd(config, cwd)
    effective_timeout = timeout_seconds or config.default_timeout_seconds
    if effective_timeout <= 0:
        raise ValueError("timeout_seconds must be positive")

    filtered_env = filter_command_env(env, config.command_env_allowlist)
    exec_env: dict[str, str] | None = None
    if filtered_env:
        exec_env = os.environ.copy()
        exec_env.update(filtered_env)

    started_at = time.monotonic()
    display_cwd = _display_cwd(config, safe_cwd)

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=effective_timeout,
            cwd=str(safe_cwd),
            env=exec_env,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - started_at) * 1000)
        stdout, stdout_truncated = _truncate_output(
            _coerce_stream_text(exc.stdout),
            config.max_output_bytes,
        )
        stderr, stderr_truncated = _truncate_output(
            _coerce_stream_text(exc.stderr),
            config.max_stderr_bytes,
        )
        return {
            "ok": False,
            "command": command,
            "argv": argv,
            "cwd": display_cwd,
            "exit_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "duration_ms": duration_ms,
            "timed_out": True,
            "truncated": stdout_truncated or stderr_truncated,
            "error": f"command timed out after {effective_timeout} seconds",
        }
    except OSError as exc:
        duration_ms = int((time.monotonic() - started_at) * 1000)
        return {
            "ok": False,
            "command": command,
            "argv": argv,
            "cwd": display_cwd,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "duration_ms": duration_ms,
            "timed_out": False,
            "truncated": False,
            "error": f"failed to execute command: {exc}",
        }

    duration_ms = int((time.monotonic() - started_at) * 1000)
    stdout, stdout_truncated = _truncate_output(proc.stdout, config.max_output_bytes)
    stderr, stderr_truncated = _truncate_output(proc.stderr, config.max_stderr_bytes)
    timed_out = False
    truncated = stdout_truncated or stderr_truncated
    ok = proc.returncode == 0
    error = None if ok else f"command exited with status {proc.returncode}"

    return {
        "ok": ok,
        "command": command,
        "argv": argv,
        "cwd": display_cwd,
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": duration_ms,
        "timed_out": timed_out,
        "truncated": truncated,
        "error": error,
    }