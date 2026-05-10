"""
Command-execution skill for the Linux Agent (phase 2 foundation).

The skill executes a tightly restricted command or command chain inside the
workspace using ``subprocess.run(..., shell=False)`` for each segment.
Policy errors raise ``PolicyViolation`` so the graph can distinguish them from
runtime failures.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import subprocess
import tempfile
import time
from typing import Any

from linux_agent.config import AgentConfig
from linux_agent.policy import (
    CommandSegment,
    CommandStage,
    command_segment_stages,
    filter_command_env,
    parse_command_sequence,
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


def _stage_payload(stage: CommandStage) -> dict[str, Any]:
    return {
        "command": stage["command"],
        "argv": stage["argv"],
    }


def _segment_payload(
    segment: CommandSegment,
    *,
    pipeline_stage_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stages = command_segment_stages(segment)
    payload: dict[str, Any] = {"command": segment["command"]}
    if len(stages) == 1:
        payload["argv"] = stages[0]["argv"]
    else:
        payload["pipeline_stages"] = (
            pipeline_stage_results
            if pipeline_stage_results is not None
            else [_stage_payload(stage) for stage in stages]
        )
    if segment["operator"] is not None:
        payload["operator"] = segment["operator"]
    return payload


def _should_run_segment(operator: str | None, previous_exit_code: int | None) -> bool:
    if operator is None or previous_exit_code is None:
        return True
    if operator == "&&":
        return previous_exit_code == 0
    if operator == "||":
        return previous_exit_code != 0
    return True


def _result_command_fields(segments: list[CommandSegment]) -> dict[str, Any]:
    if len(segments) == 1 and len(command_segment_stages(segments[0])) == 1:
        return {"argv": segments[0]["argv"]}
    return {}


def _read_temp_text(handle: Any) -> str:
    handle.seek(0)
    return str(handle.read())


def _terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                continue


def run_command(
    command: str,
    config: AgentConfig,
    *,
    cwd: str = ".",
    timeout_seconds: int | None = None,
    env: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Execute a policy-vetted command inside the workspace."""
    segments = parse_command_sequence(command)
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
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    segment_results: list[dict[str, Any]] = []
    last_exit_code: int | None = None

    for segment in segments:
        payload = _segment_payload(segment)
        stages = command_segment_stages(segment)
        if not _should_run_segment(segment["operator"], last_exit_code):
            segment_results.append(
                {**payload, "skipped": True, "exit_code": None, "timed_out": False}
            )
            continue

        remaining_timeout = effective_timeout - (time.monotonic() - started_at)
        if remaining_timeout <= 0:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            stdout, stdout_truncated = _truncate_output("".join(stdout_parts), config.max_output_bytes)
            stderr, stderr_truncated = _truncate_output("".join(stderr_parts), config.max_stderr_bytes)
            segment_results.append(
                {**payload, "skipped": False, "exit_code": None, "timed_out": True}
            )
            return {
                "ok": False,
                "command": command,
                "cwd": display_cwd,
                "exit_code": None,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": duration_ms,
                "timed_out": True,
                "truncated": stdout_truncated or stderr_truncated,
                "error": f"command timed out after {effective_timeout} seconds",
                "command_segments": segment_results,
                **_result_command_fields(segments),
            }

        if len(stages) == 1:
            try:
                proc = subprocess.run(
                    stages[0]["argv"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=remaining_timeout,
                    cwd=str(safe_cwd),
                    env=exec_env,
                    shell=False,
                )
            except subprocess.TimeoutExpired as exc:
                duration_ms = int((time.monotonic() - started_at) * 1000)
                stdout_parts.append(_coerce_stream_text(exc.stdout))
                stderr_parts.append(_coerce_stream_text(exc.stderr))
                stdout, stdout_truncated = _truncate_output("".join(stdout_parts), config.max_output_bytes)
                stderr, stderr_truncated = _truncate_output("".join(stderr_parts), config.max_stderr_bytes)
                segment_results.append(
                    {**payload, "skipped": False, "exit_code": None, "timed_out": True}
                )
                return {
                    "ok": False,
                    "command": command,
                    "cwd": display_cwd,
                    "exit_code": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "duration_ms": duration_ms,
                    "timed_out": True,
                    "truncated": stdout_truncated or stderr_truncated,
                    "error": f"command timed out after {effective_timeout} seconds",
                    "command_segments": segment_results,
                    **_result_command_fields(segments),
                }
            except OSError as exc:
                duration_ms = int((time.monotonic() - started_at) * 1000)
                segment_results.append(
                    {**payload, "skipped": False, "exit_code": None, "timed_out": False}
                )
                stdout, stdout_truncated = _truncate_output("".join(stdout_parts), config.max_output_bytes)
                stderr, stderr_truncated = _truncate_output("".join(stderr_parts), config.max_stderr_bytes)
                return {
                    "ok": False,
                    "command": command,
                    "cwd": display_cwd,
                    "exit_code": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "duration_ms": duration_ms,
                    "timed_out": False,
                    "truncated": stdout_truncated or stderr_truncated,
                    "error": f"failed to execute command: {exc}",
                    "command_segments": segment_results,
                    **_result_command_fields(segments),
                }

            stdout_parts.append(proc.stdout)
            stderr_parts.append(proc.stderr)
            last_exit_code = proc.returncode
            segment_results.append(
                {**payload, "skipped": False, "exit_code": proc.returncode, "timed_out": False}
            )
            continue

        stderr_files: list[Any] = []
        processes: list[subprocess.Popen[str]] = []
        previous_stdout: Any = None
        timed_out_stage_indexes: set[int] = set()
        final_stdout = ""
        try:
            for stage in stages:
                stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
                stderr_files.append(stderr_file)
                proc = subprocess.Popen(
                    stage["argv"],
                    stdin=previous_stdout,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(safe_cwd),
                    env=exec_env,
                    shell=False,
                )
                if previous_stdout is not None:
                    previous_stdout.close()
                previous_stdout = proc.stdout
                processes.append(proc)

            last_process = processes[-1]
            final_stdout, _ = last_process.communicate(timeout=remaining_timeout)
            for proc in processes[:-1]:
                proc.wait()
        except subprocess.TimeoutExpired:
            timed_out_stage_indexes = {
                index for index, proc in enumerate(processes) if proc.poll() is None
            }
            _terminate_processes(processes)
            if processes:
                try:
                    final_stdout, _ = processes[-1].communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    final_stdout = ""
            for proc in processes[:-1]:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    continue

            duration_ms = int((time.monotonic() - started_at) * 1000)
            stdout_parts.append(_coerce_stream_text(final_stdout))
            stderr_parts.append("".join(_read_temp_text(handle) for handle in stderr_files))
            pipeline_stage_results = [
                {
                    **_stage_payload(stage),
                    "exit_code": proc.returncode,
                    "timed_out": index in timed_out_stage_indexes,
                }
                for index, (stage, proc) in enumerate(zip(stages, processes))
            ]
            segment_results.append(
                {
                    **_segment_payload(segment, pipeline_stage_results=pipeline_stage_results),
                    "skipped": False,
                    "exit_code": None,
                    "timed_out": True,
                }
            )
            stdout, stdout_truncated = _truncate_output("".join(stdout_parts), config.max_output_bytes)
            stderr, stderr_truncated = _truncate_output("".join(stderr_parts), config.max_stderr_bytes)
            return {
                "ok": False,
                "command": command,
                "cwd": display_cwd,
                "exit_code": None,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": duration_ms,
                "timed_out": True,
                "truncated": stdout_truncated or stderr_truncated,
                "error": f"command timed out after {effective_timeout} seconds",
                "command_segments": segment_results,
                **_result_command_fields(segments),
            }
        except OSError as exc:
            _terminate_processes(processes)
            for proc in processes:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    continue

            duration_ms = int((time.monotonic() - started_at) * 1000)
            stderr_parts.append("".join(_read_temp_text(handle) for handle in stderr_files))
            pipeline_stage_results = [
                {
                    **_stage_payload(stage),
                    "exit_code": proc.returncode,
                    "timed_out": False,
                }
                for stage, proc in zip(stages, processes)
            ]
            segment_results.append(
                {
                    **_segment_payload(segment, pipeline_stage_results=pipeline_stage_results),
                    "skipped": False,
                    "exit_code": None,
                    "timed_out": False,
                }
            )
            stdout, stdout_truncated = _truncate_output("".join(stdout_parts), config.max_output_bytes)
            stderr, stderr_truncated = _truncate_output("".join(stderr_parts), config.max_stderr_bytes)
            return {
                "ok": False,
                "command": command,
                "cwd": display_cwd,
                "exit_code": None,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": duration_ms,
                "timed_out": False,
                "truncated": stdout_truncated or stderr_truncated,
                "error": f"failed to execute command: {exc}",
                "command_segments": segment_results,
                **_result_command_fields(segments),
            }
        finally:
            if previous_stdout is not None and not previous_stdout.closed:
                previous_stdout.close()
            stderr_text = "".join(_read_temp_text(handle) for handle in stderr_files)
            for handle in stderr_files:
                handle.close()

        stdout_parts.append(_coerce_stream_text(final_stdout))
        stderr_parts.append(stderr_text)
        last_exit_code = processes[-1].returncode
        pipeline_stage_results = [
            {
                **_stage_payload(stage),
                "exit_code": proc.returncode,
                "timed_out": False,
            }
            for stage, proc in zip(stages, processes)
        ]
        segment_results.append(
            {
                **_segment_payload(segment, pipeline_stage_results=pipeline_stage_results),
                "skipped": False,
                "exit_code": last_exit_code,
                "timed_out": False,
            }
        )

    duration_ms = int((time.monotonic() - started_at) * 1000)
    stdout, stdout_truncated = _truncate_output("".join(stdout_parts), config.max_output_bytes)
    stderr, stderr_truncated = _truncate_output("".join(stderr_parts), config.max_stderr_bytes)
    timed_out = False
    truncated = stdout_truncated or stderr_truncated
    ok = last_exit_code == 0
    error = None if ok else f"command exited with status {last_exit_code}"

    return {
        "ok": ok,
        "command": command,
        "cwd": display_cwd,
        "exit_code": last_exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": duration_ms,
        "timed_out": timed_out,
        "truncated": truncated,
        "error": error,
        "command_segments": segment_results,
        **_result_command_fields(segments),
    }