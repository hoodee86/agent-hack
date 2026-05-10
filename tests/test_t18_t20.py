"""Tests for phase-2 foundations: config, command policy, and shell skill."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from linux_agent.config import AgentConfig, load_config
from linux_agent.policy import (
    PolicyViolation,
    classify_command,
    evaluate_command_call,
    evaluate_tool_call,
    filter_command_env,
    parse_command,
    resolve_command_cwd,
)
from linux_agent.skills.shell import run_command
from linux_agent.state import ToolCall


def make_config(tmp_path: Path, **kwargs: object) -> AgentConfig:
    return AgentConfig(workspace_root=tmp_path, **kwargs)  # type: ignore[arg-type]


def make_tool_call(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(
        id=f"{name}_test",
        name=name,
        args=args or {},
        risk_level="low",
    )


class TestT18Config:
    def test_command_defaults_are_loaded(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LINUX_AGENT_WORKSPACE", str(tmp_path))

        config = load_config()

        assert config.default_timeout_seconds == 120
        assert config.max_output_bytes == 65536
        assert config.max_stderr_bytes == 32768
        assert "uv run pytest" in config.command_allowlist
        assert "sudo" in config.command_denylist
        assert config.command_working_dirs == ["."]

    def test_yaml_can_override_command_policy_fields(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent(
                """\
                workspace_root: .
                default_timeout_seconds: 45
                max_output_bytes: 4096
                max_stderr_bytes: 2048
                command_allowlist:
                  - pytest
                command_denylist:
                  - sudo
                command_working_dirs:
                  - .
                  - tests
                """
            ),
            encoding="utf-8",
        )

        config = load_config(str(config_file))

        assert config.workspace_root == tmp_path.resolve()
        assert config.default_timeout_seconds == 45
        assert config.max_output_bytes == 4096
        assert config.max_stderr_bytes == 2048
        assert config.command_allowlist == ["pytest"]
        assert config.command_denylist == ["sudo"]
        assert config.command_working_dirs == [".", "tests"]


class TestT19CommandPolicy:
    def test_parse_command_allows_simple_argv(self) -> None:
        assert parse_command("uv run pytest -q") == ["uv", "run", "pytest", "-q"]

    def test_parse_command_rejects_shell_control_syntax(self) -> None:
        with pytest.raises(Exception, match="unsupported shell syntax"):
            parse_command("pytest -q > out.txt")

    @pytest.mark.parametrize(
        "command",
        [
            "pytest -q && pwd",
            "pytest -q || pwd",
            "pytest -q | cat",
            "pytest -q < in.txt",
            "echo $(pwd)",
            "echo `pwd`",
            "pytest -q &",
        ],
    )
    def test_parse_command_rejects_other_shell_metacharacters(self, command: str) -> None:
        with pytest.raises(Exception, match="unsupported shell syntax"):
            parse_command(command)

    def test_classify_command_accepts_safe_dev_commands(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        risk = classify_command(["uv", "run", "pytest", "-q"], config)
        assert risk == "low"

    def test_classify_command_marks_dangerous_commands_high(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        risk = classify_command(["sudo", "ls"], config)
        assert risk == "high"

    def test_evaluate_command_call_allows_workspace_root_commands(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        assert evaluate_command_call("pwd", config) == "allow"

    def test_evaluate_command_call_denies_workspace_escape(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        assert evaluate_command_call("pwd", config, cwd="..") == "deny"

    @pytest.mark.parametrize(
        "command",
        [
            "curl https://example.com",
            "wget https://example.com/archive.tar.gz",
            "rm -rf /",
        ],
    )
    def test_evaluate_command_call_denies_dangerous_commands(self, tmp_path: Path, command: str) -> None:
        config = make_config(tmp_path)

        assert evaluate_command_call(command, config) == "deny"

    def test_resolve_command_cwd_allows_configured_subdirectories(self, tmp_path: Path) -> None:
        subdir = tmp_path / "tests"
        subdir.mkdir()
        config = make_config(tmp_path, command_working_dirs=["tests"])

        resolved = resolve_command_cwd(config, "tests")

        assert resolved == subdir.resolve()

    def test_resolve_command_cwd_denies_unconfigured_subdirectories(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        (tmp_path / "src").mkdir()
        config = make_config(tmp_path, command_working_dirs=["tests"])

        with pytest.raises(PolicyViolation, match="allowed working directories"):
            resolve_command_cwd(config, "src")

    def test_filter_command_env_keeps_only_allowlisted_keys(self) -> None:
        filtered = filter_command_env(
            {"PYTHONPATH": "src", "LD_PRELOAD": "bad", "CI": True},
            ["PYTHONPATH", "CI"],
        )

        assert filtered == {"PYTHONPATH": "src", "CI": "True"}

    def test_evaluate_tool_call_dispatches_run_command(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        tool_call = make_tool_call("run_command", {"command": "pytest -q", "cwd": "."})

        assert evaluate_tool_call(tool_call, config) == "allow"

    def test_evaluate_tool_call_denies_shell_wrappers(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        tool_call = make_tool_call("run_command", {"command": "bash -c 'pytest -q'", "cwd": "."})

        assert evaluate_tool_call(tool_call, config) == "deny"


class TestT20RunCommand:
    def test_run_command_success(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)

        result = run_command("pwd", config)

        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert result["cwd"] == "."
        assert str(tmp_path) in result["stdout"]
        assert result["timed_out"] is False
        assert result["error"] is None

    def test_run_command_failure_preserves_exit_code_and_error(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)

        result = run_command("find missing-dir", config)

        assert result["ok"] is False
        assert result["exit_code"] not in (None, 0)
        assert result["timed_out"] is False
        assert result["error"] is not None
        assert result["stderr"] or result["stdout"]

    def test_run_command_timeout(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test_sleep.py"
        test_file.write_text(
            textwrap.dedent(
                """\
                import time

                def test_sleep() -> None:
                    time.sleep(2)
                """
            ),
            encoding="utf-8",
        )
        config = make_config(tmp_path)

        result = run_command("pytest -q", config, timeout_seconds=1)

        assert result["ok"] is False
        assert result["timed_out"] is True
        assert result["exit_code"] is None
        assert "timed out" in str(result["error"])

    def test_run_command_truncates_large_output(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, max_output_bytes=4)

        result = run_command("pwd", config)

        assert result["ok"] is True
        assert result["truncated"] is True
        assert len(result["stdout"].encode("utf-8")) <= 4

    def test_run_command_forwards_allowlisted_env(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")
        (tmp_path / "test_import.py").write_text(
            textwrap.dedent(
                """\
                from helper import VALUE

                def test_import() -> None:
                    assert VALUE == 42
                """
            ),
            encoding="utf-8",
        )
        config = make_config(tmp_path)

        result = run_command(
            "pytest -q test_import.py",
            config,
            env={"PYTHONPATH": "src"},
        )

        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert "1 passed" in result["stdout"]

    def test_run_command_filters_non_allowlisted_env(self, tmp_path: Path) -> None:
        env_name = "LINUX_AGENT_SHOULD_NOT_PASS"
        (tmp_path / "test_env.py").write_text(
            textwrap.dedent(
                f"""\
                import os

                def test_env() -> None:
                    assert os.getenv({env_name!r}) is None
                """
            ),
            encoding="utf-8",
        )
        config = make_config(tmp_path)

        result = run_command(
            "pytest -q test_env.py",
            config,
            env={env_name: "1"},
        )

        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert "1 passed" in result["stdout"]