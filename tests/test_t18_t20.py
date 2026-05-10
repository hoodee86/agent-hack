"""Tests for phase-2 foundations: config, command policy, and shell skill."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shlex
import shutil
import threading
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
    parse_command_sequence,
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
        assert "cat" in config.command_allowlist
        assert "grep" in config.command_allowlist
        assert "curl -s" in config.command_allowlist
        assert "curl -sL" in config.command_allowlist
        assert "wc -c" in config.command_allowlist
        assert "uniq" in config.command_allowlist
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

    def test_parse_command_sequence_allows_compound_commands(self) -> None:
        segments = parse_command_sequence("pwd && echo done ; ls")

        assert segments == [
            {"operator": None, "argv": ["pwd"], "command": "pwd"},
            {"operator": "&&", "argv": ["echo", "done"], "command": "echo done"},
            {"operator": ";", "argv": ["ls"], "command": "ls"},
        ]

    def test_parse_command_sequence_allows_pipeline_segments(self) -> None:
        segments = parse_command_sequence("pwd | cat && echo done")

        assert segments == [
            {
                "operator": None,
                "argv": ["pwd"],
                "command": "pwd | cat",
                "stages": [
                    {"argv": ["pwd"], "command": "pwd"},
                    {"argv": ["cat"], "command": "cat"},
                ],
            },
            {"operator": "&&", "argv": ["echo", "done"], "command": "echo done"},
        ]

    def test_parse_command_sequence_allows_stderr_merge_redirect(self) -> None:
        segments = parse_command_sequence("curl -s https://example.com 2>&1 | head -200")

        assert segments == [
            {
                "operator": None,
                "argv": ["curl", "-s", "https://example.com"],
                "command": "curl -s https://example.com 2>&1 | head -200",
                "stages": [
                    {
                        "argv": ["curl", "-s", "https://example.com"],
                        "command": "curl -s https://example.com 2>&1",
                        "merge_stderr": True,
                    },
                    {"argv": ["head", "-200"], "command": "head -200"},
                ],
            }
        ]

    def test_parse_command_sequence_allows_stdout_redirect(self) -> None:
        segments = parse_command_sequence("curl -s https://example.com 2>&1 > out.json && wc -c out.json")

        assert segments == [
            {
                "operator": None,
                "argv": ["curl", "-s", "https://example.com"],
                "command": "curl -s https://example.com 2>&1 > out.json",
                "stages": [
                    {
                        "argv": ["curl", "-s", "https://example.com"],
                        "command": "curl -s https://example.com 2>&1 > out.json",
                        "merge_stderr": True,
                        "stdout_path": "out.json",
                    }
                ],
            },
            {"operator": "&&", "argv": ["wc", "-c", "out.json"], "command": "wc -c out.json"},
        ]

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

    def test_evaluate_command_call_allows_allowlisted_compound_command(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["pwd", "echo"],
            command_denylist=[],
        )

        assert evaluate_command_call("pwd && echo done", config) == "allow"

    def test_evaluate_command_call_allows_allowlisted_pipeline(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["pwd", "cat"],
            command_denylist=[],
        )

        assert evaluate_command_call("pwd | cat", config) == "allow"

    def test_evaluate_command_call_denies_compound_command_if_any_segment_is_high_risk(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["pwd"],
            command_denylist=["rm"],
        )

        assert evaluate_command_call("pwd && rm -rf tmp", config) == "deny"

    def test_evaluate_command_call_allows_allowlisted_curl_prefix(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["curl -s"],
            command_denylist=["curl"],
        )

        assert evaluate_command_call("curl -s https://example.com", config) == "allow"

    def test_evaluate_command_call_allows_allowlisted_curl_with_stderr_merge(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["curl -s", "head"],
            command_denylist=["curl"],
        )

        assert (
            evaluate_command_call(
                "curl -s -L --max-time 15 https://example.com 2>&1 | head -200",
                config,
            )
            == "allow"
        )

    def test_evaluate_command_call_allows_allowlisted_grep_pipeline(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["curl -s", "grep", "head"],
            command_denylist=["curl"],
        )

        assert (
            evaluate_command_call(
                "curl -s -L --max-time 15 https://example.com 2>&1 | grep -oP 'title' | head -50",
                config,
            )
            == "allow"
        )

    def test_evaluate_command_call_allows_safe_inline_python_json_reader(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        (tmp_path / "baidu_hot.json").write_text(
            '{"data": {"cards": [{"component": "hotList", "content": [{"index": 0, "word": "A", "desc": "B", "hotScore": "1"}]}]}}',
            encoding="utf-8",
        )
        code = textwrap.dedent(
            """\
            import json
            with open('baidu_hot.json', 'r') as f:
                data = json.load(f)
            cards = data['data']['cards']
            print(len(cards))
            """
        ).strip()

        assert evaluate_command_call(shlex.join(["python3", "-c", code]), config) == "allow"

    def test_evaluate_command_call_denies_inline_python_workspace_escape(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        code = textwrap.dedent(
            """\
            import json
            with open('../outside.json', 'r') as f:
                data = json.load(f)
            print(len(data))
            """
        ).strip()

        assert evaluate_command_call(shlex.join(["python3", "-c", code]), config) == "deny"

    def test_evaluate_command_call_denies_inline_python_write_mode(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        code = textwrap.dedent(
            """\
            with open('out.txt', 'w') as f:
                print('hello')
            """
        ).strip()

        assert evaluate_command_call(shlex.join(["python3", "-c", code]), config) == "deny"

    def test_evaluate_command_call_denies_inline_python_unsafe_import(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        code = "import os\nprint(os.listdir('.'))"

        assert evaluate_command_call(shlex.join(["python3", "-c", code]), config) == "deny"

    def test_evaluate_command_call_allows_allowlisted_compact_curl_flags(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["curl -sL", "head"],
            command_denylist=["curl"],
        )

        assert (
            evaluate_command_call(
                "curl -sL -o /tmp/out.html --max-time 15 https://example.com 2>&1 && head -c 200 /tmp/out.html",
                config,
            )
            == "allow"
        )

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

    def test_evaluate_tool_call_requires_approval_for_run_command_file_output(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["curl -sL", "head"],
            command_denylist=["curl"],
        )
        tool_call = make_tool_call(
            "run_command",
            {"command": "curl -sL -o out.html --max-time 15 https://example.com 2>&1 && head -c 200 out.html", "cwd": "."},
        )

        assert evaluate_tool_call(tool_call, config) == "needs_approval"

    def test_evaluate_tool_call_requires_approval_for_run_command_stdout_redirect(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["curl -sL", "wc -c"],
            command_denylist=["curl"],
        )
        tool_call = make_tool_call(
            "run_command",
            {"command": "curl -sL --max-time 15 https://example.com > out.json && wc -c out.json", "cwd": "."},
        )

        assert evaluate_tool_call(tool_call, config) == "needs_approval"

    def test_evaluate_tool_call_denies_run_command_file_output_outside_workspace(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["curl -sL", "head"],
            command_denylist=["curl"],
        )
        outside_path = (tmp_path.parent / "outside.html").resolve()
        tool_call = make_tool_call(
            "run_command",
            {"command": f"curl -sL -o {outside_path} --max-time 15 https://example.com 2>&1 && head -c 200 {outside_path}", "cwd": "."},
        )

        assert evaluate_tool_call(tool_call, config) == "deny"

    def test_evaluate_tool_call_denies_run_command_stdout_redirect_outside_workspace(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["curl -sL", "wc -c"],
            command_denylist=["curl"],
        )
        outside_path = (tmp_path.parent / "outside.json").resolve()
        tool_call = make_tool_call(
            "run_command",
            {"command": f"curl -sL --max-time 15 https://example.com > {outside_path} && wc -c {outside_path}", "cwd": "."},
        )

        assert evaluate_tool_call(tool_call, config) == "deny"

    def test_evaluate_tool_call_allows_run_command_file_output_when_write_approval_disabled(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["curl -sL", "head"],
            command_denylist=["curl"],
            write_requires_approval=False,
        )
        tool_call = make_tool_call(
            "run_command",
            {"command": "curl -sL -o out.html --max-time 15 https://example.com 2>&1 && head -c 200 out.html", "cwd": "."},
        )

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

    def test_run_command_executes_compound_command_segments_in_order(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["pwd", "echo"],
            command_denylist=[],
        )

        result = run_command("pwd && echo done", config)

        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert str(tmp_path) in result["stdout"]
        assert "done" in result["stdout"]
        assert [segment["exit_code"] for segment in result["command_segments"]] == [0, 0]

    def test_run_command_executes_pipeline(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["pwd", "cat"],
            command_denylist=[],
        )

        result = run_command("pwd | cat", config)

        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert str(tmp_path) in result["stdout"]
        assert len(result["command_segments"]) == 1
        assert [stage["exit_code"] for stage in result["command_segments"][0]["pipeline_stages"]] == [0, 0]

    def test_run_command_skips_and_segment_after_failure(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["find", "echo"],
            command_denylist=[],
        )

        result = run_command("find missing-dir && echo done", config)

        assert result["ok"] is False
        assert result["exit_code"] not in (None, 0)
        assert result["command_segments"][0]["skipped"] is False
        assert result["command_segments"][1]["skipped"] is True

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

    @pytest.mark.skipif(shutil.which("curl") is None, reason="curl is not installed")
    def test_run_command_supports_allowlisted_curl(self, tmp_path: Path) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                body = b"linux-agent-curl-ok\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            config = make_config(
                tmp_path,
                command_allowlist=["curl -s"],
                command_denylist=["curl"],
            )

            result = run_command(f"curl -s http://127.0.0.1:{port}/", config)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert "linux-agent-curl-ok" in result["stdout"]

    @pytest.mark.skipif(shutil.which("curl") is None, reason="curl is not installed")
    def test_run_command_supports_allowlisted_curl_pipeline(self, tmp_path: Path) -> None:
        body = b"linux-agent-curl-pipe\n"

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            config = make_config(
                tmp_path,
                command_allowlist=["curl -s", "wc -c"],
                command_denylist=["curl"],
            )

            result = run_command(f"curl -s http://127.0.0.1:{port}/ | wc -c", config)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert result["ok"] is True

    def test_run_command_supports_stdout_redirect(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            command_allowlist=["echo", "wc -c"],
            command_denylist=[],
            write_requires_approval=False,
        )

        result = run_command("echo hello > out.txt && wc -c out.txt", config)

        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello\n"
        assert "out.txt" in result["command_segments"][0]["command"]
        assert "6" in result["stdout"]

    @pytest.mark.skipif(shutil.which("curl") is None, reason="curl is not installed")
    def test_run_command_supports_allowlisted_curl_pipeline_with_stderr_merge(self, tmp_path: Path) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                body = b"linux-agent-curl-merged\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            config = make_config(
                tmp_path,
                command_allowlist=["curl -s", "head"],
                command_denylist=["curl"],
            )

            result = run_command(
                f"curl -s http://127.0.0.1:{port}/ 2>&1 | head -200",
                config,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert "linux-agent-curl-merged" in result["stdout"]
        assert result["stderr"] == ""
        assert [stage["exit_code"] for stage in result["command_segments"][0]["pipeline_stages"]] == [0, 0]