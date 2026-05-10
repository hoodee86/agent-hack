"""
Configuration loading for the Linux Agent.

AgentConfig holds all tunable parameters. load_config() resolves values
from a YAML file (optional) or environment variables, then validates them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class AgentConfig(BaseModel):
    """All runtime configuration for a single agent session."""

    # --- Workspace -------------------------------------------------------
    # Absolute path to the sandboxed directory the agent may access
    workspace_root: Path

    # --- Iteration limits ------------------------------------------------
    # Maximum number of tool executions before the agent is forced to stop
    max_iterations: int = Field(default=12, ge=1)
    # Circuit-breaker: stop after this many consecutive failures
    max_consecutive_failures: int = Field(default=3, ge=1)

    # --- Skill limits ----------------------------------------------------
    # Maximum bytes returned by read_file in a single call
    max_read_bytes: int = Field(default=65536, ge=1)
    # Maximum number of search matches returned by search_text
    max_search_results: int = Field(default=100, ge=1)
    # Maximum number of directory entries returned by list_dir
    max_list_entries: int = Field(default=200, ge=1)

    # --- Command execution limits ---------------------------------------
    # Default timeout applied by run_command when the caller omits one
    default_timeout_seconds: int = Field(default=120, ge=1)
    # Hard limit on how many shell commands a run may execute in total.
    max_command_count: int = Field(default=8, ge=1)
    # Hard cap on total wall-clock runtime for a single run.
    max_runtime_seconds: int = Field(default=900, ge=1)
    # Maximum number of plan revisions allowed before the run must stop.
    max_plan_revisions: int = Field(default=3, ge=0)
    # Maximum automatic recovery attempts per distinct failure fingerprint.
    max_recovery_attempts_per_issue: int = Field(default=2, ge=0)
    # Emit a budget warning once this fraction of a limit has been consumed.
    budget_warning_ratio: float = Field(default=0.8, gt=0.0, le=1.0)
    # Reflection scores at or below this value should trigger replanning.
    reflection_replan_threshold: int = Field(default=60, ge=0, le=100)
    # Reflection scores at or below this value should stop the run.
    reflection_stop_threshold: int = Field(default=30, ge=0, le=100)
    # How much approval detail the CLI should render once phase 4 lands.
    approval_ui_mode: Literal["compact", "detailed"] = "compact"
    # Maximum bytes returned for stdout before truncation
    max_output_bytes: int = Field(default=65536, ge=1)
    # Maximum bytes returned for stderr before truncation
    max_stderr_bytes: int = Field(default=32768, ge=1)
    # Low-risk command prefixes that are allowed in phase 2
    command_allowlist: list[str] = Field(
        default_factory=lambda: [
            "uv run pytest",
            "uv run mypy",
            "uv run ruff",
            "pytest",
            "python -m pytest",
            "mypy",
            "python -m mypy",
            "ruff",
            "python -m ruff",
            "git status",
            "git diff",
            "git --no-pager diff",
            "ls",
            "pwd",
            "find",
            "rg",
            "grep",
            "curl -s",
            "cat",
            "head",
            "curl -sL",
            "tail",
            "sort",
            "uniq",
            "echo",
            "wc -c",
        ]
    )
    # Fallback risk for commands that do not match any configured prefix.
    # Keep this strict by default; individual workspaces can opt into a more
    # permissive mode via config.yaml.
    command_default_risk: Literal["low", "high"] = "high"
    # Explicitly denied command prefixes or executables
    command_denylist: list[str] = Field(
        default_factory=lambda: [
            "sudo",
            "su",
            "sh -c",
            "bash -c",
            "zsh -c",
            "python -c",
            "python3 -c",
            "rm",
            "mv",
            "chmod",
            "chown",
            "curl",
            "wget",
            "ssh",
            "scp",
            "rsync",
            "dd",
            "mkfs",
            "systemctl",
            "crontab",
        ]
    )
    # Reserved for later approval-based phases; phase 2 still denies them
    command_approvallist: list[str] = Field(default_factory=list)
    # Only these env vars may be forwarded to run_command
    command_env_allowlist: list[str] = Field(
        default_factory=lambda: [
            "CI",
            "FORCE_COLOR",
            "MYPYPATH",
            "NO_COLOR",
            "PYTEST_ADDOPTS",
            "PYTHONPATH",
            "RUFF_OUTPUT_FORMAT",
        ]
    )
    # Workspace-relative directories where commands may execute
    command_working_dirs: list[str] = Field(default_factory=lambda: ["."])

    # --- Phase-3 write controls -----------------------------------------
    # Keep write tools behind an approval gate by default.
    write_requires_approval: bool = True
    # Maximum UTF-8 bytes accepted for a proposed patch or write payload.
    max_patch_bytes: int = Field(default=32768, ge=1)
    # Maximum patch hunks allowed before Policy Guard denies the request.
    max_patch_hunks: int = Field(default=24, ge=1)
    # Where backups will be stored once write execution is implemented.
    backup_dir: Path = Field(default=Path(".linux-agent/backups"))
    # Roll back automatically after failed verification runs by default.
    auto_rollback_on_verify_failure: bool = True

    # --- LLM -------------------------------------------------------------
    llm_model: str = "deepseek-v4-pro"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # OpenAI-compatible base URL for DeepSeek's chat-completions endpoint.
    llm_base_url: str | None = "https://api.deepseek.com"
    # API key for the configured provider; prefer env vars in practice
    llm_api_key: str | None = None

    # --- Security --------------------------------------------------------
    # Path components (str) that are never allowed, regardless of workspace
    sensitive_path_parts: list[str] = Field(
        default_factory=lambda: [".ssh", ".gnupg", "shadow", "passwd"]
    )

    # --- Audit -----------------------------------------------------------
    # Directory where per-run JSONL audit logs are written
    log_dir: Path = Field(default=Path("logs"))

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("workspace_root", mode="before")
    @classmethod
    def _expand_workspace(cls, v: Any) -> Path:
        """Resolve ~ and relative paths to an absolute path."""
        return Path(v).expanduser().resolve()

    @field_validator("log_dir", "backup_dir", mode="before")
    @classmethod
    def _expand_log_dir(cls, v: Any) -> Path:
        return Path(v).expanduser()

    @field_validator("llm_base_url", "llm_api_key", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return str(v)

    @field_validator(
        "command_allowlist",
        "command_denylist",
        "command_approvallist",
        "command_env_allowlist",
        "command_working_dirs",
        mode="before",
    )
    @classmethod
    def _normalize_string_lists(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            items = [v]
        else:
            try:
                items = list(v)
            except TypeError:
                items = [v]

        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
        return normalized

    @field_validator("command_default_risk", mode="before")
    @classmethod
    def _normalize_command_default_risk(cls, v: Any) -> str:
        if v is None:
            return "high"
        return str(v).strip().lower()

    @model_validator(mode="after")
    def _validate_workspace_exists(self) -> "AgentConfig":
        """workspace_root must exist and be a directory at load time."""
        if not self.workspace_root.exists():
            raise ValueError(
                f"workspace_root does not exist: {self.workspace_root}"
            )
        if not self.workspace_root.is_dir():
            raise ValueError(
                f"workspace_root is not a directory: {self.workspace_root}"
            )
        if self.reflection_stop_threshold > self.reflection_replan_threshold:
            raise ValueError(
                "reflection_stop_threshold must be less than or equal to "
                "reflection_replan_threshold"
            )
        return self


def load_config(path: str | None = None) -> AgentConfig:
    """
    Load AgentConfig from a YAML file or environment variables.

    Resolution order (highest priority first):
    1. YAML file fields (when *path* is given)
    2. Environment variables for workspace / provider credentials
    3. Built-in defaults

    Raises
    ------
    ValueError
        If workspace_root cannot be resolved, does not exist, or is not a
        directory.
    FileNotFoundError
        If *path* is given but the file does not exist.
    """
    data: dict[str, Any] = {}

    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                data = loaded
        for key in ("workspace_root", "log_dir", "backup_dir"):
            value = data.get(key)
            if isinstance(value, str):
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    data[key] = str((config_path.parent / candidate).resolve())

    # Fall back to environment variable for workspace_root when not in YAML
    if "workspace_root" not in data:
        env_ws = os.environ.get("LINUX_AGENT_WORKSPACE")
        if env_ws:
            data["workspace_root"] = env_ws
        else:
            raise ValueError(
                "workspace_root must be set via config file or "
                "LINUX_AGENT_WORKSPACE environment variable"
            )

    if "llm_base_url" not in data:
        env_base_url = (
            os.environ.get("LINUX_AGENT_LLM_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        if env_base_url:
            data["llm_base_url"] = env_base_url

    if "llm_api_key" not in data:
        env_api_key = (
            os.environ.get("LINUX_AGENT_LLM_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if env_api_key:
            data["llm_api_key"] = env_api_key

    return AgentConfig(**data)

