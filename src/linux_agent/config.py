"""
Configuration loading for the Linux Agent.

AgentConfig holds all tunable parameters. load_config() resolves values
from a YAML file (optional) or environment variables, then validates them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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

    # --- LLM -------------------------------------------------------------
    llm_model: str = "gpt-4o"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

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

    @field_validator("log_dir", mode="before")
    @classmethod
    def _expand_log_dir(cls, v: Any) -> Path:
        return Path(v).expanduser()

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
        return self


def load_config(path: str | None = None) -> AgentConfig:
    """
    Load AgentConfig from a YAML file or environment variables.

    Resolution order (highest priority first):
    1. YAML file fields (when *path* is given)
    2. LINUX_AGENT_WORKSPACE environment variable (for workspace_root)
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

    return AgentConfig(**data)

