"""
Security policy enforcement for the Linux Agent.

All tool calls and file-system paths must pass through this module before
any real I/O occurs. The functions here are intentionally simple and
stateless so they are easy to unit-test in isolation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shlex
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from linux_agent.config import AgentConfig
    from linux_agent.state import ToolCall

# ── Phase-1 read-only tool allowlist ──────────────────────────────────────
READ_ONLY_TOOLS: frozenset[str] = frozenset({"list_dir", "read_file", "search_text"})
COMMAND_EXECUTION_TOOLS: frozenset[str] = frozenset({"run_command"})
_UNSAFE_SHELL_PATTERNS: tuple[str, ...] = (
    "&&",
    "||",
    ">>",
    ";",
    "|",
    ">",
    "<",
    "`",
    "$(",
    "&",
)


# ── Exception ────────────────────────────────────────────────────────────
class PolicyViolation(Exception):
    """Raised when a requested operation violates the security policy."""

    def __init__(self, reason: str, path: str = "") -> None:
        self.reason = reason
        self.path = path
        msg = f"Policy violation: {reason}"
        if path:
            msg += f" (path={path})"
        super().__init__(msg)


# ── Path safety ────────────────────────────────────────────────────────
DEFAULT_SENSITIVE_PARTS: frozenset[str] = frozenset(
    {".ssh", ".gnupg", "shadow", "passwd"}
)


def resolve_safe_path(
    workspace_root: str | Path,
    path: str,
    sensitive_path_parts: frozenset[str] | list[str] | None = None,
) -> Path:
    """
    Resolve *path* relative to *workspace_root* and verify it is safe.

    Steps
    -----
    1. Canonicalise *workspace_root* with ``Path.resolve()``.
    2. Join *path* onto the root; if *path* is absolute, it is used as-is
       which will almost always fail step 3 (desired behaviour).
    3. If the candidate exists, resolve it fully (follows symlinks).  If it
       does not exist, resolve the longest existing prefix so that ``..``
       traversals are still caught without requiring the target to exist.
    4. Assert that the resolved path is inside *workspace_root*.
    5. Reject paths whose components match any entry in
       *sensitive_path_parts*.

    Returns the safe absolute ``Path``.

    Raises
    ------
    PolicyViolation
        If the path escapes the workspace or matches a sensitive pattern.
    """
    sensitive: frozenset[str] = (
        DEFAULT_SENSITIVE_PARTS
        if sensitive_path_parts is None
        else frozenset(sensitive_path_parts)
    )

    root = Path(workspace_root).resolve()

    # Build candidate – Path("/abs") ignores the root, which is intentional:
    # the subsequent boundary check will reject it.
    candidate = (root / path)

    # Resolve symlinks only if the path (or a prefix of it) exists so we
    # still catch traversal on non-existent paths via pure string analysis.
    if candidate.exists():
        resolved = candidate.resolve()
    else:
        # Walk up until we find an existing prefix, resolve that, then
        # re-append the remaining suffix.
        existing = candidate
        suffix_parts: list[str] = []
        while not existing.exists() and existing != existing.parent:
            suffix_parts.append(existing.name)
            existing = existing.parent
        resolved = existing.resolve() / Path(*reversed(suffix_parts)) if suffix_parts else existing.resolve()

    # ── Boundary check ───────────────────────────────────────────
    # resolved == root  → accessing the root itself (allowed)
    # root in resolved.parents  → resolved is a descendant (allowed)
    if resolved != root and root not in resolved.parents:
        raise PolicyViolation(
            "path escapes workspace root",
            path=str(path),
        )

    # ── Sensitive component check ──────────────────────────────────
    # Check every component of the *relative* portion only, so that the
    # workspace root itself never triggers a false positive.
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        # Should not reach here given the boundary check above, but be safe.
        raise PolicyViolation("path escapes workspace root", path=str(path))

    for part in relative.parts:
        if part in sensitive:
            raise PolicyViolation(
                f"path contains sensitive component '{part}'",
                path=str(path),
            )

    return resolved


def parse_command(raw: str) -> list[str]:
    """Parse a command string conservatively and reject shell control syntax."""
    normalized = raw.strip()
    if not normalized:
        raise PolicyViolation("command is empty")

    if any(pattern in normalized for pattern in _UNSAFE_SHELL_PATTERNS):
        raise PolicyViolation("unsupported shell syntax in command", path=raw)

    try:
        argv = shlex.split(normalized, posix=True)
    except ValueError as exc:
        raise PolicyViolation(f"failed to parse command: {exc}", path=raw) from exc

    if not argv:
        raise PolicyViolation("command is empty")

    return argv


def _normalize_command_text(argv: list[str]) -> str:
    return " ".join(part.strip() for part in argv if part.strip())


def _matches_command_prefix(command: str, patterns: list[str] | frozenset[str]) -> bool:
    for pattern in patterns:
        normalized = pattern.strip()
        if not normalized:
            continue
        if command == normalized or command.startswith(f"{normalized} "):
            return True
    return False


def _unwrap_uv_run(argv: list[str]) -> list[str] | None:
    if len(argv) >= 4 and argv[0] == "uv" and argv[1] == "run" and argv[2] == "--":
        return argv[3:]
    if len(argv) >= 3 and argv[0] == "uv" and argv[1] == "run":
        return argv[2:]
    return None


def classify_command(
    argv: list[str],
    config: "AgentConfig",
) -> Literal["low", "medium", "high"]:
    """Classify a parsed command according to the configured command policy."""
    normalized = _normalize_command_text(argv)
    if not normalized:
        return "high"

    inner = _unwrap_uv_run(argv)
    if inner:
        inner_risk = classify_command(inner, config)
        if inner_risk != "high":
            return inner_risk

    if _matches_command_prefix(normalized, config.command_denylist):
        return "high"
    if _matches_command_prefix(normalized, config.command_approvallist):
        return "medium"
    if _matches_command_prefix(normalized, config.command_allowlist):
        return "low"
    return "high"


def resolve_command_cwd(
    config: "AgentConfig",
    cwd: str | None = None,
) -> Path:
    """Resolve a run_command cwd and ensure it stays within allowed directories."""
    raw_cwd = cwd or "."
    resolved_cwd = resolve_safe_path(
        config.workspace_root,
        raw_cwd,
        config.sensitive_path_parts,
    )

    allowed_dirs = config.command_working_dirs or ["."]
    for allowed in allowed_dirs:
        allowed_path = resolve_safe_path(
            config.workspace_root,
            allowed,
            config.sensitive_path_parts,
        )
        if resolved_cwd == allowed_path or allowed_path in resolved_cwd.parents:
            return resolved_cwd

    raise PolicyViolation("command cwd is not in allowed working directories", path=raw_cwd)


def filter_command_env(
    env: Mapping[str, object] | None,
    allowed_names: list[str] | frozenset[str],
) -> dict[str, str]:
    """Keep only explicitly allowlisted env vars for command execution."""
    if not env:
        return {}

    allowset = frozenset(str(name) for name in allowed_names)
    filtered: dict[str, str] = {}
    for key, value in env.items():
        key_text = str(key)
        if key_text not in allowset:
            continue
        filtered[key_text] = str(value)
    return filtered


def evaluate_command_call(
    command: str,
    config: "AgentConfig",
    *,
    cwd: str = ".",
    env: Mapping[str, object] | None = None,
) -> Literal["allow", "deny"]:
    """Evaluate whether a run_command call should be permitted in phase 2."""
    try:
        argv = parse_command(command)
        resolve_command_cwd(config, cwd)
        if env is not None and not isinstance(env, Mapping):
            return "deny"
        risk = classify_command(argv, config)
    except PolicyViolation:
        return "deny"

    return "allow" if risk == "low" else "deny"


def evaluate_tool_call(
    tool_call: "ToolCall",
    config: "AgentConfig",
) -> Literal["allow", "deny"]:
    """
    Evaluate a proposed tool call against the current security policy.

    Returns ``"allow"`` or ``"deny"``.
    Does **not** raise – callers receive a string decision so they can log
    the reason and route the graph accordingly.
    """
    name: str = tool_call["name"]
    args: dict[str, object] = tool_call["args"]

    # ── 1. Tool allowlist ───────────────────────────────────────────
    if name in COMMAND_EXECUTION_TOOLS:
        raw_command = args.get("command")
        if not isinstance(raw_command, str):
            return "deny"
        raw_cwd = args.get("cwd", ".")
        raw_env = args.get("env")
        if raw_env is not None and not isinstance(raw_env, Mapping):
            return "deny"
        return evaluate_command_call(
            raw_command,
            config,
            cwd=str(raw_cwd) if raw_cwd is not None else ".",
            env=raw_env,
        )

    if name not in READ_ONLY_TOOLS:
        return "deny"

    # ── 2. Path boundary check (for tools that accept a path arg) ────────
    raw_path = args.get("path")
    if raw_path is not None:
        try:
            resolve_safe_path(
                config.workspace_root,
                str(raw_path),
                sensitive_path_parts=config.sensitive_path_parts,
            )
        except PolicyViolation:
            return "deny"

    return "allow"

