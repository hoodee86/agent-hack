"""
Security policy enforcement for the Linux Agent.

All tool calls and file-system paths must pass through this module before
any real I/O occurs. The functions here are intentionally simple and
stateless so they are easy to unit-test in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from linux_agent.config import AgentConfig
    from linux_agent.state import ToolCall

# ── Phase-1 read-only tool allowlist ──────────────────────────────────────
READ_ONLY_TOOLS: frozenset[str] = frozenset({"list_dir", "read_file", "search_text"})


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


# ── Tool-level checks ────────────────────────────────────────────────────
def check_read_only_tool(tool_name: str) -> None:
    """
    Assert that *tool_name* is a phase-1 read-only tool.

    Raises
    ------
    PolicyViolation
        If the tool is not in the read-only allowlist.
    """
    if tool_name not in READ_ONLY_TOOLS:
        raise PolicyViolation(
            f"tool '{tool_name}' is not permitted in phase 1 (read-only)",
        )


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

