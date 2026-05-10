"""
File-system skills for the Linux Agent (phase 1: read-only).

All functions validate paths through policy.resolve_safe_path before
performing any I/O. Errors are returned as structured dicts rather than
raised exceptions so the Tool Executor can build an Observation cleanly.
PolicyViolation is intentionally *not* caught here – it propagates up
to the Tool Executor which wraps it into a failed Observation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from linux_agent.config import AgentConfig
from linux_agent.policy import resolve_safe_path


# ─────────────────────────────────────────────────────────────────────────
# T6  –  list_dir
# ─────────────────────────────────────────────────────────────────────────

def list_dir(
    path: str,
    config: AgentConfig,
    *,
    recursive: bool = False,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """
    List the contents of *path* (relative to workspace_root).

    Parameters
    ----------
    path:
        Path relative to ``config.workspace_root``.  Use ``"."`` for the
        root itself.
    config:
        Agent configuration (provides workspace_root, max_list_entries,
        sensitive_path_parts).
    recursive:
        When True, descend into sub-directories.  Hidden directories
        (names starting with ``"."`` ) are always skipped.
    max_entries:
        Override ``config.max_list_entries`` for this call.  Entries are
        truncated (not errored) once the limit is reached.

    Returns
    -------
    dict with keys: ok, path, entries, truncated, total_visible
    On error: ok=False, error=<reason>
    """
    limit = max_entries if max_entries is not None else config.max_list_entries
    root = config.workspace_root

    # Policy check – raises PolicyViolation on bad paths (caller handles)
    safe = resolve_safe_path(root, path, config.sensitive_path_parts)

    if not safe.exists():
        return {"ok": False, "error": "path not found", "path": path}
    if not safe.is_dir():
        return {"ok": False, "error": "not a directory", "path": path}

    entries: list[dict[str, Any]] = []
    truncated = False

    def _scan(directory: Path) -> None:
        nonlocal truncated
        try:
            with os.scandir(directory) as it:
                for entry in sorted(it, key=lambda e: e.name):
                    if len(entries) >= limit:
                        truncated = True
                        return

                    stat = entry.stat(follow_symlinks=False)
                    if entry.is_symlink():
                        kind = "symlink"
                        size: int | None = stat.st_size
                    elif entry.is_dir(follow_symlinks=False):
                        kind = "directory"
                        size = None
                    else:
                        kind = "file"
                        size = stat.st_size

                    rel = Path(entry.path).relative_to(root)
                    entries.append(
                        {
                            "name": entry.name,
                            "path": rel.as_posix(),
                            "type": kind,
                            "size": size,
                        }
                    )

                    # Recurse into non-hidden sub-directories
                    if (
                        recursive
                        and entry.is_dir(follow_symlinks=False)
                        and not entry.name.startswith(".")
                    ):
                        _scan(Path(entry.path))
        except PermissionError:
            # Skip directories we cannot read
            pass

    _scan(safe)

    return {
        "ok": True,
        "path": path,
        "entries": entries,
        "truncated": truncated,
        "total_visible": len(entries),
    }


# ─────────────────────────────────────────────────────────────────────────
# T7  –  read_file
# ─────────────────────────────────────────────────────────────────────────

def read_file(
    path: str,
    config: AgentConfig,
    *,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    """
    Read a text file (or a line-range slice of it) from the workspace.

    Parameters
    ----------
    path:
        Path relative to ``config.workspace_root``.
    config:
        Agent configuration.
    start_line:
        First line to return (1-based, inclusive).
    end_line:
        Last line to return (1-based, inclusive).  ``None`` means EOF.
        Values beyond the actual line count are clamped silently.

    Returns
    -------
    dict with keys: ok, path, content, start_line, end_line,
                    total_lines, truncated
    On error: ok=False, error=<reason>
    """
    root = config.workspace_root

    safe = resolve_safe_path(root, path, config.sensitive_path_parts)

    if not safe.exists():
        return {"ok": False, "error": "file not found", "path": path}
    if safe.is_dir():
        return {"ok": False, "error": "path is a directory", "path": path}

    # Read as text; replace undecodable bytes rather than crashing
    try:
        raw = safe.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "error": str(exc), "path": path}

    all_lines = raw.splitlines(keepends=True)
    total_lines = len(all_lines)

    # Clamp line numbers (1-based → 0-based slice)
    lo = max(1, start_line) - 1                        # inclusive, 0-based
    hi = (min(end_line, total_lines) if end_line is not None else total_lines)  # exclusive
    selected = all_lines[lo:hi]

    content = "".join(selected)
    truncated = False

    # Byte-limit enforcement
    limit = config.max_read_bytes
    if len(content.encode("utf-8")) > limit:
        # Truncate at a character boundary that keeps us within the limit
        content = content.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
        truncated = True

    actual_end = min(lo + len(selected), total_lines)

    return {
        "ok": True,
        "path": path,
        "content": content,
        "start_line": lo + 1,
        "end_line": actual_end,
        "total_lines": total_lines,
        "truncated": truncated,
    }

