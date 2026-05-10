"""
Text-search skill for the Linux Agent (phase 1: read-only).

Uses ripgrep (rg) when available; falls back to a pure-Python
re-based implementation so tests can run without rg installed.

All paths are validated through policy.resolve_safe_path.  The ripgrep
process is invoked with a list of arguments (shell=False) so user-supplied
query strings cannot inject shell metacharacters.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from linux_agent.config import AgentConfig
from linux_agent.policy import resolve_safe_path

_MAX_LINE_CHARS = 200  # truncate individual matching lines to avoid LLM overflow


# ── Public API ────────────────────────────────────────────────────────────────
def search_text(
    query: str,
    config: AgentConfig,
    *,
    path: str = ".",
    glob: str = "**/*",
    max_results: int | None = None,
    context_lines: int = 2,
) -> dict[str, Any]:
    """
    Search for *query* (treated as a literal string) inside the workspace.

    Parameters
    ----------
    query:
        Text to search for (literal, not regex).
    config:
        Agent configuration.
    path:
        Sub-directory to search in, relative to ``config.workspace_root``.
    glob:
        File-name glob pattern (e.g. ``"**/*.py"``).
    max_results:
        Override ``config.max_search_results``.
    context_lines:
        Lines of surrounding context to include per match.

    Returns
    -------
    dict with keys: ok, query, matches, total_matches, truncated
    On error: ok=False, error=<reason>
    """
    limit = max_results if max_results is not None else config.max_search_results
    root = config.workspace_root

    safe_search_root = resolve_safe_path(root, path, config.sensitive_path_parts)

    if not safe_search_root.exists():
        return {"ok": False, "error": "search path not found", "query": query}

    if shutil.which("rg") is not None:
        matches, truncated = _search_ripgrep(
            query, safe_search_root, root, glob, limit, context_lines
        )
    else:
        matches, truncated = _search_python(
            query, safe_search_root, root, glob, limit, context_lines
        )

    return {
        "ok": True,
        "query": query,
        "matches": matches,
        "total_matches": len(matches),
        "truncated": truncated,
    }


# ── ripgrep backend ────────────────────────────────────────────────────────────
def _search_ripgrep(
    query: str,
    search_root: Path,
    workspace_root: Path,
    glob: str,
    limit: int,
    context_lines: int,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Run ``rg --json`` and parse its JSON stream output.

    Each rg JSON message is one of: ``begin``, ``match``, ``context``,
    ``end``, ``summary``.  We collect ``match`` messages and attach the
    surrounding ``context`` lines from the preceding/following messages.
    """
    cmd = [
        "rg",
        "--json",
        "-F",                        # fixed string (not regex)
        f"-C{context_lines}",         # context lines
        "-g", glob,
        "--", query,
        str(search_root),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            shell=False,  # MUST remain False – security requirement
        )
    except subprocess.TimeoutExpired:
        return [], False

    # rg exits 1 when no matches found, 2 on error; 0 = matches found
    if proc.returncode == 2:
        return [], False

    # Parse the JSON stream; buffer context lines around each match
    pending_context_before: list[str] = []
    # Maps (file, line_no) -> partial match dict waiting for context_after
    waiting: dict[tuple[str, int], dict[str, Any]] = {}
    matches: list[dict[str, Any]] = []
    truncated = False

    for raw_line in proc.stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        kind = msg.get("type")
        data = msg.get("data", {})

        if kind == "begin":
            pending_context_before = []

        elif kind == "context":
            line_text = data.get("lines", {}).get("text", "").rstrip("\n")
            pending_context_before.append(_trunc(line_text))
            # Feed context_after to any waiting matches in this file
            file_path = data.get("path", {}).get("text", "")
            line_no = data.get("line_number", 0)
            for key, m in list(waiting.items()):
                if key[0] == file_path and len(m["context_after"]) < context_lines:
                    m["context_after"].append(_trunc(line_text))
                    if len(m["context_after"]) >= context_lines:
                        del waiting[key]

        elif kind == "match":
            if len(matches) >= limit:
                truncated = True
                break

            file_path = data.get("path", {}).get("text", "")
            line_no = data.get("line_number", 0)
            line_text = data.get("lines", {}).get("text", "").rstrip("\n")

            # Make file path relative to workspace_root
            try:
                rel = Path(file_path).relative_to(workspace_root).as_posix()
            except ValueError:
                rel = file_path

            ctx_before = list(pending_context_before[-context_lines:]) if context_lines else []
            match_entry: dict[str, Any] = {
                "file": rel,
                "line_number": line_no,
                "line": _trunc(line_text),
                "context_before": ctx_before,
                "context_after": [],
            }
            matches.append(match_entry)
            if context_lines:
                waiting[(file_path, line_no)] = match_entry
            pending_context_before = []

    return matches, truncated


# ── Python fallback backend ─────────────────────────────────────────────────────
def _search_python(
    query: str,
    search_root: Path,
    workspace_root: Path,
    glob: str,
    limit: int,
    context_lines: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Pure-Python fallback using ``re.search`` (literal match)."""
    pattern = re.compile(re.escape(query))
    matches: list[dict[str, Any]] = []
    truncated = False

    for file_path in sorted(search_root.rglob(glob)):
        if not file_path.is_file():
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for i, line in enumerate(lines):
            if not pattern.search(line):
                continue
            if len(matches) >= limit:
                truncated = True
                return matches, truncated

            try:
                rel = file_path.relative_to(workspace_root).as_posix()
            except ValueError:
                rel = str(file_path)

            lo = max(0, i - context_lines)
            hi = min(len(lines), i + context_lines + 1)

            matches.append(
                {
                    "file": rel,
                    "line_number": i + 1,
                    "line": _trunc(line),
                    "context_before": [_trunc(l) for l in lines[lo:i]],
                    "context_after": [_trunc(l) for l in lines[i + 1 : hi]],
                }
            )

    return matches, truncated


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _trunc(text: str) -> str:
    """Truncate a single line to _MAX_LINE_CHARS characters."""
    if len(text) > _MAX_LINE_CHARS:
        return text[:_MAX_LINE_CHARS] + " …"
    return text

