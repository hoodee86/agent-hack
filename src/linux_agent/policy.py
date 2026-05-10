"""
Security policy enforcement for the Linux Agent.

All tool calls and file-system paths must pass through this module before
any real I/O occurs. The functions here are intentionally simple and
stateless so they are easy to unit-test in isolation.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
import re
import shlex
from typing import TYPE_CHECKING, Literal, cast
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from linux_agent.config import AgentConfig
    from linux_agent.state import ApprovalRequest, ToolCall

# ── Phase-1 read-only tool allowlist ──────────────────────────────────────
READ_ONLY_TOOLS: frozenset[str] = frozenset({"list_dir", "read_file", "search_text"})
COMMAND_EXECUTION_TOOLS: frozenset[str] = frozenset({"run_command"})
WRITE_APPROVAL_TOOLS: frozenset[str] = frozenset({"apply_patch", "write_file"})
_RAW_UNSAFE_SHELL_PATTERNS: tuple[str, ...] = (
    "`",
    "$(",
)
_SUPPORTED_SEQUENCE_OPERATORS: frozenset[str] = frozenset({"&&", "||", ";"})
_SUPPORTED_PIPE_OPERATOR = "|"
_SUPPORTED_STDOUT_REDIRECT_OPERATORS: frozenset[str] = frozenset({">", ">>"})
_BINARY_PATH_SUFFIXES: frozenset[str] = frozenset(
    {
        ".7z",
        ".bin",
        ".class",
        ".dll",
        ".dylib",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".lockb",
        ".mp3",
        ".mp4",
        ".npy",
        ".o",
        ".pdf",
        ".png",
        ".pyc",
        ".so",
        ".tar",
        ".tgz",
        ".wav",
        ".webp",
        ".whl",
        ".zip",
    }
)
_PATCH_FILE_RE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$", re.MULTILINE)
_UNIFIED_DIFF_PATH_RE = re.compile(r"^(?:--- a/|\+\+\+ b/)(.+)$", re.MULTILINE)
_PATCH_HUNK_RE = re.compile(r"^@@", re.MULTILINE)
_INLINE_PYTHON_EXECUTABLES: frozenset[str] = frozenset({"python", "python3"})
_INLINE_PYTHON_SAFE_IMPORTS: frozenset[str] = frozenset({"json"})
_INLINE_PYTHON_SAFE_BUILTINS: frozenset[str] = frozenset({"len", "open", "print"})
_INLINE_PYTHON_SAFE_JSON_CALLS: frozenset[str] = frozenset({"load", "loads"})
_INLINE_PYTHON_SAFE_READ_MODES: frozenset[str] = frozenset({"r", "rb", "rt"})
_INLINE_PYTHON_SAFE_NODE_TYPES: tuple[type[ast.AST], ...] = (
    ast.Add,
    ast.Assign,
    ast.Attribute,
    ast.BinOp,
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Dict,
    ast.Eq,
    ast.Expr,
    ast.For,
    ast.FormattedValue,
    ast.If,
    ast.Import,
    ast.JoinedStr,
    ast.List,
    ast.Load,
    ast.Module,
    ast.Mult,
    ast.Name,
    ast.Store,
    ast.Subscript,
    ast.Tuple,
    ast.With,
    ast.alias,
    ast.keyword,
    ast.withitem,
)


class CommandStage(TypedDict):
    """One argv stage inside a parsed command segment."""

    argv: list[str]
    command: str
    merge_stderr: NotRequired[bool]
    stdout_path: NotRequired[str]
    stdout_append: NotRequired[bool]


class PolicyAssessment(TypedDict):
    """Structured policy output consumed by Policy Guard and tests."""

    decision: Literal["allow", "deny", "needs_approval"]
    reason: str | None
    approval_request: ApprovalRequest | None


class CommandSegment(TypedDict):
    """One parsed command segment inside a compound command string."""

    operator: Literal["&&", "||", ";"] | None
    argv: list[str]
    command: str
    stages: NotRequired[list[CommandStage]]


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


def _allow() -> PolicyAssessment:
    return PolicyAssessment(decision="allow", reason=None, approval_request=None)


def _deny(reason: str) -> PolicyAssessment:
    return PolicyAssessment(decision="deny", reason=reason, approval_request=None)


def _needs_approval(request: "ApprovalRequest") -> PolicyAssessment:
    return PolicyAssessment(
        decision="needs_approval",
        reason=request["reason"],
        approval_request=request,
    )


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


def _tokenize_command(raw: str) -> list[str]:
    normalized = raw.strip()
    if not normalized:
        raise PolicyViolation("command is empty")

    if any(pattern in normalized for pattern in _RAW_UNSAFE_SHELL_PATTERNS):
        raise PolicyViolation("unsupported shell syntax in command", path=raw)

    try:
        lexer = shlex.shlex(normalized, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError as exc:
        raise PolicyViolation(f"failed to parse command: {exc}", path=raw) from exc

    if not tokens:
        raise PolicyViolation("command is empty")

    return tokens


def _is_unsupported_shell_token(token: str) -> bool:
    if (
        token in _SUPPORTED_SEQUENCE_OPERATORS
        or token == _SUPPORTED_PIPE_OPERATOR
        or token in _SUPPORTED_STDOUT_REDIRECT_OPERATORS
    ):
        return False
    punctuation = set(token)
    if punctuation and punctuation.issubset({"&", "|", ";", ">", "<"}):
        return True
    return False


def _build_command_stage(
    argv: list[str],
    *,
    merge_stderr: bool = False,
    stdout_path: str | None = None,
    stdout_append: bool = False,
) -> CommandStage:
    if not argv:
        raise PolicyViolation("unsupported shell syntax in command")
    stage_argv = list(argv)
    stage = CommandStage(argv=stage_argv, command=shlex.join(stage_argv))
    if merge_stderr:
        stage["merge_stderr"] = True
        stage["command"] = f"{stage['command']} 2>&1"
    if stdout_path is not None:
        stage["stdout_path"] = stdout_path
        if stdout_append:
            stage["stdout_append"] = True
        redirect_op = ">>" if stdout_append else ">"
        stage["command"] = f"{stage['command']} {redirect_op} {shlex.quote(stdout_path)}"
    return stage


def command_segment_stages(segment: CommandSegment) -> list[CommandStage]:
    stages = segment.get("stages")
    if stages:
        return stages
    return [CommandStage(argv=list(segment["argv"]), command=segment["command"])]


def parse_command_sequence(raw: str) -> list[CommandSegment]:
    """Parse a command string into sequential command segments.

    Supported separators are ``&&``, ``||``, and ``;``. Each segment may also
    contain a pipeline joined with ``|``. Supported redirections are ``2>&1``
    to merge stderr into stdout for a stage and ``>``/``>>`` to write a stage's
    stdout to a file. Other shell features such as subshells and background
    execution still raise ``PolicyViolation``.
    """
    tokens = _tokenize_command(raw)
    segments: list[CommandSegment] = []
    argv: list[str] = []
    stages: list[CommandStage] = []
    merge_stderr = False
    stdout_path: str | None = None
    stdout_append = False
    next_operator: Literal["&&", "||", ";"] | None = None

    def _flush_segment() -> None:
        nonlocal argv, stages, merge_stderr, stdout_path, stdout_append
        stage = _build_command_stage(
            argv,
            merge_stderr=merge_stderr,
            stdout_path=stdout_path,
            stdout_append=stdout_append,
        )
        all_stages = [*stages, stage]
        segment = CommandSegment(
            operator=next_operator,
            argv=all_stages[0]["argv"],
            command=" | ".join(item["command"] for item in all_stages),
        )
        if len(all_stages) > 1 or stage.get("merge_stderr") or stage.get("stdout_path") is not None:
            segment["stages"] = all_stages
        segments.append(segment)
        argv = []
        stages = []
        merge_stderr = False
        stdout_path = None
        stdout_append = False

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token == "2"
            and index + 2 < len(tokens)
            and tokens[index + 1] == ">&"
            and tokens[index + 2] == "1"
        ):
            if not argv:
                raise PolicyViolation("unsupported shell syntax in command", path=raw)
            merge_stderr = True
            index += 3
            continue

        if _is_unsupported_shell_token(token):
            raise PolicyViolation("unsupported shell syntax in command", path=raw)

        if token in _SUPPORTED_STDOUT_REDIRECT_OPERATORS:
            if not argv or stdout_path is not None or index + 1 >= len(tokens):
                raise PolicyViolation("unsupported shell syntax in command", path=raw)
            target = tokens[index + 1]
            if (
                target == _SUPPORTED_PIPE_OPERATOR
                or target in _SUPPORTED_SEQUENCE_OPERATORS
                or target in _SUPPORTED_STDOUT_REDIRECT_OPERATORS
                or _is_unsupported_shell_token(target)
            ):
                raise PolicyViolation("unsupported shell syntax in command", path=raw)
            stdout_path = target
            stdout_append = token == ">>"
            index += 2
            continue

        if token == _SUPPORTED_PIPE_OPERATOR:
            if stdout_path is not None:
                raise PolicyViolation("unsupported shell syntax in command", path=raw)
            stages.append(_build_command_stage(argv, merge_stderr=merge_stderr))
            argv = []
            merge_stderr = False
            index += 1
            continue

        if token in _SUPPORTED_SEQUENCE_OPERATORS:
            if not argv:
                raise PolicyViolation("unsupported shell syntax in command", path=raw)
            _flush_segment()
            next_operator = cast("Literal['&&', '||', ';']", token)
            index += 1
            continue

        argv.append(token)
        index += 1

    if not argv:
        raise PolicyViolation("unsupported shell syntax in command", path=raw)

    _flush_segment()
    return segments


def parse_command(raw: str) -> list[str]:
    """Parse a single command argv and reject shell control syntax."""
    segments = parse_command_sequence(raw)
    stages = command_segment_stages(segments[0]) if len(segments) == 1 else []
    if (
        len(segments) != 1
        or len(stages) != 1
        or stages[0].get("merge_stderr")
        or stages[0].get("stdout_path") is not None
    ):
        raise PolicyViolation("unsupported shell syntax in command", path=raw)
    return segments[0]["argv"]


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


def _best_command_prefix_length(
    command: str,
    patterns: list[str] | frozenset[str],
) -> int | None:
    best: int | None = None
    for pattern in patterns:
        normalized = pattern.strip()
        if not normalized:
            continue
        if command == normalized or command.startswith(f"{normalized} "):
            length = len(normalized)
            best = length if best is None else max(best, length)
    return best


def _unwrap_uv_run(argv: list[str]) -> list[str] | None:
    if len(argv) >= 4 and argv[0] == "uv" and argv[1] == "run" and argv[2] == "--":
        return argv[3:]
    if len(argv) >= 3 and argv[0] == "uv" and argv[1] == "run":
        return argv[2:]
    return None


def _is_probably_binary_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _BINARY_PATH_SUFFIXES


def _contains_binary_content(value: object) -> bool:
    if isinstance(value, bytes):
        return b"\x00" in value
    if isinstance(value, str):
        return "\x00" in value
    return False


def _byte_length(value: object) -> int:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8", errors="ignore"))
    return len(str(value).encode("utf-8", errors="ignore"))


def _preview_text(value: object, *, limit: int = 320) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    stripped = text.strip()
    if not stripped:
        return None
    preview = "\n".join(line.rstrip() for line in stripped.splitlines()[:12]).strip()
    if len(preview) > limit:
        return preview[:limit].rstrip() + " …"
    return preview


def _workspace_display_path(config: "AgentConfig", path: Path) -> str:
    try:
        return str(path.relative_to(config.workspace_root)) or "."
    except ValueError:
        return str(path)


def _backup_plan(config: "AgentConfig", run_id: str | None) -> str:
    run_segment = run_id or "<run_id>"
    return f"Backups will be written under {config.backup_dir / run_segment} before any overwrite."


def _extract_patch_targets(patch_text: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for match in _PATCH_FILE_RE.finditer(patch_text):
        action = match.group(1).lower()
        path = match.group(2).strip()
        key = (action, path)
        if path and key not in seen:
            targets.append(key)
            seen.add(key)

    if targets:
        return targets

    for match in _UNIFIED_DIFF_PATH_RE.finditer(patch_text):
        path = match.group(1).strip()
        if path == "/dev/null":
            continue
        key = ("update", path)
        if path and key not in seen:
            targets.append(key)
            seen.add(key)

    return targets


def _count_patch_hunks(patch_text: str) -> int:
    hunk_count = len(_PATCH_HUNK_RE.findall(patch_text))
    if hunk_count:
        return hunk_count
    file_sections = len(_PATCH_FILE_RE.findall(patch_text))
    if file_sections:
        return file_sections
    return 1 if patch_text.strip() else 0


def _build_approval_request(
    tool_call: "ToolCall",
    *,
    reason: str,
    impact_summary: str,
    diff_preview: str | None,
    backup_plan: str | None,
    affected_files: list[str],
    suggested_verification_command: str | None,
    rollback_command: str | None,
) -> "ApprovalRequest":
    return {
        "id": f"approval_{uuid4()}",
        "tool": tool_call["name"],
        "args": tool_call["args"],
        "reason": reason,
        "impact_summary": impact_summary,
        "diff_preview": diff_preview,
        "backup_plan": backup_plan,
        "affected_files": affected_files,
        "risk_level": tool_call["risk_level"],
        "suggested_verification_command": suggested_verification_command,
        "rollback_command": rollback_command,
    }


def _suggest_verification_command(
    config: "AgentConfig",
    affected_files: list[str],
) -> str | None:
    if not affected_files:
        return None

    suffixes = {Path(path).suffix.lower() for path in affected_files}
    if not suffixes.intersection({".py", ".pyi", ".toml", ".yaml", ".yml"}):
        return None

    candidates = (
        "uv run pytest",
        "pytest",
        "python -m pytest",
        "uv run ruff",
        "ruff",
        "python -m ruff",
    )
    allowlist = [str(item).strip() for item in config.command_allowlist if str(item).strip()]
    for candidate in candidates:
        for allowed in allowlist:
            if allowed == candidate or allowed.startswith(f"{candidate} "):
                return allowed
    return None


def _assess_write_file_call(
    tool_call: "ToolCall",
    config: "AgentConfig",
    *,
    run_id: str | None,
) -> PolicyAssessment:
    args = tool_call["args"]
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return _deny("write_file is missing a valid path")

    raw_content = args.get("content", "")
    if raw_content is not None and not isinstance(raw_content, (str, bytes)):
        return _deny("write_file content must be text")
    if _contains_binary_content(raw_content):
        return _deny("write_file content appears to be binary")
    if _is_probably_binary_path(raw_path):
        return _deny("write_file target appears to be a binary file")
    if _byte_length(raw_content) > config.max_patch_bytes:
        return _deny("write_file content exceeds max_patch_bytes")

    mode = str(args.get("mode", "overwrite") or "overwrite").strip().lower()
    if mode not in {"overwrite", "create_only", "append"}:
        return _deny(f"unsupported write_file mode: {mode}")

    try:
        resolved = resolve_safe_path(
            config.workspace_root,
            raw_path,
            sensitive_path_parts=config.sensitive_path_parts,
        )
    except PolicyViolation as exc:
        return _deny(exc.reason)

    display_path = _workspace_display_path(config, resolved)
    if mode == "create_only" or not resolved.exists():
        action = "create"
    elif mode == "append":
        action = "append to"
    else:
        action = "overwrite"

    request = _build_approval_request(
        tool_call,
        reason="Write operations require explicit approval before execution.",
        impact_summary=f"This request would {action} workspace file '{display_path}'.",
        diff_preview=_preview_text(raw_content),
        backup_plan=_backup_plan(config, run_id),
        affected_files=[display_path],
        suggested_verification_command=_suggest_verification_command(config, [display_path]),
        rollback_command=None if run_id is None else f"--rollback-run {run_id}",
    )
    return _needs_approval(request)


def _assess_apply_patch_call(
    tool_call: "ToolCall",
    config: "AgentConfig",
    *,
    run_id: str | None,
) -> PolicyAssessment:
    args = tool_call["args"]
    raw_patch = args.get("patch", args.get("diff"))
    if not isinstance(raw_patch, str) or not raw_patch.strip():
        return _deny("apply_patch is missing a patch payload")
    if _contains_binary_content(raw_patch):
        return _deny("apply_patch payload appears to be binary")
    if _byte_length(raw_patch) > config.max_patch_bytes:
        return _deny("patch payload exceeds max_patch_bytes")

    hunk_count = _count_patch_hunks(raw_patch)
    if hunk_count > config.max_patch_hunks:
        return _deny("patch payload exceeds max_patch_hunks")

    targets = _extract_patch_targets(raw_patch)
    if not targets:
        raw_path = args.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            targets = [("update", raw_path.strip())]

    if not targets:
        return _deny("apply_patch payload does not identify any target files")

    display_paths: list[str] = []
    counts = {"add": 0, "update": 0, "delete": 0}
    for action, raw_path in targets:
        if _is_probably_binary_path(raw_path):
            return _deny("apply_patch target appears to be a binary file")
        try:
            resolved = resolve_safe_path(
                config.workspace_root,
                raw_path,
                sensitive_path_parts=config.sensitive_path_parts,
            )
        except PolicyViolation as exc:
            return _deny(exc.reason)
        display_paths.append(_workspace_display_path(config, resolved))
        if action in counts:
            counts[action] += 1

    summary_parts: list[str] = []
    for action, label in (("add", "add"), ("update", "update"), ("delete", "delete")):
        count = counts[action]
        if count:
            summary_parts.append(f"{label} {count} file(s)")
    if not summary_parts:
        summary_parts.append(f"update {len(display_paths)} file(s)")

    request = _build_approval_request(
        tool_call,
        reason="Patch application requires explicit approval before execution.",
        impact_summary=(
            "This patch would "
            + ", ".join(summary_parts)
            + f": {', '.join(display_paths[:5])}."
        ),
        diff_preview=_preview_text(raw_patch),
        backup_plan=_backup_plan(config, run_id),
        affected_files=display_paths,
        suggested_verification_command=_suggest_verification_command(config, display_paths),
        rollback_command=None if run_id is None else f"--rollback-run {run_id}",
    )
    return _needs_approval(request)


def assess_tool_call(
    tool_call: "ToolCall",
    config: "AgentConfig",
    *,
    run_id: str | None = None,
) -> PolicyAssessment:
    """Return a structured policy assessment for the proposed tool call."""
    name: str = tool_call["name"]
    args: dict[str, object] = tool_call["args"]

    if name in WRITE_APPROVAL_TOOLS:
        if name == "write_file":
            return _assess_write_file_call(tool_call, config, run_id=run_id)
        return _assess_apply_patch_call(tool_call, config, run_id=run_id)

    if name in COMMAND_EXECUTION_TOOLS:
        raw_command = args.get("command")
        if not isinstance(raw_command, str):
            return _deny("run_command is missing a valid command string")
        raw_cwd = args.get("cwd", ".")
        raw_env = args.get("env")
        if raw_env is not None and not isinstance(raw_env, Mapping):
            return _deny("run_command env must be a mapping")
        decision = evaluate_command_call(
            raw_command,
            config,
            cwd=str(raw_cwd) if raw_cwd is not None else ".",
            env=raw_env,
        )
        if decision == "allow":
            write_assessment = _assess_run_command_write_effects(
                tool_call,
                config,
                cwd=str(raw_cwd) if raw_cwd is not None else ".",
                run_id=run_id,
            )
            if write_assessment is not None:
                return write_assessment
            return _allow()
        return _deny("command denied by command policy")

    if name not in READ_ONLY_TOOLS:
        return _deny(f"tool '{name}' is not in the allowlist")

    raw_path = args.get("path")
    if raw_path is not None:
        try:
            resolve_safe_path(
                config.workspace_root,
                str(raw_path),
                sensitive_path_parts=config.sensitive_path_parts,
            )
        except PolicyViolation as exc:
            return _deny(exc.reason)

    return _allow()


def classify_command(
    argv: list[str],
    config: "AgentConfig",
    *,
    cwd: str = ".",
) -> Literal["low", "medium", "high"]:
    """Classify a parsed command according to the configured command policy."""
    normalized = _normalize_command_text(argv)
    if not normalized:
        return "high"

    inner = _unwrap_uv_run(argv)
    if inner:
        inner_risk = classify_command(inner, config, cwd=cwd)
        if inner_risk != "high":
            return inner_risk

    if _is_safe_inline_python_command(argv, config, cwd=cwd):
        return "low"

    deny_match = _best_command_prefix_length(normalized, config.command_denylist)
    approval_match = _best_command_prefix_length(normalized, config.command_approvallist)
    allow_match = _best_command_prefix_length(normalized, config.command_allowlist)

    strongest_non_deny = max(
        length for length in (approval_match, allow_match) if length is not None
    ) if approval_match is not None or allow_match is not None else None

    if deny_match is not None and (strongest_non_deny is None or deny_match >= strongest_non_deny):
        return "high"
    if approval_match is not None and (allow_match is None or approval_match >= allow_match):
        return "medium"
    if allow_match is not None:
        return "low"
    return config.command_default_risk


def classify_command_sequence(
    segments: list[CommandSegment],
    config: "AgentConfig",
    *,
    cwd: str = ".",
) -> Literal["low", "medium", "high"]:
    """Classify a sequential command chain using the highest segment risk."""
    highest: Literal["low", "medium", "high"] = "low"
    for segment in segments:
        for stage in command_segment_stages(segment):
            risk = classify_command(stage["argv"], config, cwd=cwd)
            if risk == "high":
                return "high"
            if risk == "medium":
                highest = "medium"
    return highest


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


def _resolve_command_output_path(
    config: "AgentConfig",
    raw_path: str,
    *,
    cwd: str = ".",
) -> Path:
    output_path = Path(raw_path)
    if output_path.is_absolute():
        return resolve_safe_path(
            config.workspace_root,
            raw_path,
            config.sensitive_path_parts,
        )

    resolved_cwd = resolve_command_cwd(config, cwd)
    relative_cwd = resolved_cwd.relative_to(config.workspace_root)
    workspace_relative = relative_cwd / output_path
    return resolve_safe_path(
        config.workspace_root,
        str(workspace_relative),
        config.sensitive_path_parts,
    )


def _resolve_command_input_path(
    config: "AgentConfig",
    raw_path: str,
    *,
    cwd: str = ".",
) -> Path:
    input_path = Path(raw_path)
    if input_path.is_absolute():
        return resolve_safe_path(
            config.workspace_root,
            raw_path,
            config.sensitive_path_parts,
        )

    resolved_cwd = resolve_command_cwd(config, cwd)
    relative_cwd = resolved_cwd.relative_to(config.workspace_root)
    workspace_relative = relative_cwd / input_path
    return resolve_safe_path(
        config.workspace_root,
        str(workspace_relative),
        config.sensitive_path_parts,
    )


class _InlinePythonPolicyValidator(ast.NodeVisitor):
    def __init__(self, config: "AgentConfig", *, cwd: str) -> None:
        self._config = config
        self._cwd = cwd
        self._parents: dict[ast.AST, ast.AST] = {}
        self._reserved_names = (
            _INLINE_PYTHON_SAFE_IMPORTS | _INLINE_PYTHON_SAFE_BUILTINS
        )

    def validate(self, tree: ast.AST) -> bool:
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self._parents[child] = parent
        self.visit(tree)
        return True

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _INLINE_PYTHON_SAFE_NODE_TYPES):
            raise PolicyViolation(
                "inline python command is not in the safe read-only subset"
            )
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in _INLINE_PYTHON_SAFE_IMPORTS or alias.asname is not None:
                raise PolicyViolation(
                    "inline python command is not in the safe read-only subset"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parent = self._parents.get(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "json"
            and node.attr in _INLINE_PYTHON_SAFE_JSON_CALLS
            and isinstance(parent, ast.Call)
            and parent.func is node
        ):
            self.generic_visit(node)
            return
        raise PolicyViolation("inline python command is not in the safe read-only subset")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise PolicyViolation("inline python command is not in the safe read-only subset")
        if isinstance(node.ctx, ast.Store) and node.id in self._reserved_names:
            raise PolicyViolation("inline python command is not in the safe read-only subset")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id == "open":
                self._validate_open_call(node)
            elif node.func.id == "print":
                if node.keywords:
                    raise PolicyViolation(
                        "inline python command is not in the safe read-only subset"
                    )
            elif node.func.id == "len":
                if len(node.args) != 1 or node.keywords:
                    raise PolicyViolation(
                        "inline python command is not in the safe read-only subset"
                    )
            else:
                raise PolicyViolation(
                    "inline python command is not in the safe read-only subset"
                )
        elif isinstance(node.func, ast.Attribute):
            if not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
                and node.func.attr in _INLINE_PYTHON_SAFE_JSON_CALLS
                and len(node.args) == 1
                and not node.keywords
            ):
                raise PolicyViolation(
                    "inline python command is not in the safe read-only subset"
                )
        else:
            raise PolicyViolation("inline python command is not in the safe read-only subset")
        self.generic_visit(node)

    def _validate_open_call(self, node: ast.Call) -> None:
        if not node.args or len(node.args) > 2 or node.keywords:
            raise PolicyViolation("inline python command is not in the safe read-only subset")
        path_arg = node.args[0]
        if not isinstance(path_arg, ast.Constant) or not isinstance(path_arg.value, str):
            raise PolicyViolation("inline python command is not in the safe read-only subset")

        mode = "r"
        if len(node.args) == 2:
            mode_arg = node.args[1]
            if not isinstance(mode_arg, ast.Constant) or not isinstance(mode_arg.value, str):
                raise PolicyViolation(
                    "inline python command is not in the safe read-only subset"
                )
            mode = mode_arg.value

        if mode not in _INLINE_PYTHON_SAFE_READ_MODES:
            raise PolicyViolation("inline python command is not in the safe read-only subset")

        _resolve_command_input_path(self._config, path_arg.value, cwd=self._cwd)


def _is_safe_inline_python_command(
    argv: list[str],
    config: "AgentConfig",
    *,
    cwd: str = ".",
) -> bool:
    inner = _unwrap_uv_run(argv) or argv
    if len(inner) != 3:
        return False
    if inner[0] not in _INLINE_PYTHON_EXECUTABLES or inner[1] != "-c":
        return False

    try:
        tree = ast.parse(inner[2], mode="exec")
        return _InlinePythonPolicyValidator(config, cwd=cwd).validate(tree)
    except (PolicyViolation, SyntaxError, ValueError):
        return False


def _extract_curl_output_targets(argv: list[str]) -> tuple[list[str], bool]:
    inner = _unwrap_uv_run(argv) or argv
    if not inner or inner[0] != "curl":
        return [], False

    targets: list[str] = []
    writes_to_cwd = False
    index = 1
    while index < len(inner):
        arg = inner[index]
        if arg in {"-o", "--output"}:
            if index + 1 >= len(inner):
                raise PolicyViolation("curl output flag is missing a path")
            targets.append(inner[index + 1])
            index += 2
            continue
        if arg.startswith("--output="):
            output = arg.split("=", 1)[1]
            if not output:
                raise PolicyViolation("curl output flag is missing a path")
            targets.append(output)
            index += 1
            continue
        if arg.startswith("-o") and len(arg) > 2:
            targets.append(arg[2:])
            index += 1
            continue
        if arg in {"-O", "--remote-name", "--remote-name-all"}:
            writes_to_cwd = True
        index += 1

    return targets, writes_to_cwd


def _extract_redirect_output_target(stage: CommandStage) -> str | None:
    target = stage.get("stdout_path")
    if not isinstance(target, str) or not target.strip():
        return None
    return target


def _assess_run_command_write_effects(
    tool_call: "ToolCall",
    config: "AgentConfig",
    *,
    cwd: str = ".",
    run_id: str | None,
) -> PolicyAssessment | None:
    raw_command = str(tool_call["args"].get("command", "") or "")
    segments = parse_command_sequence(raw_command)
    display_paths: list[str] = []
    seen_paths: set[str] = set()
    writes_to_dirs: list[str] = []
    seen_dirs: set[str] = set()

    for segment in segments:
        for stage in command_segment_stages(segment):
            output_targets, writes_to_cwd = _extract_curl_output_targets(stage["argv"])
            redirect_target = _extract_redirect_output_target(stage)
            if redirect_target is not None:
                output_targets = [*output_targets, redirect_target]
            for raw_path in output_targets:
                try:
                    resolved = _resolve_command_output_path(config, raw_path, cwd=cwd)
                except PolicyViolation as exc:
                    return _deny(exc.reason)
                display_path = _workspace_display_path(config, resolved)
                if display_path not in seen_paths:
                    display_paths.append(display_path)
                    seen_paths.add(display_path)
            if writes_to_cwd:
                resolved_cwd = resolve_command_cwd(config, cwd)
                display_dir = _workspace_display_path(config, resolved_cwd)
                if display_dir not in seen_dirs:
                    writes_to_dirs.append(display_dir)
                    seen_dirs.add(display_dir)

    if not display_paths and not writes_to_dirs:
        return None

    if not config.write_requires_approval:
        return None

    if display_paths:
        impact_summary = (
            "This command would write workspace file(s): "
            + ", ".join(display_paths[:5])
            + ("." if len(display_paths) <= 5 else f" (+{len(display_paths) - 5} more).")
        )
        affected_files = display_paths
    else:
        impact_summary = (
            "This command would write downloaded file(s) into workspace directory "
            f"'{writes_to_dirs[0]}'."
        )
        affected_files = writes_to_dirs

    request = _build_approval_request(
        tool_call,
        reason="Command execution that writes files requires explicit approval before execution.",
        impact_summary=impact_summary,
        diff_preview=_preview_text(raw_command),
        backup_plan="No automatic backup is available for run_command writes.",
        affected_files=affected_files,
        suggested_verification_command=_suggest_verification_command(config, display_paths),
        rollback_command=None if run_id is None else f"--rollback-run {run_id}",
    )
    return _needs_approval(request)


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
        segments = parse_command_sequence(command)
        resolve_command_cwd(config, cwd)
        if env is not None and not isinstance(env, Mapping):
            return "deny"
        risk = classify_command_sequence(segments, config, cwd=cwd)
    except PolicyViolation:
        return "deny"

    return "allow" if risk == "low" else "deny"


def evaluate_tool_call(
    tool_call: "ToolCall",
    config: "AgentConfig",
) -> Literal["allow", "deny", "needs_approval"]:
    """
    Evaluate a proposed tool call against the current security policy.

    Returns ``"allow"``, ``"deny"``, or ``"needs_approval"``.
    Does **not** raise – callers receive a string decision so they can log
    the reason and route the graph accordingly.
    """
    return assess_tool_call(tool_call, config)["decision"]

