"""Text-writing skills for the Linux Agent."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Literal
from uuid import uuid4

from linux_agent.config import AgentConfig
from linux_agent.policy import PolicyViolation, resolve_safe_path

_PATCH_BEGIN = "*** Begin Patch"
_PATCH_END = "*** End Patch"
_ADD_FILE_PREFIX = "*** Add File: "
_UPDATE_FILE_PREFIX = "*** Update File: "
_DELETE_FILE_PREFIX = "*** Delete File: "
_MANIFEST_FILENAME = "manifest.json"
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


@dataclass(slots=True)
class _PatchSection:
    action: Literal["add", "update", "delete"]
    path: str
    body: list[str]


@dataclass(slots=True)
class _VirtualFileState:
    safe_path: Path
    display_path: str
    original_text: str | None
    current_text: str | None
    added_lines: int = 0
    removed_lines: int = 0


def _byte_length(text: str) -> int:
    return len(text.encode("utf-8"))


def _contains_nul_bytes(text: str) -> bool:
    return "\x00" in text


def _count_patch_hunks(patch_text: str) -> int:
    hunk_count = sum(1 for line in patch_text.splitlines() if line.startswith("@@"))
    if hunk_count:
        return hunk_count
    section_count = sum(
        1
        for line in patch_text.splitlines()
        if line.startswith(_ADD_FILE_PREFIX) or line.startswith(_UPDATE_FILE_PREFIX)
    )
    return section_count if section_count else 1


def _display_path(config: AgentConfig, safe_path: Path) -> str:
    try:
        relative = safe_path.relative_to(config.workspace_root)
    except ValueError:
        return str(safe_path)
    return relative.as_posix() or "."


def _is_probably_binary_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _BINARY_PATH_SUFFIXES


def _read_text_file(path: Path) -> tuple[str | None, str | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, str(exc)

    if b"\x00" in raw:
        return None, "binary files are not supported"

    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "file is not valid UTF-8 text"


def _lines_to_text(lines: list[str], *, trailing_newline: bool) -> str:
    if not lines:
        return ""
    text = "\n".join(lines)
    if trailing_newline:
        text += "\n"
    return text


def _backup_root(config: AgentConfig, run_id: str | None) -> Path:
    root = config.backup_dir
    if not root.is_absolute():
        root = config.workspace_root / root
    run_segment = run_id or f"write-{uuid4().hex[:8]}"
    return root / run_segment


def _create_backup(source: Path, config: AgentConfig, run_id: str | None) -> Path:
    backup_root = _backup_root(config, run_id)
    relative = source.relative_to(config.workspace_root)
    destination = backup_root / relative
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _manifest_path(config: AgentConfig, run_id: str | None) -> Path:
    return _backup_root(config, run_id) / _MANIFEST_FILENAME


def _load_manifest(config: AgentConfig, run_id: str | None) -> dict[str, Any]:
    manifest_path = _manifest_path(config, run_id)
    if not manifest_path.exists():
        return {
            "run_id": run_id or manifest_path.parent.name,
            "entries": [],
        }

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"write manifest is invalid: {manifest_path}")
    return payload


def _write_manifest(config: AgentConfig, run_id: str | None, manifest: dict[str, Any]) -> Path:
    manifest_path = _manifest_path(config, run_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        template_path=manifest_path if manifest_path.exists() else None,
    )
    return manifest_path


def _ensure_manifest_entry(
    config: AgentConfig,
    run_id: str | None,
    *,
    safe_path: Path,
    display_path: str,
    action: str,
    created: bool,
) -> tuple[dict[str, Any], Path]:
    manifest = _load_manifest(config, run_id)
    entries = manifest.setdefault("entries", [])
    for entry in entries:
        if entry.get("path") == display_path:
            manifest_path = _write_manifest(config, run_id, manifest)
            return entry, manifest_path

    backup_path: str | None = None
    if not created and safe_path.exists():
        backup_path = str(_create_backup(safe_path, config, run_id))

    entry = {
        "path": display_path,
        "action": action,
        "created": created,
        "backup_path": backup_path,
    }
    entries.append(entry)
    manifest_path = _write_manifest(config, run_id, manifest)
    return entry, manifest_path


def _atomic_write_text(path: Path, content: str, *, template_path: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=path.parent,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)

        if template_path is not None and template_path.exists():
            mode = stat.S_IMODE(template_path.stat().st_mode)
            os.chmod(temp_path, mode)

        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_restore_file(path: Path, backup_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(delete=False, dir=path.parent)
    temp_path = Path(handle.name)
    handle.close()
    try:
        shutil.copy2(backup_path, temp_path)
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _build_diff(display_path: str, old_text: str | None, new_text: str) -> str:
    diff_lines = list(
        unified_diff(
            [] if old_text is None else old_text.splitlines(),
            new_text.splitlines(),
            fromfile="/dev/null" if old_text is None else f"a/{display_path}",
            tofile=f"b/{display_path}",
            lineterm="",
        )
    )
    return "\n".join(diff_lines)


def _parse_patch_sections(patch_text: str) -> tuple[list[_PatchSection], str | None]:
    lines = patch_text.splitlines()
    if not lines or lines[0] != _PATCH_BEGIN:
        return [], "patch must begin with '*** Begin Patch'"
    if lines[-1] != _PATCH_END:
        return [], "patch must end with '*** End Patch'"

    sections: list[_PatchSection] = []
    current: _PatchSection | None = None

    for raw_line in lines[1:-1]:
        if raw_line.startswith(_ADD_FILE_PREFIX):
            if current is not None:
                sections.append(current)
            current = _PatchSection("add", raw_line[len(_ADD_FILE_PREFIX) :].strip(), [])
            continue

        if raw_line.startswith(_UPDATE_FILE_PREFIX):
            if current is not None:
                sections.append(current)
            current = _PatchSection(
                "update",
                raw_line[len(_UPDATE_FILE_PREFIX) :].strip(),
                [],
            )
            continue

        if raw_line.startswith(_DELETE_FILE_PREFIX):
            if current is not None:
                sections.append(current)
            current = _PatchSection(
                "delete",
                raw_line[len(_DELETE_FILE_PREFIX) :].strip(),
                [],
            )
            continue

        if current is None:
            if raw_line.strip():
                return [], "patch contains content before the first file header"
            continue

        current.body.append(raw_line)

    if current is not None:
        sections.append(current)

    if not sections:
        return [], "patch contains no file sections"

    for section in sections:
        if not section.path:
            return [], "patch contains a file section without a path"

    return sections, None


def _parse_add_body(body: list[str]) -> tuple[list[str], str | None]:
    lines: list[str] = []
    for raw_line in body:
        if raw_line.startswith("@@"):
            continue
        if not raw_line.startswith("+"):
            return [], "Add File sections must use '+' prefixes for file content"
        lines.append(raw_line[1:])
    return lines, None


def _apply_update_section(
    current_text: str,
    body: list[str],
) -> tuple[str | None, int, int, str | None]:
    pre_context: list[str] = []
    removed_lines: list[str] = []
    added_lines: list[str] = []
    post_context: list[str] = []
    phase: Literal["pre", "change", "post"] = "pre"

    for raw_line in body:
        if raw_line.startswith("@@"):
            continue

        if raw_line.startswith("-"):
            if phase == "post":
                return None, 0, 0, "multiple change groups in one Update File section are not supported"
            phase = "change"
            removed_lines.append(raw_line[1:])
            continue

        if raw_line.startswith("+"):
            if phase == "post":
                return None, 0, 0, "multiple change groups in one Update File section are not supported"
            phase = "change"
            added_lines.append(raw_line[1:])
            continue

        if phase == "pre":
            pre_context.append(raw_line)
            continue

        if phase == "change":
            phase = "post"
        post_context.append(raw_line)

    if not removed_lines and not added_lines:
        return None, 0, 0, "Update File section contains no added or removed lines"

    old_block = pre_context + removed_lines + post_context
    new_block = pre_context + added_lines + post_context
    if not old_block:
        return None, 0, 0, "Update File section must include context or removed lines"

    current_lines = current_text.splitlines()
    window = len(old_block)
    matches = [
        index
        for index in range(len(current_lines) - window + 1)
        if current_lines[index : index + window] == old_block
    ]
    if not matches:
        return None, 0, 0, "patch hunk context did not match the target file"
    if len(matches) > 1:
        return None, 0, 0, "patch hunk matched multiple locations; add more context"

    index = matches[0]
    next_lines = current_lines[:index] + new_block + current_lines[index + window :]
    next_text = _lines_to_text(next_lines, trailing_newline=current_text.endswith("\n"))
    return next_text, len(added_lines), len(removed_lines), None


def _empty_apply_patch_result(error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "changed_files": [],
        "added_lines": 0,
        "removed_lines": 0,
        "diff": "",
        "backup_paths": [],
        "backup_root": None,
        "manifest_path": None,
        "file_summaries": [],
        "rolled_back": False,
        "restored_files": [],
        "removed_files": [],
        "error": error,
    }


def _rollback_manifest_entries(
    entries: list[dict[str, Any]],
    config: AgentConfig,
) -> dict[str, Any]:
    restored_files: list[str] = []
    removed_files: list[str] = []

    for entry in reversed(entries):
        display_path = str(entry.get("path", ""))
        safe_path = resolve_safe_path(
            config.workspace_root,
            display_path,
            config.sensitive_path_parts,
        )
        backup_path = entry.get("backup_path")
        if backup_path:
            _atomic_restore_file(safe_path, Path(str(backup_path)))
            restored_files.append(display_path)
            continue
        if safe_path.exists():
            safe_path.unlink()
            removed_files.append(display_path)

    return {
        "ok": True,
        "restored_files": restored_files,
        "removed_files": removed_files,
        "error": None,
    }


def rollback_run(run_id: str, config: AgentConfig) -> dict[str, Any]:
    manifest_path = _manifest_path(config, run_id)
    if not manifest_path.exists():
        return {
            "ok": False,
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "backup_root": str(_backup_root(config, run_id)),
            "restored_files": [],
            "removed_files": [],
            "error": "write manifest not found",
        }

    manifest = _load_manifest(config, run_id)
    entries = [entry for entry in manifest.get("entries", []) if isinstance(entry, dict)]
    try:
        rollback_result = _rollback_manifest_entries(entries, config)
    except Exception as exc:
        return {
            "ok": False,
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "backup_root": str(_backup_root(config, run_id)),
            "restored_files": [],
            "removed_files": [],
            "error": str(exc),
        }

    return {
        "ok": True,
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "backup_root": str(_backup_root(config, run_id)),
        "restored_files": rollback_result["restored_files"],
        "removed_files": rollback_result["removed_files"],
        "error": None,
    }


def apply_patch(
    patch_text: str,
    config: AgentConfig,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Apply a constrained patch using the repository's patch format."""
    effective_run_id = run_id or f"write-{uuid4().hex[:8]}"
    if not patch_text.strip():
        return _empty_apply_patch_result("patch payload is empty")
    if _contains_nul_bytes(patch_text):
        return _empty_apply_patch_result("patch payload appears to be binary")
    if _byte_length(patch_text) > config.max_patch_bytes:
        return _empty_apply_patch_result("patch payload exceeds max_patch_bytes")
    if _count_patch_hunks(patch_text) > config.max_patch_hunks:
        return _empty_apply_patch_result("patch payload exceeds max_patch_hunks")

    sections, parse_error = _parse_patch_sections(patch_text)
    if parse_error is not None:
        return _empty_apply_patch_result(parse_error)

    states: dict[Path, _VirtualFileState] = {}

    for section in sections:
        if section.action == "delete":
            return _empty_apply_patch_result("Delete File sections are not supported yet")

        try:
            safe_path = resolve_safe_path(
                config.workspace_root,
                section.path,
                config.sensitive_path_parts,
            )
        except PolicyViolation as exc:
            return _empty_apply_patch_result(str(exc))
        display_path = _display_path(config, safe_path)

        if _is_probably_binary_path(display_path):
            return _empty_apply_patch_result("binary files are not supported")

        state = states.get(safe_path)
        if state is None:
            original_text: str | None = None
            if safe_path.exists():
                if safe_path.is_dir():
                    return _empty_apply_patch_result("target path is a directory")
                original_text, read_error = _read_text_file(safe_path)
                if read_error is not None:
                    return _empty_apply_patch_result(read_error)
            state = _VirtualFileState(
                safe_path=safe_path,
                display_path=display_path,
                original_text=original_text,
                current_text=original_text,
            )
            states[safe_path] = state

        if section.action == "add":
            if state.original_text is not None or state.current_text is not None:
                return _empty_apply_patch_result(f"file already exists: {display_path}")
            if not safe_path.parent.exists() or not safe_path.parent.is_dir():
                return _empty_apply_patch_result("parent directory does not exist")

            added_content, add_error = _parse_add_body(section.body)
            if add_error is not None:
                return _empty_apply_patch_result(add_error)

            state.current_text = _lines_to_text(added_content, trailing_newline=bool(added_content))
            state.added_lines += len(added_content)
            continue

        if state.current_text is None:
            return _empty_apply_patch_result(f"file not found: {display_path}")

        next_text, added_count, removed_count, update_error = _apply_update_section(
            state.current_text,
            section.body,
        )
        if update_error is not None or next_text is None:
            return _empty_apply_patch_result(update_error or "failed to apply update section")

        state.current_text = next_text
        state.added_lines += added_count
        state.removed_lines += removed_count

    changed_states = [state for state in states.values() if state.current_text != state.original_text]
    if not changed_states:
        return _empty_apply_patch_result("patch produced no file changes")

    backup_paths: list[str] = []
    file_summaries: list[dict[str, Any]] = []
    diff_parts: list[str] = []
    manifest_path: Path | None = None
    applied_entries: list[dict[str, Any]] = []

    try:
        for state in changed_states:
            original_text = state.original_text
            current_text = state.current_text or ""
            action = "add" if original_text is None else "update"
            entry, manifest_path = _ensure_manifest_entry(
                config,
                effective_run_id,
                safe_path=state.safe_path,
                display_path=state.display_path,
                action=action,
                created=original_text is None,
            )
            applied_entries.append(entry)
            if entry.get("backup_path"):
                backup_paths.append(str(entry["backup_path"]))

            _atomic_write_text(
                state.safe_path,
                current_text,
                template_path=state.safe_path if state.safe_path.exists() else None,
            )

            diff_parts.append(_build_diff(state.display_path, original_text, current_text))
            file_summaries.append(
                {
                    "path": state.display_path,
                    "action": action,
                    "added_lines": state.added_lines,
                    "removed_lines": state.removed_lines,
                }
            )
    except Exception as exc:
        rollback_result = {
            "restored_files": [],
            "removed_files": [],
        }
        if applied_entries:
            rollback_result = _rollback_manifest_entries(applied_entries, config)
        return {
            **_empty_apply_patch_result(str(exc)),
            "backup_paths": backup_paths,
            "backup_root": str(_backup_root(config, effective_run_id)),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "rolled_back": bool(applied_entries),
            "restored_files": rollback_result["restored_files"],
            "removed_files": rollback_result["removed_files"],
        }

    return {
        "ok": True,
        "changed_files": [summary["path"] for summary in file_summaries],
        "added_lines": sum(summary["added_lines"] for summary in file_summaries),
        "removed_lines": sum(summary["removed_lines"] for summary in file_summaries),
        "diff": "\n\n".join(part for part in diff_parts if part),
        "backup_paths": backup_paths,
        "backup_root": str(_backup_root(config, effective_run_id)),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "file_summaries": file_summaries,
        "rolled_back": False,
        "restored_files": [],
        "removed_files": [],
        "error": None,
    }


def write_file(
    path: str,
    content: str,
    config: AgentConfig,
    *,
    mode: str = "overwrite",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Write bounded text content to a workspace file after approval."""
    effective_run_id = run_id or f"write-{uuid4().hex[:8]}"
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"create_only", "append", "overwrite"}:
        return {
            "ok": False,
            "path": path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": f"unsupported write_file mode: {normalized_mode}",
        }

    if _contains_nul_bytes(content):
        return {
            "ok": False,
            "path": path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": "write_file content appears to be binary",
        }

    if _byte_length(content) > config.max_patch_bytes:
        return {
            "ok": False,
            "path": path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": "write_file content exceeds max_patch_bytes",
        }

    try:
        safe_path = resolve_safe_path(
            config.workspace_root,
            path,
            config.sensitive_path_parts,
        )
    except PolicyViolation as exc:
        return {
            "ok": False,
            "path": path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": str(exc),
        }
    display_path = _display_path(config, safe_path)
    if _is_probably_binary_path(display_path):
        return {
            "ok": False,
            "path": display_path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": "binary files are not supported",
        }

    if not safe_path.parent.exists() or not safe_path.parent.is_dir():
        return {
            "ok": False,
            "path": display_path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": "parent directory does not exist",
        }

    if safe_path.exists() and safe_path.is_dir():
        return {
            "ok": False,
            "path": display_path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": "target path is a directory",
        }

    old_text: str | None = None
    if safe_path.exists():
        old_text, read_error = _read_text_file(safe_path)
        if read_error is not None:
            return {
                "ok": False,
                "path": display_path,
                "mode": normalized_mode,
                "changed_files": [],
                "added_lines": 0,
                "removed_lines": 0,
                "diff": "",
                "backup_paths": [],
                "backup_root": None,
                "manifest_path": None,
                "created": False,
                "bytes_written": 0,
                "rolled_back": False,
                "restored_files": [],
                "removed_files": [],
                "error": read_error,
            }

    if normalized_mode == "create_only" and old_text is not None:
        return {
            "ok": False,
            "path": display_path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": "file already exists",
        }

    if normalized_mode == "append" and old_text is None:
        return {
            "ok": False,
            "path": display_path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": "file not found",
        }

    if normalized_mode == "append":
        new_text = (old_text or "") + content
    else:
        new_text = content

    if old_text is not None and new_text == old_text:
        return {
            "ok": True,
            "path": display_path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": [],
            "backup_root": None,
            "manifest_path": None,
            "created": False,
            "bytes_written": 0,
            "rolled_back": False,
            "restored_files": [],
            "removed_files": [],
            "error": None,
        }

    backup_paths: list[str] = []
    entry, manifest_path = _ensure_manifest_entry(
        config,
        effective_run_id,
        safe_path=safe_path,
        display_path=display_path,
        action=normalized_mode,
        created=old_text is None,
    )
    if entry.get("backup_path"):
        backup_paths.append(str(entry["backup_path"]))

    try:
        _atomic_write_text(
            safe_path,
            new_text,
            template_path=safe_path if safe_path.exists() else None,
        )
    except Exception as exc:
        rollback_result = _rollback_manifest_entries([entry], config)
        return {
            "ok": False,
            "path": display_path,
            "mode": normalized_mode,
            "changed_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "diff": "",
            "backup_paths": backup_paths,
            "backup_root": str(_backup_root(config, effective_run_id)),
            "manifest_path": str(manifest_path),
            "created": False,
            "bytes_written": 0,
            "rolled_back": True,
            "restored_files": rollback_result["restored_files"],
            "removed_files": rollback_result["removed_files"],
            "error": str(exc),
        }

    diff = _build_diff(display_path, old_text, new_text)
    if normalized_mode == "create_only":
        added_lines = len(new_text.splitlines())
        removed_lines = 0
    elif normalized_mode == "append":
        added_lines = len(content.splitlines())
        removed_lines = 0
    else:
        added_lines = sum(
            1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
        )
        removed_lines = sum(
            1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")
        )

    return {
        "ok": True,
        "path": display_path,
        "mode": normalized_mode,
        "changed_files": [display_path],
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "diff": diff,
        "backup_paths": backup_paths,
        "backup_root": str(_backup_root(config, effective_run_id)),
        "manifest_path": str(manifest_path),
        "created": old_text is None,
        "bytes_written": _byte_length(content),
        "rolled_back": False,
        "restored_files": [],
        "removed_files": [],
        "error": None,
    }