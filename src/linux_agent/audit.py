"""
Audit logging for the Linux Agent.

Every agent run writes a JSONL file to <log_dir>/<run_id>.jsonl.
Each line is a self-contained JSON object (AuditEvent) that can be
streamed, grep-ed, or replayed without parsing the entire file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from typing_extensions import TypedDict


class AuditEvent(TypedDict):
    """Schema for a single audit log line."""

    run_id: str
    ts: str        # ISO 8601 UTC timestamp
    event: str     # event type name (see EVENT_* constants below)
    data: dict[str, Any]


# ── canonical event type names ──────────────────────────────────────────────
EVENT_RUN_START = "run_start"
EVENT_PLAN_UPDATE = "plan_update"
EVENT_TOOL_PROPOSED = "tool_proposed"
EVENT_POLICY_DECISION = "policy_decision"
EVENT_TOOL_RESULT = "tool_result"
EVENT_REFLECTOR_ACTION = "reflector_action"
EVENT_RUN_END = "run_end"


class AuditLogger:
    """
    Append-only JSONL audit logger scoped to a single agent run.

    Usage
    -----
    Direct::

        logger = AuditLogger(run_id="abc", log_dir=Path("logs"))
        logger.log(EVENT_RUN_START, {"user_goal": "..."})
        logger.close()

    As context manager::

        with AuditLogger(run_id="abc", log_dir=Path("logs")) as logger:
            logger.log(EVENT_PLAN_UPDATE, {"plan": [...]})
    """

    def __init__(self, run_id: str, log_dir: Path) -> None:
        self._run_id = run_id
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{run_id}.jsonl"
        # Open in append mode so that resuming a run keeps history intact
        self._fh = log_path.open("a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event: str, data: dict[str, Any]) -> None:
        """Write one audit event to the JSONL file."""
        record: AuditEvent = {
            "run_id": self._run_id,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "event": event,
            "data": data,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()  # ensure visibility even if the process crashes

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    @property
    def log_path(self) -> Path:
        """Absolute path to the JSONL log file for this run."""
        return self._log_dir / f"{self._run_id}.jsonl"

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

