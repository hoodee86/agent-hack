"""
Audit logging for the Linux Agent.

Every agent run writes a JSONL file to <log_dir>/<run_id>.jsonl.
Each line is a self-contained JSON object (AuditEvent) that can be
streamed, grep-ed, or replayed without parsing the entire file.
"""

from __future__ import annotations

from collections.abc import Callable
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


AuditEventListener = Callable[[AuditEvent], None]


# ── canonical event type names ──────────────────────────────────────────────
EVENT_RUN_START = "run_start"
EVENT_MODEL_INPUT = "model_input"
EVENT_PLAN_UPDATE = "plan_update"
EVENT_TOOL_PROPOSED = "tool_proposed"
EVENT_POLICY_DECISION = "policy_decision"
EVENT_APPROVAL_REQUESTED = "approval_requested"
EVENT_APPROVAL_PRESENTED = "approval_presented"
EVENT_APPROVAL_RESPONSE = "approval_response"
EVENT_TOOL_RESULT = "tool_result"
EVENT_WRITE_APPLIED = "write_applied"
EVENT_WRITE_ROLLBACK = "write_rollback"
EVENT_REFLECTOR_ACTION = "reflector_action"
EVENT_PLAN_REVISED = "plan_revised"
EVENT_REFLECTION_SCORED = "reflection_scored"
EVENT_RECOVERY_ATTEMPTED = "recovery_attempted"
EVENT_RECOVERY_EXHAUSTED = "recovery_exhausted"
EVENT_RECOVERY_CLEARED = "recovery_cleared"
EVENT_BUDGET_WARNING = "budget_warning"
EVENT_BUDGET_EXHAUSTED = "budget_exhausted"
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

    def __init__(
        self,
        run_id: str,
        log_dir: Path,
        listener: AuditEventListener | None = None,
    ) -> None:
        self._run_id = run_id
        self._log_dir = log_dir
        self._listener = listener
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
        if self._listener is not None:
            self._listener(record)

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

