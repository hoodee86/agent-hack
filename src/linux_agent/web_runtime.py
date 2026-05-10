"""Background run management and data shaping for the Linux Agent web console."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import threading
from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from linux_agent.approval_ui import build_approval_view
from linux_agent.audit import AuditEvent, AuditLogger, EVENT_RUN_END, EVENT_RUN_START
from linux_agent.config import AgentConfig, load_config
from linux_agent.graph import build_graph
from linux_agent.run_store import load_run_state, state_dir, state_path
from linux_agent.state import AgentState


def _preview_text(value: Any, *, limit: int = 220) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    stripped = text.strip()
    if not stripped:
        return None
    compact = "\n".join(line.rstrip() for line in stripped.splitlines()[:10]).strip()
    if len(compact) > limit:
        return compact[:limit].rstrip() + " …"
    return compact


def _override_workspace(config: AgentConfig, workspace_root: Path) -> AgentConfig:
    return AgentConfig(
        **{
            **config.model_dump(),
            "workspace_root": workspace_root,
        }
    )


def _resolve_workspace(workspace: str | None) -> Path | None:
    if workspace is None:
        return None
    resolved = Path(workspace).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"workspace does not exist or is not a directory: {workspace}")
    return resolved


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _default_initial_state(run_id: str, goal: str, config: AgentConfig) -> AgentState:
    return AgentState(
        run_id=run_id,
        user_goal=goal,
        workspace_root=str(config.workspace_root),
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        messages=[],
        plan=[],
        command_count=0,
        plan_version=0,
        plan_revision_count=0,
        plan_steps=[],
        last_reflection=None,
        recovery_state=None,
        recovery_attempt_total=0,
        budget_status={
            "iteration_count": 0,
            "command_count": 0,
            "elapsed_seconds": 0,
            "warning_triggered": False,
        },
        budget_stop_reason=None,
        current_step=None,
        proposed_tool_call=None,
        observations=[],
        risk_decision=None,
        pending_approval=None,
        resume_action=None,
        approval_response_note=None,
        pending_verification=None,
        last_write=None,
        last_verification=None,
        last_rollback=None,
        iteration_count=0,
        consecutive_failures=0,
        final_answer=None,
    )


def _load_config_with_optional_workspace(
    default_config_path: str | None,
    *,
    config_path: str | None = None,
    workspace: str | None = None,
) -> AgentConfig:
    config = load_config(config_path or default_config_path)
    workspace_root = _resolve_workspace(workspace)
    if workspace_root is not None:
        return _override_workspace(config, workspace_root)
    return config


def _log_run_start(
    run_id: str,
    config: AgentConfig,
    *,
    user_goal: str | None,
    mode: str,
    resume_action: str | None,
) -> None:
    with AuditLogger(run_id, config.log_dir) as audit:
        audit.log(
            EVENT_RUN_START,
            {
                "user_goal": user_goal,
                "workspace_root": str(config.workspace_root),
                "mode": mode,
                "resume_action": resume_action,
                "config": {
                    "max_iterations": config.max_iterations,
                    "max_consecutive_failures": config.max_consecutive_failures,
                    "llm_model": config.llm_model,
                },
            },
        )


def _log_runtime_error(run_id: str, config: AgentConfig, error: str) -> None:
    with AuditLogger(run_id, config.log_dir) as audit:
        audit.log(
            EVENT_RUN_END,
            {
                "status": "runtime_error",
                "final_answer": f"Runtime error: {error}",
                "iteration_count": 0,
                "observation_count": 0,
                "command_count": 0,
                "command_summaries": [],
                "write_count": 0,
                "write_summaries": [],
                "approval_request_id": None,
            },
        )


def _state_path_if_exists(config: AgentConfig, run_id: str) -> Path | None:
    path = state_path(run_id, config)
    return path if path.exists() else None


def _load_state_if_exists(config: AgentConfig, run_id: str) -> AgentState | None:
    try:
        return load_run_state(run_id, config)
    except (FileNotFoundError, ValueError):
        return None


def _read_events(log_path: Path) -> list[AuditEvent]:
    if not log_path.exists():
        return []
    events: list[AuditEvent] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        events.append(cast(AuditEvent, payload))
    return events


@dataclass
class JobRecord:
    run_id: str
    mode: Literal["new", "resume"]
    user_goal: str | None
    started_at: str
    thread: threading.Thread
    resume_action: str | None = None
    error: str | None = None
    finished_at: str | None = None

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "user_goal": self.user_goal,
            "resume_action": self.resume_action,
            "started_at": self.started_at,
            "running": self.running,
            "error": self.error,
            "finished_at": self.finished_at,
        }


class WebRuntime:
    """Coordinates background graph runs and exposes web-friendly run data."""

    def __init__(self, default_config_path: str | None = None) -> None:
        self.default_config_path = default_config_path
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def _register_job(self, job: JobRecord) -> None:
        with self._lock:
            self._jobs[job.run_id] = job

    def _finish_job(self, run_id: str, *, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return
            job.error = error
            job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()

    def _get_job(self, run_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(run_id)

    def _ensure_idle(self, run_id: str) -> None:
        job = self._get_job(run_id)
        if job is not None and job.running:
            raise ValueError(f"run {run_id} is already active")

    def _invoke_graph(
        self,
        run_id: str,
        config: AgentConfig,
        initial_state: AgentState,
        *,
        user_goal: str | None,
        mode: str,
        resume_action: str | None,
    ) -> None:
        _log_run_start(
            run_id,
            config,
            user_goal=user_goal,
            mode=mode,
            resume_action=resume_action,
        )
        graph_app = build_graph(config)
        graph_app.invoke(initial_state)

    def _run_new_job(
        self,
        run_id: str,
        goal: str,
        config_path: str | None,
        workspace: str | None,
    ) -> None:
        config: AgentConfig | None = None
        try:
            config = _load_config_with_optional_workspace(
                self.default_config_path,
                config_path=config_path,
                workspace=workspace,
            )
            initial_state = _default_initial_state(run_id, goal, config)
            self._invoke_graph(
                run_id,
                config,
                initial_state,
                user_goal=goal,
                mode="new",
                resume_action=None,
            )
            self._finish_job(run_id)
        except Exception as exc:  # noqa: BLE001
            if config is not None:
                _log_runtime_error(run_id, config, str(exc))
            self._finish_job(run_id, error=str(exc))

    def _run_resume_job(
        self,
        run_id: str,
        config: AgentConfig,
        initial_state: AgentState,
        *,
        user_goal: str,
        decision: Literal["approve", "reject"],
    ) -> None:
        try:
            self._invoke_graph(
                run_id,
                config,
                initial_state,
                user_goal=user_goal,
                mode="resume",
                resume_action=decision,
            )
            self._finish_job(run_id)
        except Exception as exc:  # noqa: BLE001
            _log_runtime_error(run_id, config, str(exc))
            self._finish_job(run_id, error=str(exc))

    def start_run(
        self,
        goal: str,
        *,
        config_path: str | None = None,
        workspace: str | None = None,
    ) -> str:
        run_id = str(uuid4())
        thread = threading.Thread(
            target=self._run_new_job,
            args=(run_id, goal, config_path, workspace),
            daemon=True,
            name=f"linux-agent-web-run-{run_id}",
        )
        self._register_job(
            JobRecord(
                run_id=run_id,
                mode="new",
                user_goal=goal,
                started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                thread=thread,
            )
        )
        thread.start()
        return run_id

    def resume_run(
        self,
        run_id: str,
        decision: Literal["approve", "reject"],
        *,
        note: str | None = None,
        config_path: str | None = None,
        workspace: str | None = None,
    ) -> None:
        self._ensure_idle(run_id)
        base_config = _load_config_with_optional_workspace(
            self.default_config_path,
            config_path=config_path,
        )
        initial_state = load_run_state(run_id, base_config)
        resume_workspace = Path(initial_state["workspace_root"]).expanduser().resolve()
        requested_workspace = _resolve_workspace(workspace)
        if requested_workspace is not None and requested_workspace != resume_workspace:
            raise ValueError("workspace does not match the paused run's workspace_root")
        config = (
            base_config
            if resume_workspace == base_config.workspace_root
            else _override_workspace(base_config, resume_workspace)
        )
        initial_state["resume_action"] = decision
        initial_state["approval_response_note"] = note
        thread = threading.Thread(
            target=self._run_resume_job,
            args=(run_id, config, initial_state),
            kwargs={"user_goal": initial_state["user_goal"], "decision": decision},
            daemon=True,
            name=f"linux-agent-web-resume-{run_id}",
        )
        self._register_job(
            JobRecord(
                run_id=run_id,
                mode="resume",
                user_goal=initial_state["user_goal"],
                started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                thread=thread,
                resume_action=decision,
            )
        )
        thread.start()

    def _summarize_run(
        self,
        run_id: str,
        *,
        events: list[AuditEvent],
        state: AgentState | None,
        job: JobRecord | None,
        include_full_final_answer: bool = False,
    ) -> dict[str, Any]:
        run_start = next((event for event in events if event["event"] == EVENT_RUN_START), None)
        run_end = next((event for event in reversed(events) if event["event"] == EVENT_RUN_END), None)
        latest_event = events[-1] if events else None

        status = "unknown"
        if job is not None and job.running:
            status = "running"
        elif state is not None and state.get("pending_approval") is not None:
            status = "paused"
        elif run_end is not None:
            raw_status = str(run_end["data"].get("status", "completed"))
            status = "failed" if raw_status in {"runtime_error", "rollback_failed"} else "completed"
        elif job is not None and job.error is not None:
            status = "failed"

        user_goal = None
        workspace_root = None
        mode = None
        if run_start is not None:
            user_goal = run_start["data"].get("user_goal")
            workspace_root = run_start["data"].get("workspace_root")
            mode = run_start["data"].get("mode")
        if user_goal is None and job is not None:
            user_goal = job.user_goal

        final_answer = None
        if run_end is not None:
            final_answer = run_end["data"].get("final_answer")

        return {
            "run_id": run_id,
            "status": status,
            "mode": mode,
            "user_goal": user_goal,
            "workspace_root": workspace_root,
            "created_at": run_start["ts"] if run_start is not None else (job.started_at if job is not None else None),
            "updated_at": latest_event["ts"] if latest_event is not None else (job.finished_at or job.started_at if job is not None else None),
            "last_event": latest_event["event"] if latest_event is not None else None,
            "approval_pending": bool(state is not None and state.get("pending_approval") is not None),
            "final_answer_preview": _preview_text(final_answer),
            "final_answer": final_answer if include_full_final_answer else None,
            "command_count": run_end["data"].get("command_count") if run_end is not None else (state.get("command_count") if state is not None else None),
            "iteration_count": run_end["data"].get("iteration_count") if run_end is not None else (state.get("iteration_count") if state is not None else None),
            "error": job.error if job is not None else None,
        }

    def list_runs(self, *, config_path: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        config = _load_config_with_optional_workspace(
            self.default_config_path,
            config_path=config_path,
        )
        log_ids = {path.stem for path in config.log_dir.glob("*.jsonl")}
        state_ids = set()
        snapshot_dir = state_dir(config)
        if snapshot_dir.exists():
            state_ids = {path.stem for path in snapshot_dir.glob("*.json")}
        with self._lock:
            job_ids = set(self._jobs.keys())
        run_ids = log_ids | state_ids | job_ids

        summaries: list[dict[str, Any]] = []
        for run_id in run_ids:
            events = _read_events(config.log_dir / f"{run_id}.jsonl")
            state = _load_state_if_exists(config, run_id)
            summaries.append(
                self._summarize_run(
                    run_id,
                    events=events,
                    state=state,
                    job=self._get_job(run_id),
                )
            )

        summaries.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return summaries[:limit]

    def get_run_detail(self, run_id: str, *, config_path: str | None = None) -> dict[str, Any]:
        config = _load_config_with_optional_workspace(
            self.default_config_path,
            config_path=config_path,
        )
        events = _read_events(config.log_dir / f"{run_id}.jsonl")
        state = _load_state_if_exists(config, run_id)
        job = self._get_job(run_id)
        if not events and state is None and job is None:
            raise FileNotFoundError(f"run not found: {run_id}")

        effective_config = config
        if state is not None:
            workspace_root = Path(state["workspace_root"]).expanduser().resolve()
            if workspace_root != config.workspace_root:
                effective_config = _override_workspace(config, workspace_root)

        approval_view = None
        snapshot_path = _state_path_if_exists(effective_config, run_id)
        if state is not None and state.get("pending_approval") is not None:
            approval_view = build_approval_view(
                state,
                effective_config,
                state_path=snapshot_path,
            )

        approval_history = [
            event
            for event in events
            if event["event"] in {"approval_requested", "approval_presented", "approval_response"}
        ]

        messages = []
        if state is not None:
            messages = [_serialize_message(message) for message in state["messages"]]

        return {
            "summary": self._summarize_run(
                run_id,
                events=events,
                state=state,
                job=job,
                include_full_final_answer=True,
            ),
            "events": events,
            "messages": messages,
            "pending_approval": approval_view,
            "approval_history": approval_history,
            "active_job": None if job is None else job.snapshot(),
        }