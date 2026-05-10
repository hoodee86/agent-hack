from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

from linux_agent.audit import AuditLogger, EVENT_APPROVAL_PRESENTED, EVENT_RUN_END, EVENT_RUN_START
from linux_agent.run_store import save_run_state
from linux_agent.state import AgentState
from linux_agent.web import create_app


def _write_config(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "workspace_root: ./workspace",
                "log_dir: ./logs",
                "backup_dir: ./.linux-agent/backups",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _paused_state(run_id: str, workspace_root: str) -> AgentState:
    return AgentState(
        run_id=run_id,
        user_goal="Update README with safer approval wording",
        workspace_root=workspace_root,
        started_at="2026-05-10T00:00:00+00:00",
        messages=[],
        plan=["Inspect README", "Write patch"],
        command_count=1,
        plan_version=1,
        plan_revision_count=0,
        plan_steps=[
            {"id": "step_1", "title": "Inspect README", "status": "completed", "rationale": None, "evidence_refs": [1]},
            {"id": "step_2", "title": "Write patch", "status": "in_progress", "rationale": None, "evidence_refs": [2]},
        ],
        last_reflection=None,
        recovery_state=None,
        recovery_attempt_total=0,
        budget_status={
            "iteration_count": 2,
            "command_count": 1,
            "elapsed_seconds": 12,
            "warning_triggered": False,
        },
        budget_stop_reason=None,
        current_step="Write patch",
        proposed_tool_call=None,
        observations=[],
        risk_decision="needs_approval",
        pending_approval={
            "id": "approval_1",
            "tool": "apply_patch",
            "args": {"patch": "*** Begin Patch\n..."},
            "reason": "Patch application requires explicit approval before execution.",
            "impact_summary": "This patch would update README.md.",
            "diff_preview": "- old\n+ new",
            "backup_plan": "Backups will be written before overwrite.",
            "affected_files": ["README.md"],
            "risk_level": "high",
            "suggested_verification_command": "uv run pytest",
            "rollback_command": "--rollback-run paused-run",
        },
        resume_action=None,
        approval_response_note=None,
        pending_verification=None,
        last_write=None,
        last_verification=None,
        last_rollback=None,
        iteration_count=2,
        consecutive_failures=0,
        final_answer=None,
    )


def test_runs_endpoint_lists_completed_and_paused_runs(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    workspace_root = str((tmp_path / "workspace").resolve())
    logs_dir = tmp_path / "logs"

    with AuditLogger("completed-run", logs_dir) as audit:
        audit.log(EVENT_RUN_START, {"user_goal": "List files", "workspace_root": workspace_root, "mode": "new", "resume_action": None, "config": {}})
        audit.log(EVENT_RUN_END, {"status": "completed", "final_answer": "Done.", "iteration_count": 1, "command_count": 0})

    with AuditLogger("paused-run", logs_dir) as audit:
        audit.log(EVENT_RUN_START, {"user_goal": "Update README", "workspace_root": workspace_root, "mode": "new", "resume_action": None, "config": {}})
        audit.log(EVENT_APPROVAL_PRESENTED, {"impact_summary": "This patch would update README.md."})

    app = create_app(default_config_path=str(config_path))
    from linux_agent.config import load_config
    save_run_state(_paused_state("paused-run", workspace_root), load_config(str(config_path)))

    client = TestClient(app)
    response = client.get("/api/runs")

    assert response.status_code == 200
    runs = response.json()["runs"]
    by_id = {run["run_id"]: run for run in runs}
    assert by_id["completed-run"]["status"] == "completed"
    assert by_id["paused-run"]["status"] == "paused"
    assert by_id["paused-run"]["approval_pending"] is True


def test_run_detail_returns_pending_approval_and_events(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    workspace_root = str((tmp_path / "workspace").resolve())
    logs_dir = tmp_path / "logs"
    with AuditLogger("paused-run", logs_dir) as audit:
        audit.log(EVENT_RUN_START, {"user_goal": "Update README", "workspace_root": workspace_root, "mode": "new", "resume_action": None, "config": {}})
        audit.log(EVENT_APPROVAL_PRESENTED, {"impact_summary": "This patch would update README.md."})

    from linux_agent.config import load_config
    save_run_state(_paused_state("paused-run", workspace_root), load_config(str(config_path)))

    client = TestClient(create_app(default_config_path=str(config_path)))
    response = client.get("/api/runs/paused-run")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["status"] == "paused"
    assert payload["pending_approval"]["tool"] == "apply_patch"
    assert payload["pending_approval"]["affected_files"] == ["README.md"]
    assert len(payload["events"]) == 2


def test_run_detail_returns_full_final_answer(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    workspace_root = str((tmp_path / "workspace").resolve())
    logs_dir = tmp_path / "logs"
    final_answer = "Line 1\nLine 2\nLine 3 with full content"

    with AuditLogger("completed-run", logs_dir) as audit:
        audit.log(EVENT_RUN_START, {"user_goal": "Summarize notes", "workspace_root": workspace_root, "mode": "new", "resume_action": None, "config": {}})
        audit.log(EVENT_RUN_END, {"status": "completed", "final_answer": final_answer, "iteration_count": 1, "command_count": 0})

    client = TestClient(create_app(default_config_path=str(config_path)))
    response = client.get("/api/runs/completed-run")

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["final_answer"] == final_answer
    assert summary["final_answer_preview"] is not None


def test_start_and_approval_endpoints_delegate_to_runtime(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    app = create_app(default_config_path=str(config_path))
    calls: dict[str, list[dict[str, object]]] = {"start": [], "resume": []}

    def fake_start_run(goal: str, *, config_path: str | None = None, workspace: str | None = None) -> str:
        calls["start"].append({"goal": goal, "config_path": config_path, "workspace": workspace})
        return "run-123"

    def fake_resume_run(
        run_id: str,
        decision: str,
        *,
        note: str | None = None,
        config_path: str | None = None,
        workspace: str | None = None,
    ) -> None:
        calls["resume"].append(
            {
                "run_id": run_id,
                "decision": decision,
                "note": note,
                "config_path": config_path,
                "workspace": workspace,
            }
        )

    app.state.runtime.start_run = fake_start_run
    app.state.runtime.resume_run = fake_resume_run

    client = TestClient(app)
    start_response = client.post(
        "/api/runs",
        json={"goal": "Inspect current failures", "workspace": str(tmp_path / "workspace")},
    )
    approval_response = client.post(
        "/api/runs/run-123/approval",
        json={"decision": "approve", "note": "Looks safe"},
    )

    assert start_response.status_code == 200
    assert start_response.json() == {"run_id": "run-123", "status": "started"}
    assert approval_response.status_code == 200
    assert approval_response.json() == {"run_id": "run-123", "status": "started", "decision": "approve"}
    assert calls["start"] == [{"goal": "Inspect current failures", "config_path": None, "workspace": str(tmp_path / "workspace")}]
    assert calls["resume"] == [{"run_id": "run-123", "decision": "approve", "note": "Looks safe", "config_path": None, "workspace": None}]