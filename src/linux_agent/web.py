"""FastAPI-powered web console for the Linux Agent."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from linux_agent.web_runtime import WebRuntime


_STATIC_DIR = Path(__file__).resolve().parent / "web_static"


class StartRunRequest(BaseModel):
    goal: str = Field(min_length=1)
    workspace: str | None = None
    config_path: str | None = None


class ApprovalDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = None
    workspace: str | None = None
    config_path: str | None = None


def create_app(default_config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Linux Agent Web Console")
    app.state.runtime = WebRuntime(default_config_path=default_config_path)
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, str | None]:
        return {
            "status": "ok",
            "default_config_path": default_config_path,
        }

    @app.get("/api/runs")
    def list_runs() -> dict[str, object]:
        runtime: WebRuntime = app.state.runtime
        try:
            runs = runtime.list_runs()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"runs": runs}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, object]:
        runtime: WebRuntime = app.state.runtime
        try:
            return runtime.get_run_detail(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs")
    def start_run(request: StartRunRequest) -> dict[str, str]:
        runtime: WebRuntime = app.state.runtime
        try:
            run_id = runtime.start_run(
                request.goal,
                config_path=request.config_path,
                workspace=request.workspace,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": run_id, "status": "started"}

    @app.post("/api/runs/{run_id}/approval")
    def approval_action(run_id: str, request: ApprovalDecisionRequest) -> dict[str, str]:
        runtime: WebRuntime = app.state.runtime
        try:
            runtime.resume_run(
                run_id,
                request.decision,
                note=request.note,
                config_path=request.config_path,
                workspace=request.workspace,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": run_id, "status": "started", "decision": request.decision}

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linux-agent-web",
        description="Web console for the Linux Agent.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the Linux Agent config file used by the web console.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    uvicorn.run(
        create_app(default_config_path=args.config),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())