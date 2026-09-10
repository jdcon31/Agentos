"""
main.py
AgentOS FastAPI backend.

Shape of the thing: a task is submitted over HTTP, runs on a background worker,
and streams its logs to the browser over a WebSocket keyed by task id. All
state lives in SQLite, so a server restart loses running work but never
finished work — the startup hook reconciles anything left mid-flight.

Agents are synchronous and CPU/IO-bound (model calls, image rendering), so they
run in a thread pool rather than on the event loop. The log callback they're
given is called from those worker threads, which is why the broadcaster hops
back onto the main loop with run_coroutine_threadsafe.

EXCERPT: the full application also serves client management, an asset library,
a style library, training-data endpoints and OAuth publishing routes. Those are
omitted. What is here is the task lifecycle — the spine everything else hangs
off.
"""
import asyncio
import json
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database
from agents.ceo import CEOAgent
from config import config

app = FastAPI(title="AgentOS", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Active WebSocket connections keyed by task_id.
_ws_connections: dict[str, list[WebSocket]] = {}
_main_loop: asyncio.AbstractEventLoop | None = None


# ── Lifecycle ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    """Capture the loop for cross-thread broadcasts and reconcile stale tasks."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()

    database.init()

    # A task still marked 'running' means the server died mid-flight. Nothing is
    # going to advance it, so mark it rather than leaving a permanent spinner.
    for task in database.list_tasks():
        if task["status"] == "running":
            database.update_task(task["id"], status="error",
                                 error="Server restarted while this task was running")


# ── Log broadcasting ─────────────────────────────────────────────────────────

def _broadcast(task_id: str, line: str):
    """Push a log line to every listener on this task.

    Called from agent worker threads, so it schedules onto the main loop rather
    than touching the sockets directly.
    """
    if _main_loop is None:
        return
    sockets = list(_ws_connections.get(task_id, []))
    if not sockets:
        return
    payload = json.dumps({"type": "log", "line": line})

    async def _send_all():
        dead = []
        for ws in sockets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        live = _ws_connections.get(task_id, [])
        for ws in dead:
            if ws in live:
                live.remove(ws)

    asyncio.run_coroutine_threadsafe(_send_all(), _main_loop)


def _run_task(task_id: str, task_input: str, client: str):
    """Worker-thread entry point. Owns the task's terminal status."""
    agent = CEOAgent(task_id, log_callback=lambda line: _broadcast(task_id, line))
    try:
        result = agent.run(task=task_input, client=client)
        database.update_task(task_id, status="done", result=json.dumps(result))
        _broadcast(task_id, "[done]")
    except Exception as exc:
        database.update_task(task_id, status="error", error=str(exc))
        _broadcast(task_id, f"[error] {exc}")


# ── Routes ───────────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    input: str
    client: str = "default"


@app.get("/health")
async def health():
    return {"status": "ok", "local_model": config.get("ollama_model")}


@app.post("/tasks", status_code=202)
async def create_task(req: TaskRequest):
    """Accept a task and hand it to a worker. Returns immediately.

    202 rather than 200: the work has been accepted, not completed. The client
    then opens /ws/{task_id} to watch it.
    """
    if not req.input.strip():
        raise HTTPException(status_code=422, detail="input must not be empty")

    task_id = uuid.uuid4().hex
    database.create_task(task_id, req.input, req.client)
    database.update_task(task_id, status="running")

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_task, task_id, req.input, req.client)

    return {"task_id": task_id}


@app.get("/tasks")
async def list_tasks(limit: int = 50):
    return {"tasks": database.list_tasks(limit=limit)}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = database.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="unknown task")
    task["runs"] = database.get_agent_runs(task_id)
    return task


@app.websocket("/ws/{task_id}")
async def task_logs(websocket: WebSocket, task_id: str):
    """Stream a task's logs.

    Replays what already happened before attaching to the live stream, so a
    client that connects late (or reconnects) sees the whole run rather than
    starting mid-sentence.
    """
    await websocket.accept()
    _ws_connections.setdefault(task_id, []).append(websocket)

    try:
        for run in database.get_agent_runs(task_id):
            for line in run["log"].splitlines():
                if line:
                    await websocket.send_text(json.dumps({"type": "log", "line": line}))

        while True:
            await websocket.receive_text()  # client keepalive; nothing to parse
    except WebSocketDisconnect:
        pass
    finally:
        live = _ws_connections.get(task_id, [])
        if websocket in live:
            live.remove(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.get("server_host"), port=config.get("server_port"))
