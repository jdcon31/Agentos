"""
database.py
SQLite schema and query helpers for AgentOS.

Tables (excerpt):
  tasks       — top-level user-submitted tasks
  agent_runs  — individual agent executions within a task, with streamed logs

The full system carries additional tables (approval inbox, asset library,
scheduled posts). They are not part of this excerpt.
"""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent / "agentos.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init():
    """Create tables if they don't exist."""
    with closing(_connect()) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                input       TEXT NOT NULL,
                client      TEXT NOT NULL DEFAULT 'default',
                output_dir  TEXT,
                result      TEXT,
                error       TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id     TEXT NOT NULL REFERENCES tasks(id),
                agent       TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                finished_at TEXT,
                status      TEXT NOT NULL DEFAULT 'running',
                log         TEXT NOT NULL DEFAULT ''
            );

        """)

        conn.commit()


# ── Tasks ─────────────────────────────────────────────────────────────────────

def create_task(task_id: str, input_text: str, client: str = "default") -> dict:
    with closing(_connect()) as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO tasks (id, created_at, status, input, client) VALUES (?,?,?,?,?)",
            (task_id, now, "pending", input_text, client),
        )
        conn.commit()
    return get_task(task_id)


def get_task(task_id: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None


def list_tasks(limit: int = 50) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def update_task(task_id: str, **kwargs):
    fields = ", ".join(f"{k}=?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    with closing(_connect()) as conn:
        conn.execute(f"UPDATE tasks SET {fields} WHERE id=?", values)
        conn.commit()


# ── Agent runs ────────────────────────────────────────────────────────────────

def start_agent_run(task_id: str, agent: str) -> int:
    with closing(_connect()) as conn:
        cur = conn.execute(
            "INSERT INTO agent_runs (task_id, agent, started_at, status) VALUES (?,?,?,?)",
            (task_id, agent, datetime.now(timezone.utc).isoformat(), "running"),
        )
        conn.commit()
        return cur.lastrowid


def append_log(run_id: int, line: str):
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE agent_runs SET log = log || ? WHERE id=?",
            (line + "\n", run_id),
        )
        conn.commit()


def finish_agent_run(run_id: int, status: str = "done"):
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE agent_runs SET finished_at=?, status=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), status, run_id),
        )
        conn.commit()


def get_agent_runs(task_id: str) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM agent_runs WHERE task_id=? ORDER BY id", (task_id,)
        ).fetchall()
        return [dict(r) for r in rows]
