"""Tests for the task / agent-run bookkeeping."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    import database
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "test.db")
    database.init()
    return database


def test_create_and_fetch_task(db):
    db.create_task("t1", "make something", "acme")
    task = db.get_task("t1")
    assert task["input"] == "make something"
    assert task["client"] == "acme"
    assert task["status"] == "pending"


def test_unknown_task_is_none(db):
    assert db.get_task("nope") is None


def test_update_task_status_and_result(db):
    db.create_task("t2", "x")
    db.update_task("t2", status="done", result='{"ok": true}')
    task = db.get_task("t2")
    assert task["status"] == "done"
    assert task["result"] == '{"ok": true}'


def test_list_tasks_respects_limit(db):
    for i in range(5):
        db.create_task(f"t{i}", f"task {i}")
    assert len(db.list_tasks(limit=3)) == 3


def test_agent_run_lifecycle_records_logs(db):
    db.create_task("t3", "x")
    run_id = db.start_agent_run("t3", "ceo")
    db.append_log(run_id, "[ceo] planning")
    db.append_log(run_id, "[ceo] dispatching")
    db.finish_agent_run(run_id, status="done")

    runs = db.get_agent_runs("t3")
    assert len(runs) == 1
    assert runs[0]["agent"] == "ceo"
    assert runs[0]["status"] == "done"
    assert "planning" in runs[0]["log"]
    assert "dispatching" in runs[0]["log"]


def test_agent_runs_are_scoped_to_their_task(db):
    db.create_task("a", "x")
    db.create_task("b", "y")
    db.start_agent_run("a", "ceo")
    assert len(db.get_agent_runs("a")) == 1
    assert db.get_agent_runs("b") == []
