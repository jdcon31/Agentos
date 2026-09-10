"""
agents/base.py
BaseAgent — all agents inherit from this.
Provides logging, status reporting, and a standard run() interface.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Callable

import database


class BaseAgent(ABC):
    name: str = "agent"

    def __init__(self, task_id: str, log_callback: Callable[[str], None] | None = None):
        self.task_id = task_id
        self._run_id: int | None = None
        self._log_callback = log_callback  # called with each log line (for WebSocket streaming)

    def log(self, line: str):
        print(f"[{self.name}] {line}")
        if self._run_id is not None:
            database.append_log(self._run_id, f"[{self.name}] {line}")
        if self._log_callback:
            self._log_callback(f"[{self.name}] {line}")

    def run(self, **kwargs) -> dict:
        """
        Execute the agent. Returns a result dict.
        Handles DB bookkeeping around the actual _run() call.
        """
        self._run_id = database.start_agent_run(self.task_id, self.name)
        try:
            result = self._run(**kwargs)
            database.finish_agent_run(self._run_id, status="done")
            return result
        except Exception as e:
            self.log(f"ERROR: {e}")
            database.finish_agent_run(self._run_id, status="error")
            raise

    @abstractmethod
    def _run(self, **kwargs) -> dict:
        """Subclasses implement their logic here."""
        ...
