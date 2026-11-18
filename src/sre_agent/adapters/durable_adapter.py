"""
Durable Workflow Adapter (Temporal Pattern — Local SQLite Checkpointing + Retry).

Concept:
Just like Temporal.io's Event History:
  - Every step ("activity") in an SRE investigation is executed via `run_activity()`.
  - Once an activity succeeds, its output is persisted in SQLite keyed by `(workflow_id, step_name)`.
  - If the server crashes, laptop closes, or network drops on Step 3, restarting with the
    SAME `workflow_id` replays Steps 1 & 2 instantly from disk (ZERO re-execution, ZERO wasted tokens!)
    and resumes directly at Step 3!
  - Built-in automatic retry with backoff for flaky network/API calls.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class DurableWorkflowRunner:
    """
    Crash-resilient activity runner backed by SQLite event history.
    Drop-in compatible pattern for Temporal.io Activities & Workflows.
    """

    def __init__(self, db_path: str | Path = "sre_workflow_state.db") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_events (
                workflow_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                result_json TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (workflow_id, step_name)
            )
            """
        )
        self._conn.commit()

    def _load_checkpoint(self, workflow_id: str, step_name: str) -> tuple[bool, Any]:
        row = self._conn.execute(
            "SELECT result_json FROM workflow_events WHERE workflow_id = ? AND step_name = ?",
            (workflow_id, step_name),
        ).fetchone()
        if row is None:
            return False, None
        return True, json.loads(row[0])

    def _save_checkpoint(self, workflow_id: str, step_name: str, result: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO workflow_events (workflow_id, step_name, result_json, completed_at)
            VALUES (?, ?, ?, ?)
            """,
            (workflow_id, step_name, json.dumps(result), now),
        )
        self._conn.commit()

    def run_activity(
        self,
        workflow_id: str,
        step_name: str,
        fn: Callable[..., T],
        *args: Any,
        max_retries: int = 3,
        retry_delay_sec: float = 0.5,
        verbose: bool = True,
        **kwargs: Any,
    ) -> T:
        """
        Execute one durable activity step:
          1. If this step already succeeded in SQLite -> REPLAY immediately (skip execution!).
          2. Otherwise run `fn(*args, **kwargs)` with automatic retries.
          3. Persist result to SQLite before returning.
        """
        found, cached_result = self._load_checkpoint(workflow_id, step_name)
        if found:
            if verbose:
                print(f"[DURABLE REPLAY] Step '{step_name}' already completed — loaded from checkpoint!")
            return cached_result

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                if verbose:
                    print(f"[DURABLE EXEC] Running step '{step_name}' (attempt {attempt}/{max_retries})...")
                result = fn(*args, **kwargs)
                self._save_checkpoint(workflow_id, step_name, result)
                return result
            except Exception as exc:
                last_error = exc
                if verbose:
                    print(f"[DURABLE RETRY] Step '{step_name}' failed: {exc}")
                if attempt < max_retries:
                    time.sleep(retry_delay_sec * attempt)

        raise RuntimeError(
            f"Activity '{step_name}' in workflow '{workflow_id}' failed after {max_retries} attempts: {last_error}"
        )

    def list_completed_steps(self, workflow_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT step_name FROM workflow_events WHERE workflow_id = ? ORDER BY completed_at ASC",
            (workflow_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()
