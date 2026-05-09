"""Evidence chain database — SQLite-backed persistent storage for reasoning traces.

Supports:
  - Individual step persistence (incremental)
  - Full-text search by tool name
  - Filter by error, thread_id, time range
  - Step count, error rate stats
  - Case lifecycle (save/load/delete/list)
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class EvidenceDB:
    """SQLite-backed evidence database.

    Schema:
        cases: (id TEXT PK, created_at REAL, conclusion TEXT, confidence REAL)
        steps: (step_id TEXT PK, case_id TEXT FK, step_type TEXT, tool TEXT,
                args TEXT, result TEXT, error TEXT, duration_ms REAL,
                confidence_before REAL, confidence_after REAL,
                parent_id TEXT, iteration INTEGER, thread_id TEXT,
                run_id TEXT, timestamp REAL, metadata TEXT)
    """

    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path or Path.home() / ".kairos" / "evidence.db")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                conclusion TEXT,
                confidence REAL
            );
            CREATE TABLE IF NOT EXISTS steps (
                step_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                step_type TEXT NOT NULL DEFAULT 'tool_call',
                tool TEXT NOT NULL DEFAULT '',
                args TEXT DEFAULT '{}',
                result TEXT,
                error TEXT,
                duration_ms REAL DEFAULT 0,
                confidence_before REAL,
                confidence_after REAL,
                parent_id TEXT,
                iteration INTEGER DEFAULT 0,
                thread_id TEXT DEFAULT '',
                run_id TEXT DEFAULT '',
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_steps_case ON steps(case_id);
            CREATE INDEX IF NOT EXISTS idx_steps_tool ON steps(tool);
            CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps(thread_id);
            CREATE INDEX IF NOT EXISTS idx_steps_timestamp ON steps(timestamp);
            CREATE INDEX IF NOT EXISTS idx_steps_error ON steps(error) WHERE error IS NOT NULL;
        """)
        self._conn.commit()

    # ── Case operations ────────────────────────────────────────

    def save(self, case: Any) -> Path:
        """Save a complete case with all its steps."""
        case_id = case.id if hasattr(case, "id") else str(time.time())
        confidence = getattr(case, "confidence", None)
        conclusion = getattr(case, "conclusion", None)

        self._conn.execute(
            "INSERT OR REPLACE INTO cases (id, created_at, conclusion, confidence) VALUES (?, ?, ?, ?)",
            (case_id, time.time(), conclusion, confidence),
        )

        if hasattr(case, "steps"):
            for s in case.steps:
                self._conn.execute(
                    """INSERT OR REPLACE INTO steps
                    (step_id, case_id, step_type, tool, args, result,
                     duration_ms, thread_id, timestamp)
                    VALUES (?, ?, 'tool_call', ?, ?, ?, ?, '', ?)""",
                    (
                        s.id, case_id, s.tool,
                        json.dumps(s.args, ensure_ascii=False),
                        json.dumps(s.result, ensure_ascii=False) if s.result else None,
                        getattr(s, "duration_ms", 0), time.time(),
                    ),
                )

        self._conn.commit()
        return Path(self._conn.execute("PRAGMA database_list").fetchone()[2])

    def load(self, case_id: str) -> Any | None:
        """Load a case from the database."""
        row = self._conn.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
        if not row:
            return None

        from kairos.core.state import Case, Step
        case = Case(id=row["id"])
        case.conclusion = row["conclusion"]
        case.confidence = row["confidence"]

        step_rows = self._conn.execute(
            "SELECT * FROM steps WHERE case_id = ? ORDER BY timestamp",
            (case_id,),
        ).fetchall()

        for sr in step_rows:
            step = Step(
                id=sr["step_id"],
                tool=sr["tool"],
                args=json.loads(sr["args"]) if sr["args"] else {},
                result=json.loads(sr["result"]) if sr["result"] else None,
                duration_ms=sr["duration_ms"],
            )
            case.steps.append(step)

        return case

    def list_cases(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent cases with step counts."""
        rows = self._conn.execute(
            """SELECT c.*, COUNT(s.step_id) as step_count
               FROM cases c LEFT JOIN steps s ON c.id = s.case_id
               GROUP BY c.id ORDER BY c.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "conclusion": r["conclusion"],
                "confidence": r["confidence"],
                "steps_count": r["step_count"],
            }
            for r in rows
        ]

    def delete_case(self, case_id: str) -> None:
        """Delete a case and all its steps (CASCADE)."""
        self._conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        self._conn.commit()

    # ── Step operations ────────────────────────────────────────

    def save_step(self, step: Any) -> None:
        """Save an individual evidence step (incremental persistence)."""
        step_id = step.step_id if hasattr(step, "step_id") else f"{step.case_id}-{int(time.time()*1000)}"
        case_id = step.case_id if hasattr(step, "case_id") else ""
        step_type = step.step_type.value if hasattr(step, "step_type") else "tool_call"
        tool = step.tool if hasattr(step, "tool") else ""
        args = json.dumps(step.args if hasattr(step, "args") else {}, ensure_ascii=False)
        result = json.dumps(step.result, ensure_ascii=False) if hasattr(step, "result") and step.result else None
        error = step.error if hasattr(step, "error") else None
        duration = step.duration_ms if hasattr(step, "duration_ms") else 0
        conf_before = step.confidence_before if hasattr(step, "confidence_before") else None
        conf_after = step.confidence_after if hasattr(step, "confidence_after") else None
        parent_id = step.parent_id if hasattr(step, "parent_id") else None
        iteration = step.iteration if hasattr(step, "iteration") else 0
        thread_id = step.thread_id if hasattr(step, "thread_id") else ""
        run_id = step.run_id if hasattr(step, "run_id") else ""
        ts = step.timestamp if hasattr(step, "timestamp") else time.time()
        metadata = json.dumps(step.metadata, ensure_ascii=False) if hasattr(step, "metadata") and step.metadata else "{}"

        self._conn.execute(
            """INSERT OR REPLACE INTO steps
            (step_id, case_id, step_type, tool, args, result, error,
             duration_ms, confidence_before, confidence_after, parent_id,
             iteration, thread_id, run_id, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (step_id, case_id, step_type, tool, args, result, error,
             duration, conf_before, conf_after, parent_id,
             iteration, thread_id, run_id, ts, metadata),
        )
        self._conn.commit()

    def get_steps(self, case_id: str) -> list[dict[str, Any]]:
        """Get all steps for a case, ordered by timestamp."""
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE case_id = ? ORDER BY timestamp",
            (case_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_step(self, case_id: str, step_id: str) -> dict[str, Any] | None:
        """Get a specific step."""
        row = self._conn.execute(
            "SELECT * FROM steps WHERE case_id = ? AND step_id = ?",
            (case_id, step_id),
        ).fetchone()
        return dict(row) if row else None

    def search_steps(
        self,
        tool: str | None = None,
        error: bool = False,
        thread_id: str | None = None,
        since: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search evidence steps with filters."""
        conditions = ["1=1"]
        params: list = []

        if tool:
            conditions.append("tool = ?")
            params.append(tool)
        if error:
            conditions.append("error IS NOT NULL")
        if thread_id:
            conditions.append("thread_id = ?")
            params.append(thread_id)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)

        query = f"SELECT * FROM steps WHERE {' AND '.join(conditions)} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Return evidence database statistics."""
        case_count = self._conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        step_count = self._conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        error_count = self._conn.execute(
            "SELECT COUNT(*) FROM steps WHERE error IS NOT NULL"
        ).fetchone()[0]
        avg_duration = self._conn.execute(
            "SELECT AVG(duration_ms) FROM steps WHERE duration_ms > 0"
        ).fetchone()[0] or 0

        return {
            "cases": case_count,
            "steps": step_count,
            "errors": error_count,
            "error_rate": round(error_count / step_count * 100, 2) if step_count else 0,
            "avg_duration_ms": round(avg_duration, 1),
        }

    def close(self) -> None:
        self._conn.close()
