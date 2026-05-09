"""Cron scheduler — register, persist, and fire timer-based agent tasks.

Features:
  - Cron-style scheduling (minute, hour, day, month, weekday)
  - SQLite persistence across restarts
  - Background daemon thread with 60s resolution
  - Job lifecycle: pending → running → done / error / paused
  - Context variables injected per execution
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("kairos.cron")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class CronSchedule:
    """Cron-style schedule definition.  All fields are lists of valid values.
    Empty list means "every" (wildcard).

    Example — every Monday at 9:00:
        CronSchedule(hour=[9], minute=[0], weekday=[1])
    Example — every 30 minutes:
        CronSchedule(minute=[0, 30])
    """

    minute: list[int] = field(default_factory=list)   # 0-59
    hour: list[int] = field(default_factory=list)      # 0-23
    day: list[int] = field(default_factory=list)       # 1-31
    month: list[int] = field(default_factory=list)     # 1-12
    weekday: list[int] = field(default_factory=list)   # 0=Mon .. 6=Sun

    @classmethod
    def every(cls, minutes: int = 0, hours: int = 0) -> CronSchedule:
        """Convenience: run every N minutes/hours."""
        if minutes:
            return cls(minute=[m for m in range(0, 60, minutes)])
        if hours:
            return cls(hour=[h for h in range(0, 24, hours)])
        return cls()

    @classmethod
    def daily_at(cls, hour: int = 9, minute: int = 0) -> CronSchedule:
        return cls(hour=[hour], minute=[minute])

    @classmethod
    def weekly_on(cls, weekday: int, hour: int = 9, minute: int = 0) -> CronSchedule:
        """weekday: 0=Mon .. 6=Sun"""
        return cls(hour=[hour], minute=[minute], weekday=[weekday])

    def matches(self, dt: datetime) -> bool:
        """Check if this schedule fires at the given datetime."""

        def _matches(values: list[int], actual: int) -> bool:
            return len(values) == 0 or actual in values

        return (
            _matches(self.month, dt.month)
            and _matches(self.day, dt.day)
            and _matches(self.weekday, dt.weekday())
            and _matches(self.hour, dt.hour)
            and _matches(self.minute, dt.minute)
        )

    def next_fire(self, after: datetime | None = None) -> datetime:
        """Return the next datetime this schedule fires after `after` (or now)."""
        after = after or datetime.now(tz=after.tzinfo) if after and after.tzinfo else after
        if after is None:
            after = datetime.now()
        if after.tzinfo is None:
            after = after.replace(tzinfo=timezone.utc)

        # Walk forward minute by minute until we find a match (max 366 days)
        candidate = after.replace(second=0, microsecond=0) + __import__("datetime").timedelta(minutes=1)
        deadline = candidate + __import__("datetime").timedelta(days=366)
        while candidate < deadline:
            if self.matches(candidate):
                return candidate
            candidate += __import__("datetime").timedelta(minutes=1)
        raise ValueError("No next fire time found within 366 days")


@dataclass
class Job:
    """A cron job that invokes a callback when its schedule fires.

    Attributes:
        id: Unique job identifier (auto-generated if empty).
        name: Human-readable name.
        schedule: Cron-style schedule definition.
        callback_data: JSON-serializable payload passed to the handler.
        status: Current lifecycle status.
        run_count: How many times this job has executed.
        last_run: ISO timestamp of last execution.
        next_run: ISO timestamp of next scheduled execution.
        last_error: Error message from last failed run, if any.
        repeat: Max repeat count (0 = unlimited).
        created_at: ISO timestamp of creation.
        context: Arbitrary key-value context injected per run.
    """

    id: str = ""
    name: str = ""
    schedule: CronSchedule = field(default_factory=CronSchedule)
    callback_data: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    run_count: int = 0
    last_run: str = ""
    next_run: str = ""
    last_error: str = ""
    repeat: int = 0  # 0 = unlimited
    created_at: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"job_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "schedule": {
                "minute": self.schedule.minute,
                "hour": self.schedule.hour,
                "day": self.schedule.day,
                "month": self.schedule.month,
                "weekday": self.schedule.weekday,
            },
            "callback_data": self.callback_data,
            "status": self.status.value,
            "run_count": self.run_count,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "last_error": self.last_error,
            "repeat": self.repeat,
            "created_at": self.created_at,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Job:
        sched = CronSchedule(**d.get("schedule", {}))
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            schedule=sched,
            callback_data=d.get("callback_data", {}),
            status=JobStatus(d.get("status", "pending")),
            run_count=d.get("run_count", 0),
            last_run=d.get("last_run", ""),
            next_run=d.get("next_run", ""),
            last_error=d.get("last_error", ""),
            repeat=d.get("repeat", 0),
            created_at=d.get("created_at", ""),
            context=d.get("context", {}),
        )


class CronScheduler:
    """Background cron scheduler with SQLite persistence.

    Usage::

        scheduler = CronScheduler(db_path="~/.kairos/cron.db")
        scheduler.register(Job(
            name="daily-report",
            schedule=CronSchedule.daily_at(9, 0),
            callback_data={"agent": "reporter", "task": "daily_summary"},
        ))
        scheduler.start()  # background daemon thread

        # Define your handler
        @scheduler.on_fire
        def handle_job(job: Job):
            agent = Agent.load(job.callback_data["agent"])
            agent.run(job.callback_data["task"])
    """

    # Default polling interval (seconds)
    TICK_INTERVAL = 30

    def __init__(self, db_path: str = "~/.kairos/cron.db"):
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._handlers: list[Callable[[Job], None]] = []
        self._init_db()

    # ── Public API ──────────────────────────────────────────────

    def register(self, job: Job) -> Job:
        """Register a new job. Returns the job with its assigned ID."""
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO jobs (id, name, schedule_json, callback_data_json,
                   status, run_count, last_run, next_run, last_error, repeat,
                   created_at, context_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job.id,
                    job.name,
                    json.dumps(job.to_dict()["schedule"]),
                    json.dumps(job.callback_data),
                    job.status.value,
                    job.run_count,
                    job.last_run,
                    job.next_run or "",
                    job.last_error,
                    job.repeat,
                    job.created_at,
                    json.dumps(job.context),
                ),
            )
            conn.commit()
            conn.close()
            logger.info("Registered cron job %s (%s)", job.id, job.name)
            return job

    def get(self, job_id: str) -> Job | None:
        """Retrieve a job by ID."""
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            conn.close()
            if not row:
                return None
            return self._row_to_job(row)

    def list(self, status: JobStatus | None = None) -> list[Job]:
        """List all jobs, optionally filtered by status."""
        with self._lock:
            conn = self._connect()
            if status:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC"
                ).fetchall()
            conn.close()
            return [self._row_to_job(r) for r in rows]

    def update(self, job: Job) -> Job:
        """Update an existing job."""
        with self._lock:
            conn = self._connect()
            conn.execute(
                """UPDATE jobs SET name=?, schedule_json=?, callback_data_json=?,
                   status=?, run_count=?, last_run=?, next_run=?, last_error=?,
                   repeat=?, context_json=?
                   WHERE id=?""",
                (
                    job.name,
                    json.dumps(job.to_dict()["schedule"]),
                    json.dumps(job.callback_data),
                    job.status.value,
                    job.run_count,
                    job.last_run,
                    job.next_run,
                    job.last_error,
                    job.repeat,
                    json.dumps(job.context),
                    job.id,
                ),
            )
            conn.commit()
            conn.close()
            return job

    def pause(self, job_id: str) -> Job | None:
        """Pause a job."""
        job = self.get(job_id)
        if not job:
            return None
        job.status = JobStatus.PAUSED
        return self.update(job)

    def resume(self, job_id: str) -> Job | None:
        """Resume a paused job."""
        job = self.get(job_id)
        if not job:
            return None
        job.status = JobStatus.PENDING
        return self.update(job)

    def cancel(self, job_id: str) -> Job | None:
        """Cancel a job permanently."""
        job = self.get(job_id)
        if not job:
            return None
        job.status = JobStatus.CANCELLED
        return self.update(job)

    def remove(self, job_id: str) -> bool:
        """Permanently delete a job."""
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            conn.commit()
            conn.close()
            return True

    def on_fire(self, handler: Callable[[Job], None]) -> Callable[[Job], None]:
        """Register a handler that is called when a job fires."""
        self._handlers.append(handler)
        return handler

    def start(self) -> None:
        """Start the background scheduler daemon."""
        if self._thread and self._thread.is_alive():
            logger.warning("Cron scheduler already running.")
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="kairos-cron")
        self._thread.start()
        logger.info("Cron scheduler started (tick=%ds)", self.TICK_INTERVAL)

    def stop(self) -> None:
        """Stop the background scheduler gracefully."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Cron scheduler stopped.")

    def tick(self) -> list[Job]:
        """Manually fire one tick — checks all pending jobs and fires those due.
        Returns the list of jobs that were triggered.
        """
        now = datetime.now(timezone.utc)
        fired: list[Job] = []

        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('pending', 'error')"
            ).fetchall()
            conn.close()

            for row in rows:
                job = self._row_to_job(row)

                # Check repeat limit
                if job.repeat > 0 and job.run_count >= job.repeat:
                    job.status = JobStatus.DONE
                    self.update(job)
                    continue

                # Check schedule
                if not job.schedule.matches(now):
                    continue

                job.status = JobStatus.RUNNING
                job.run_count += 1
                job.last_run = now.isoformat()
                try:
                    job.next_run = job.schedule.next_fire(now).isoformat()
                except ValueError:
                    job.next_run = "exhausted"

                self.update(job)
                fired.append(job)

        # Execute handlers outside the lock
        for job in fired:
            if self._handlers:
                any_success = False
                for handler in self._handlers:
                    try:
                        handler(job)
                        any_success = True
                    except Exception as e:
                        job.last_error = str(e)
                        logger.error("Job %s failed: %s", job.id, e)
                job.status = JobStatus.DONE if any_success else JobStatus.ERROR
            else:
                # No handlers registered — mark as done immediately
                job.status = JobStatus.DONE
            self.update(job)

        return fired

    # ── Internal ────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                schedule_json TEXT NOT NULL DEFAULT '{}',
                callback_data_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                run_count INTEGER NOT NULL DEFAULT 0,
                last_run TEXT NOT NULL DEFAULT '',
                next_run TEXT NOT NULL DEFAULT '',
                last_error TEXT NOT NULL DEFAULT '',
                repeat INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT '',
                context_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.commit()
        conn.close()

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        d = dict(row)
        return Job.from_dict({
            "id": d["id"],
            "name": d["name"],
            "schedule": json.loads(d["schedule_json"]),
            "callback_data": json.loads(d["callback_data_json"]),
            "status": d["status"],
            "run_count": d["run_count"],
            "last_run": d["last_run"],
            "next_run": d["next_run"],
            "last_error": d["last_error"],
            "repeat": d["repeat"],
            "created_at": d["created_at"],
            "context": json.loads(d["context_json"]),
        })

    def _loop(self) -> None:
        """Background daemon loop."""
        while self._running.is_set():
            try:
                self.tick()
            except Exception as e:
                logger.error("Cron tick error: %s", e)
            self._running.wait(self.TICK_INTERVAL)
