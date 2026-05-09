"""Evidence Tracker — records every decision, tool call, and outcome as structured evidence.

Kairos original middleware (not in Hermes or DeerFlow).
Builds an auditable, queryable, replayable evidence chain for every agent case.

Key capabilities:
  - 5 step types: tool_call, user_decision, error, fallback, checkpoint
  - Causal chain: parent-child step relationships
  - Incremental persistence: save after each step
  - Rich metadata: thread_id, run_id, iteration, timestamp, duration
  - Evidence query API: search by tool, error, time range
  - Confidence evolution tracking
  - Step hooks for downstream middleware
  - Observability metrics
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.evidence")


# ── Step Types ──────────────────────────────────────────────────

class StepType(str, Enum):
    """Classification of evidence steps."""
    TOOL_CALL = "tool_call"         # Agent invoked a tool
    USER_DECISION = "user_decision" # User made a choice (clarify)
    ERROR = "error"                  # Something went wrong
    FALLBACK = "fallback"            # Primary failed, used alternative
    CHECKPOINT = "checkpoint"        # Significant state snapshot


@dataclass
class EvidenceStep:
    """A single step in the evidence chain.

    Uniquely identified by (case_id, step_id). Steps form a DAG via parent_id.
    """
    case_id: str
    step_id: str
    step_type: StepType
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0
    confidence_before: float | None = None
    confidence_after: float | None = None
    parent_id: str | None = None
    iteration: int = 0
    thread_id: str = ""
    run_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "tool": self.tool,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "parent_id": self.parent_id,
            "iteration": self.iteration,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceStep:
        return cls(
            case_id=data["case_id"],
            step_id=data["step_id"],
            step_type=StepType(data.get("step_type", "tool_call")),
            tool=data.get("tool", ""),
            args=data.get("args", {}),
            result=data.get("result"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms", 0),
            confidence_before=data.get("confidence_before"),
            confidence_after=data.get("confidence_after"),
            parent_id=data.get("parent_id"),
            iteration=data.get("iteration", 0),
            thread_id=data.get("thread_id", ""),
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", 0),
            metadata=data.get("metadata", {}),
        )


class StepHook:
    """Hook called after each evidence step is recorded."""
    def __call__(self, step: EvidenceStep, runtime: dict) -> None:
        pass


# ── Evidence Tracker ────────────────────────────────────────────

class EvidenceTracker(Middleware):
    """Records every tool invocation and decision as auditable evidence.

    Hook: wrap_tool_call — intercepts every tool execution.
          after_agent — persists the complete case.

    Usage:
        tracker = EvidenceTracker(db_path="~/.kairos/evidence.db")
        tracker.on_step(lambda step, rt: print(f"Step: {step.tool}"))
    """

    def __init__(self, db_path: str | None = None, max_steps_per_case: int = 200):
        self._db_path = db_path
        self._db = None  # Lazy init
        self._max_steps = max_steps_per_case
        self._iteration = 0
        self._hooks: list[StepHook] = []
        self._current_case: Any = None
        self._step_counter: dict[str, int] = {}

    # ── Hook registration ──────────────────────────────────────

    def on_step(self, hook: StepHook) -> None:
        """Register a hook called after each evidence step."""
        self._hooks.append(hook)

    # ── wrap_tool_call — primary interception ──────────────────

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        state = kwargs.get("state")
        runtime = kwargs.get("runtime", {})

        self._iteration = runtime.get("turn", self._iteration + 1)

        case = self._get_or_create_case(state, runtime)
        thread_id = runtime.get("thread_id", "")
        run_id = runtime.get("run_id", "")

        # Capture pre-execution confidence
        confidence_before = getattr(case, "confidence", None) if case else None

        # Track parent step (last step in this case)
        parent_id = None
        if case and hasattr(case, "steps") and case.steps:
            parent_id = case.steps[-1].id if hasattr(case.steps[-1], "id") else None

        # Execute the tool
        start = time.time()
        error = None
        try:
            result = handler(tool_name, args, **kwargs)
        except Exception as e:
            error = str(e)
            result = {"error": error}
            logger.warning("Tool '%s' failed: %s", tool_name, e)

        elapsed = (time.time() - start) * 1000

        # Capture post-execution confidence
        confidence_after = getattr(case, "confidence", None) if case else None

        # Build evidence step
        step_type = StepType.ERROR if error else StepType.TOOL_CALL
        case_id = case.id if case else "unknown"
        step_count = self._step_counter.get(case_id, 0) + 1
        self._step_counter[case_id] = step_count

        evidence = EvidenceStep(
            case_id=case_id,
            step_id=f"{case_id}-{step_count:04d}",
            step_type=step_type,
            tool=tool_name,
            args=args,
            result=result if not error else None,
            error=error,
            duration_ms=round(elapsed, 2),
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            parent_id=parent_id,
            iteration=self._iteration,
            thread_id=thread_id,
            run_id=run_id,
        )

        # Add to in-memory case
        if case and hasattr(case, "add_step"):
            if len(getattr(case, "steps", [])) < self._max_steps:
                case.add_step(tool_name, args)
                if hasattr(case, "complete_step"):
                    case.complete_step(case.steps[-1], result, elapsed)

        # Fire hooks
        for hook in self._hooks:
            try:
                hook(evidence, runtime)
            except Exception:
                pass

        # Incremental persistence
        if self._db:
            try:
                self._get_db().save_step(evidence)
            except Exception as e:
                logger.debug("Failed to persist evidence step: %s", e)

        return result

    # ── after_agent — finalize ─────────────────────────────────

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Persist the complete evidence chain on session end."""
        if not self._db:
            return None

        case = self._current_case
        if not case and state and hasattr(state, "case"):
            case = state.case

        if case and hasattr(case, "steps") and case.steps:
            try:
                self._get_db().save(case)
                logger.debug(
                    "Persisted evidence chain: case=%s steps=%d",
                    case.id, len(case.steps),
                )
            except Exception as e:
                logger.error("Failed to persist evidence chain: %s", e)

        self._iteration = 0
        self._current_case = None
        self._step_counter.clear()
        return None

    # ── Query API ──────────────────────────────────────────────

    def get_steps(self, case_id: str) -> list[EvidenceStep]:
        """Get all steps for a case."""
        if self._db:
            return self._get_db().get_steps(case_id)
        return []

    def get_step(self, case_id: str, step_id: str) -> EvidenceStep | None:
        """Get a specific step."""
        if self._db:
            return self._get_db().get_step(case_id, step_id)
        return None

    def search_steps(
        self,
        tool: str | None = None,
        error: bool = False,
        thread_id: str | None = None,
        since: float | None = None,
        limit: int = 50,
    ) -> list[EvidenceStep]:
        """Search evidence steps with filters."""
        if self._db:
            return self._get_db().search_steps(
                tool=tool, error=error, thread_id=thread_id,
                since=since, limit=limit,
            )
        return []

    def list_cases(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent cases."""
        if self._db:
            return self._get_db().list_cases(limit)
        return []

    def load_case(self, case_id: str):
        """Load a complete case from the database."""
        if self._db:
            return self._get_db().load(case_id)
        return None

    def delete_case(self, case_id: str) -> bool:
        """Delete a case and all its steps."""
        if self._db:
            self._get_db().delete_case(case_id)
            return True
        return False

    # ── Observability ──────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return evidence tracking statistics."""
        if self._db:
            return self._get_db().stats()
        return {"cases": 0, "steps": 0, "errors": 0}

    # ── Internal ──────────────────────────────────────────────

    def _get_or_create_case(self, state: Any, runtime: dict) -> Any:
        """Get the current case or create a new one."""
        if self._current_case:
            return self._current_case

        if state and hasattr(state, "case") and state.case:
            self._current_case = state.case
            return state.case

        # Create a new case
        from kairos.core.state import Case
        case_id = runtime.get("thread_id") or runtime.get("session_id") or "unknown"
        self._current_case = Case(id=case_id)
        if state:
            state.case = self._current_case
        return self._current_case

    def _get_db(self):
        """Lazy-init the evidence database."""
        if self._db is None and self._db_path:
            from kairos.infra.evidence.tracker import EvidenceDB
            self._db = EvidenceDB(self._db_path)
        return self._db

    def __repr__(self) -> str:
        return f"EvidenceTracker(db={self._db_path}, max_steps={self._max_steps})"
