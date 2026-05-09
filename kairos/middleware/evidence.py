"""Evidence tracker middleware — records every tool invocation as a structured step."""

from __future__ import annotations

import time
from typing import Any

from kairos.core.middleware import Middleware
from kairos.core.state import Case
from kairos.infra.evidence.tracker import EvidenceDB


class EvidenceTracker(Middleware):
    """Records every tool invocation as a Step in the evidence chain.

    Hook: wrap_tool_call
    Persistence: EvidenceDB (optional, enabled when db_path is set)
    """

    def __init__(self, db_path: str | None = None):
        self._db = EvidenceDB(db_path) if db_path else None

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        state = kwargs.get("state")
        start = time.time()
        result = handler(tool_name, args, **kwargs)
        elapsed = (time.time() - start) * 1000  # ms

        if state and state.case:
            step = state.case.add_step(tool_name, args)
            state.case.complete_step(step, result, elapsed)

        return result

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        """Persist the completed evidence chain."""
        if self._db and state and state.case and state.case.steps:
            self._db.save(state.case)
        return None

    def load_case(self, case_id: str) -> Case | None:
        """Load a previously persisted case."""
        if self._db:
            return self._db.load(case_id)
        return None

    def list_cases(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent cases from the database."""
        if self._db:
            return self._db.list_cases(limit)
        return []
