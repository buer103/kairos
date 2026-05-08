"""Confidence scorer middleware — evaluates output confidence after agent finishes."""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware


class ConfidenceScorer(Middleware):
    """Evaluates output confidence and attaches an evidence summary."""

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        if not state.case or not state.case.steps:
            return None

        completed = [s for s in state.case.steps if s.result is not None]
        if not completed:
            return None

        # Heuristic: more completed steps = higher confidence
        ratio = len(completed) / max(len(state.case.steps), 1)
        state.case.confidence = min(0.5 + ratio * 0.5, 0.99)
        return None
