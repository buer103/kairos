"""Confidence scorer middleware — evaluates output confidence with cited evidence."""

from __future__ import annotations

from typing import Any

from kairos.core.middleware import Middleware


class ConfidenceScorer(Middleware):
    """Evaluates output confidence and attaches an evidence summary.

    Hook: after_agent — scores the completed session.

    Scoring factors:
      - Step completion ratio (completed / total)
      - Step depth (more distinct tool types → higher confidence)
      - Tool result quality (presence of errors in tool results)
      - Evidence chain completeness
    """

    def after_agent(self, state: Any, runtime: dict[str, Any]) -> dict[str, Any] | None:
        if not state.case or not state.case.steps:
            return None

        steps = state.case.steps
        completed = [s for s in steps if s.result is not None]

        if not completed:
            state.case.confidence = 0.1
            return None

        # Factor 1: Completion ratio (0.3 weight)
        completion_ratio = len(completed) / max(len(steps), 1)
        completion_score = 0.3 + completion_ratio * 0.7  # 0.3–1.0

        # Factor 2: Tool diversity (0.3 weight)
        distinct_tools = len(set(s.tool for s in completed))
        tool_score = min(distinct_tools / max(len(completed), 1), 1.0) * 0.7 + 0.3

        # Factor 3: Error-free ratio (0.4 weight)
        error_free = [s for s in completed
                      if s.result and not isinstance(s.result, dict)
                      or not (isinstance(s.result, dict) and "error" in s.result)]
        error_ratio = len(error_free) / max(len(completed), 1)
        error_score = error_ratio

        # Weighted combination
        confidence = (
            completion_score * 0.3 + tool_score * 0.2 + error_score * 0.5
        )
        state.case.confidence = round(min(confidence, 0.99), 4)

        # Build evidence summary on case
        state.case.evidence_summary = [
            {
                "step": s.id,
                "tool": s.tool,
                "finding": (
                    str(s.result)[:200] if s.result else "no result"
                ),
                "duration_ms": s.duration_ms,
            }
            for s in completed
        ]

        return None
