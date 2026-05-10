"""Typed state extending agent messages with structured fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Step:
    """A single tool invocation in an evidence chain."""

    id: int
    tool: str
    args: dict[str, Any]
    result: dict[str, Any] | None = None
    started_at: datetime | None = None
    duration_ms: float = 0.0


@dataclass
class Case:
    """A complete agent session with evidence tracking."""

    id: str
    steps: list[Step] = field(default_factory=list)
    conclusion: str | None = None
    confidence: float | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def add_step(self, tool: str, args: dict[str, Any]) -> Step:
        step = Step(
            id=len(self.steps) + 1,
            tool=tool,
            args=args,
            started_at=datetime.now(),
        )
        self.steps.append(step)
        return step

    def complete_step(self, step: Step, result: dict[str, Any], duration_ms: float):
        step.result = result
        step.duration_ms = duration_ms


@dataclass
class ThreadState:
    """Typed state for an agent conversation thread."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    case: Case | None = None
    todos: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_context: Any | None = None  # TraceContext from kairos.core.tracing


def merge_artifacts(existing: list[str], new: list[str]) -> list[str]:
    """Reducer: deduplicate and preserve order when merging artifact lists."""
    if not existing:
        return new or []
    if not new:
        return existing
    return list(dict.fromkeys(existing + new))
