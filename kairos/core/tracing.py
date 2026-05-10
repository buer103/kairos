"""Trace ID — full-chain observability across parent and sub-agent calls.

Each agent.run() creates a **span**. Sub-agents create **child spans**
that form a tree. All trajectory events carry trace_id + span_id.

Flows:
  parent agent.run()
    span: root-abc123
    trajectory events all carry trace_id=trace-xyz, span_id=root-abc123
      └── sub-agent.run()
            span: sub-def456, parent_span_id=root-abc123
            events carry trace_id=trace-xyz, span_id=sub-def456
            ├── sub-sub-agent.run()
            │     span: sub-sub-ghi789, parent_span_id=sub-def456
            └── another sub-agent.run()
                  span: sub-jkl012, parent_span_id=sub-def456

This enables:
  - Query all spans for a trace: SELECT * WHERE trace_id = ?
  - Reconstruct the call tree: parent_span_id links
  - Performance analysis: span duration
  - Debug: full chain context when a sub-agent fails
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Context variable for implicit trace propagation across thread boundaries.
# Set by Agent.run() before entering the loop, read by delegate_task tools.
_current_trace: contextvars.ContextVar = contextvars.ContextVar(
    "kairos_trace", default=None
)


def get_current_trace() -> TraceContext | None:
    """Get the current trace context for this thread."""
    return _current_trace.get()


def set_current_trace(ctx: TraceContext | None) -> None:
    """Set the current trace context for this thread."""
    _current_trace.set(ctx)


# ============================================================================
# Trace Context
# ============================================================================


@dataclass
class TraceContext:
    """Carried through the entire agent call tree."""

    trace_id: str      # Root ID — same for all spans in a chain
    span_id: str       # This agent's span — unique per agent.run()
    parent_span_id: str | None = None  # Parent agent's span_id (None at root)
    depth: int = 0     # Nesting depth (0 = root)

    @staticmethod
    def new_root() -> TraceContext:
        """Create a root trace context (no parent)."""
        return TraceContext(
            trace_id=f"trace_{uuid.uuid4().hex[:12]}",
            span_id=f"root_{uuid.uuid4().hex[:8]}",
            parent_span_id=None,
            depth=0,
        )

    def child(self) -> TraceContext:
        """Create a child span context for a sub-agent."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=f"sub_{uuid.uuid4().hex[:8]}",
            parent_span_id=self.span_id,
            depth=self.depth + 1,
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "depth": self.depth,
        }


# ============================================================================
# Trace Event
# ============================================================================


@dataclass
class TraceEvent:
    """A single event within a span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    depth: int
    event_type: str       # "span_start" | "span_end" | "tool_start" | "tool_done" | ...
    timestamp: float
    data: dict[str, Any]
    iteration: int = 0

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "depth": self.depth,
            "type": self.event_type,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            **self.data,
        }


# ============================================================================
# Trace Recorder
# ============================================================================


class TraceRecorder:
    """Collects and persists trace events.

    Events are buffered in memory and flushed to JSONL on span end.
    Thread-safe for concurrent sub-agent recording.
    """

    def __init__(self, output_dir: Path | None = None):
        self._output_dir = output_dir or Path(".kairos/traces")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        ctx: TraceContext,
        event_type: str,
        data: dict[str, Any] | None = None,
        iteration: int = 0,
    ) -> TraceEvent:
        """Record a trace event. Returns the event for potential buffering."""
        event = TraceEvent(
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            depth=ctx.depth,
            event_type=event_type,
            timestamp=time.time(),
            data=data or {},
            iteration=iteration,
        )
        return event

    def flush_span(self, ctx: TraceContext, events: list[TraceEvent]) -> Path:
        """Write a span's events to JSONL. Returns the file path."""
        if not events:
            return None
        path = self._output_dir / f"{ctx.trace_id}_{ctx.span_id}.jsonl"
        with open(path, "w") as f:
            for e in events:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False, default=str) + "\n")
        return path

    def query_by_trace(self, trace_id: str) -> list[dict]:
        """Retrieve all events for a trace across all span files."""
        results = []
        for f in sorted(self._output_dir.glob(f"{trace_id}_*.jsonl")):
            try:
                for line in f.read_text().strip().split("\n"):
                    if line:
                        results.append(json.loads(line))
            except Exception:
                pass
        return sorted(results, key=lambda e: e.get("timestamp", 0))

    def get_span_tree(self, trace_id: str) -> dict:
        """Reconstruct the span tree for a trace."""
        events = self.query_by_trace(trace_id)
        spans: dict[str, dict] = {}
        for e in events:
            sid = e["span_id"]
            if sid not in spans:
                spans[sid] = {
                    "span_id": sid,
                    "parent_span_id": e.get("parent_span_id"),
                    "depth": e.get("depth", 0),
                    "events": [],
                    "children": [],
                    "start_time": None,
                    "end_time": None,
                    "error": None,
                }
            spans[sid]["events"].append(e)
            if e["type"] == "span_start":
                spans[sid]["start_time"] = e["timestamp"]
            if e["type"] == "span_end":
                spans[sid]["end_time"] = e["timestamp"]
                spans[sid]["duration_ms"] = round(
                    (e["timestamp"] - (spans[sid]["start_time"] or e["timestamp"])) * 1000, 1
                )
            if e["type"] == "error" and "error" in e:
                spans[sid]["error"] = e["error"]

        # Build tree
        root = None
        for sid, s in spans.items():
            pid = s["parent_span_id"]
            if pid and pid in spans:
                spans[pid]["children"].append(s)
            elif pid is None:
                root = s
        return root
