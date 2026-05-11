"""Batch runner — run multiple agent queries in parallel with progress tracking.

Usage:
    runner = BatchRunner(agent, max_workers=4)
    results = runner.run([
        "Analyze auth.py",
        "Review test coverage",
        "Suggest performance improvements",
    ])
    for r in results:
        print(r.query, r.content)

Features:
  - Parallel execution via ThreadPoolExecutor
  - Progress tracking: completed/total, ETA
  - Per-query error isolation (one failure doesn't stop others)
  - Token usage aggregation
  - CSV/JSON export
  - Summary statistics
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("kairos.batch")


@dataclass
class BatchQuery:
    """A single query in a batch run."""

    id: int
    query: str


@dataclass
class BatchResult:
    """Result of a single batch query."""

    id: int
    query: str
    content: str = ""
    confidence: float | None = None
    error: str | None = None
    duration_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    tool_calls: int = 0

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.token_usage.get("total_tokens", 0)


@dataclass
class BatchSummary:
    """Aggregate statistics for a batch run."""

    total_queries: int = 0
    completed: int = 0
    failed: int = 0
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0
    results: list[BatchResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 1.0
        return self.completed / self.total_queries

    @property
    def avg_duration_ms(self) -> float:
        if self.completed == 0:
            return 0.0
        return self.total_duration_ms / self.completed

    @property
    def avg_tokens(self) -> float:
        if self.completed == 0:
            return 0.0
        return self.total_tokens / self.completed

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "completed": self.completed,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 3),
            "total_duration_ms": round(self.total_duration_ms, 1),
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "total_tokens": self.total_tokens,
            "avg_tokens": round(self.avg_tokens, 1),
            "total_tool_calls": self.total_tool_calls,
        }


class BatchRunner:
    """Execute multiple agent queries concurrently.

    Progress is printed via callback. Errors are isolated per query.

    Usage:
        runner = BatchRunner(agent, max_workers=4)
        summary = runner.run([
            "Analyze auth.py",
            "Review test coverage",
        ])
        print(summary.to_dict())
    """

    def __init__(
        self,
        agent: Any = None,
        max_workers: int = 4,
        timeout_per_query: float = 600.0,
        progress_callback: Any = None,
    ):
        """Initialize the batch runner.

        Args:
            agent: Kairos Agent or StatefulAgent instance.
            max_workers: Maximum concurrent queries.
            timeout_per_query: Max seconds per individual query.
            progress_callback: Optional callable(completed, total, current_query).
        """
        self._agent = agent
        self._max_workers = max(max_workers, 1)
        self._timeout = timeout_per_query
        self._progress_cb = progress_callback

    def run(
        self,
        queries: list[str],
        system_prompt: str = "",
    ) -> BatchSummary:
        """Run a batch of queries concurrently.

        Args:
            queries: List of query strings.
            system_prompt: Optional system prompt override.

        Returns:
            BatchSummary with results and statistics.
        """
        if not queries:
            return BatchSummary()

        start = time.time()
        results: list[BatchResult] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            futures = {}
            for i, query in enumerate(queries):
                future = executor.submit(
                    self._run_single, BatchQuery(id=i, query=query), system_prompt
                )
                futures[future] = i

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=self._timeout)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    idx = futures[future]
                    results.append(BatchResult(
                        id=idx, query=queries[idx],
                        error="Timeout",
                        duration_ms=self._timeout * 1000,
                    ))
                except Exception as e:
                    idx = futures[future]
                    results.append(BatchResult(
                        id=idx, query=queries[idx],
                        error=str(e),
                    ))

                completed += 1
                if self._progress_cb:
                    try:
                        self._progress_cb(completed, len(queries),
                                         queries[futures[future]])
                    except Exception:
                        pass

        # Sort by original order
        results.sort(key=lambda r: r.id)

        # Build summary
        summary = BatchSummary(
            total_queries=len(queries),
            results=results,
        )
        for r in results:
            if r.success:
                summary.completed += 1
                summary.total_tokens += r.total_tokens
                summary.total_tool_calls += r.tool_calls
            else:
                summary.failed += 1
        summary.total_duration_ms = (time.time() - start) * 1000

        return summary

    def run_with_agent_factory(
        self,
        queries: list[str],
        agent_factory: Any,
        system_prompt: str = "",
    ) -> BatchSummary:
        """Run batch with a fresh agent per query (no shared state).

        Args:
            queries: List of query strings.
            agent_factory: Callable that returns a new Agent instance.
            system_prompt: Optional system prompt.

        Returns:
            BatchSummary with results.
        """
        original_agent = self._agent
        try:
            results: list[BatchResult] = []
            start = time.time()

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers
            ) as executor:
                futures = {}
                for i, query in enumerate(queries):
                    agent = agent_factory()
                    future = executor.submit(
                        self._run_single_with_agent,
                        agent, BatchQuery(id=i, query=query), system_prompt,
                    )
                    futures[future] = i

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=self._timeout)
                        results.append(result)
                    except Exception as e:
                        idx = futures[future]
                        results.append(BatchResult(
                            id=idx, query=queries[idx], error=str(e),
                        ))

            results.sort(key=lambda r: r.id)

            summary = BatchSummary(
                total_queries=len(queries), results=results,
                total_duration_ms=(time.time() - start) * 1000,
            )
            for r in results:
                if r.success:
                    summary.completed += 1
                    summary.total_tokens += r.total_tokens
                    summary.total_tool_calls += r.tool_calls
                else:
                    summary.failed += 1

            return summary
        finally:
            self._agent = original_agent

    # ── Internal ─────────────────────────────────────────────────

    def _run_single(self, bq: BatchQuery, system_prompt: str) -> BatchResult:
        """Run a single query on the shared agent."""
        return self._run_single_with_agent(self._agent, bq, system_prompt)

    def _run_single_with_agent(
        self, agent: Any, bq: BatchQuery, system_prompt: str
    ) -> BatchResult:
        """Run a single query on a specific agent instance."""
        start = time.time()
        result = BatchResult(id=bq.id, query=bq.query)

        try:
            # Override system prompt if provided
            if system_prompt and hasattr(agent, "system_prompt"):
                original_sp = agent.system_prompt
                agent.system_prompt = system_prompt

            try:
                response = agent.run(bq.query)
            finally:
                if system_prompt and hasattr(agent, "system_prompt"):
                    agent.system_prompt = original_sp

            result.content = response.get("content", "")
            result.confidence = response.get("confidence")
            result.token_usage = response.get("usage", {})
            result.tool_calls = response.get("tool_calls_count",
                                   len(response.get("evidence", [])))
        except Exception as e:
            result.error = str(e)
            logger.error("Batch query %d failed: %s", bq.id, e)

        result.duration_ms = (time.time() - start) * 1000
        return result

    # ── Export ───────────────────────────────────────────────────

    @staticmethod
    def export_csv(summary: BatchSummary, path: str | Path) -> None:
        """Export batch results to CSV."""
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "query", "success", "content", "error",
                            "duration_ms", "tokens", "tool_calls"])
            for r in summary.results:
                writer.writerow([
                    r.id, r.query, r.success,
                    r.content[:200].replace("\n", " "),
                    r.error or "", r.duration_ms,
                    r.total_tokens, r.tool_calls,
                ])

    @staticmethod
    def export_jsonl(summary: BatchSummary, path: str | Path) -> None:
        """Export batch results to JSONL (one JSON per line)."""
        with open(path, "w") as f:
            for r in summary.results:
                f.write(json.dumps({
                    "id": r.id,
                    "query": r.query,
                    "success": r.success,
                    "content": r.content,
                    "confidence": r.confidence,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                    "token_usage": r.token_usage,
                    "tool_calls": r.tool_calls,
                }, ensure_ascii=False) + "\n")
