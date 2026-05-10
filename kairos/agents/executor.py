"""Sub-Agent executor — manages sub-agent lifecycle, evidence inheritance, and parallel execution."""

from __future__ import annotations

import concurrent.futures
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from kairos.agents.types import BUILTIN_TYPES, SubAgentType
from kairos.core.state import Case
from kairos.providers.base import ModelConfig, ModelProvider

# Thread pool for parallel sub-agent execution
_executor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)


@dataclass
class TaskSpec:
    """Specification for a single sub-agent task in a batch."""

    prompt: str
    sub_type: SubAgentType
    description: str = ""
    parent_case: Case | None = None
    timeout: float = 300.0  # seconds
    toolsets: list[str] | None = None  # e.g. ["terminal", "file"]
    role: str = "leaf"  # "leaf" | "orchestrator"
    parent_trace: Any | None = None  # TraceContext from parent agent


@dataclass
class SubAgentResult:
    """Result of a sub-agent execution."""

    subagent_type: str
    description: str
    status: str  # "success", "error", "timeout", "cancelled"
    content: str | None
    confidence: float | None
    evidence: list[dict[str, Any]]
    sub_case_id: str
    error: str | None = None
    child_results: list[SubAgentResult] = field(default_factory=list)
    duration_ms: float = 0.0


class SubAgentExecutor:
    """Manages creation and execution of typed sub-agents.

    Handles:
      - Model inheritance from parent agent
      - Type-based tool filtering + toolsets isolation
      - Evidence chain inheritance (sub -> parent)
      - Sequential (run_sync) and parallel (run_parallel) execution
      - Orchestrator role support with delegation tree
    """

    def __init__(self, model_provider: ModelProvider):
        self._model = model_provider
        self._futures: dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()

    # ---- Sequential Execution -----------------------------------------

    def run_sync(
        self,
        prompt: str,
        sub_type: SubAgentType,
        parent_case: Case | None = None,
        toolsets: list[str] | None = None,
        parent_trace: Any | None = None,
    ) -> SubAgentResult:
        """Run a sub-agent synchronously, merging evidence into the parent case."""
        sub_case = Case(id=f"sub_{uuid.uuid4().hex[:8]}")
        t0 = time.time()

        try:
            agent = self._build_agent(prompt, sub_type, sub_case, toolsets=toolsets)
            result = agent.run(prompt, parent_trace=parent_trace)

            if parent_case:
                self._merge_evidence(sub_case, parent_case)

            return SubAgentResult(
                subagent_type=sub_type.name,
                description=prompt[:100],
                status="success",
                content=result.get("content"),
                confidence=result.get("confidence"),
                evidence=result.get("evidence", []),
                sub_case_id=sub_case.id,
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SubAgentResult(
                subagent_type=sub_type.name,
                description=prompt[:100],
                status="error",
                content=None,
                confidence=None,
                evidence=[],
                sub_case_id=sub_case.id,
                error=str(e),
                duration_ms=(time.time() - t0) * 1000,
            )

    # ---- Parallel Batch Execution -------------------------------------

    def run_parallel(
        self,
        tasks: list[TaskSpec],
        batch_timeout: float = 600.0,
        max_workers: int = 8,
    ) -> list[SubAgentResult]:
        """Execute multiple sub-agents in parallel and collect results.

        Args:
            tasks: List of TaskSpec defining each sub-agent.
            batch_timeout: Maximum total time for the entire batch.
            max_workers: Max concurrent workers for this batch.

        Returns:
            Results in the same order as input tasks.
            Timed-out tasks get status="timeout".
        """
        n = len(tasks)
        if n == 0:
            return []
        if n == 1:
            t = tasks[0]
            return [self.run_sync(t.prompt, t.sub_type, t.parent_case, t.toolsets, t.parent_trace)]

        results: list[SubAgentResult | None] = [None] * n
        errors: list[tuple[int, str]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
            future_to_idx: dict[concurrent.futures.Future, int] = {}

            for i, task in enumerate(tasks):
                fut = pool.submit(
                    self._run_single_task, task, i
                )
                future_to_idx[fut] = i

            done, not_done = concurrent.futures.wait(
                future_to_idx.keys(),
                timeout=batch_timeout,
                return_when=concurrent.futures.ALL_COMPLETED,
            )

            for fut in not_done:
                idx = future_to_idx[fut]
                fut.cancel()
                results[idx] = SubAgentResult(
                    subagent_type=tasks[idx].sub_type.name,
                    description=tasks[idx].description or tasks[idx].prompt[:100],
                    status="timeout",
                    content=None,
                    confidence=None,
                    evidence=[],
                    sub_case_id=f"timeout_{idx}",
                    error=f"Task timed out after {batch_timeout}s",
                )

            for fut in done:
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result(timeout=1)
                except Exception as e:
                    results[idx] = SubAgentResult(
                        subagent_type=tasks[idx].sub_type.name,
                        description=tasks[idx].description or tasks[idx].prompt[:100],
                        status="error",
                        content=None,
                        confidence=None,
                        evidence=[],
                        sub_case_id=f"err_{idx}",
                        error=str(e),
                    )

        return results  # type: ignore[return-value]

    def _run_single_task(self, task: TaskSpec, idx: int) -> SubAgentResult:
        """Wrapper for single-task execution with timeout."""
        t0 = time.time()

        try:
            future = _executor_pool.submit(
                self.run_sync, task.prompt, task.sub_type, task.parent_case, task.toolsets, task.parent_trace
            )
            return future.result(timeout=task.timeout)
        except concurrent.futures.TimeoutError:
            return SubAgentResult(
                subagent_type=task.sub_type.name,
                description=task.description or task.prompt[:100],
                status="timeout",
                content=None,
                confidence=None,
                evidence=[],
                sub_case_id=f"timeout_{idx}",
                error=f"Task timed out after {task.timeout}s",
                duration_ms=(time.time() - t0) * 1000,
            )

    # ---- Async Execution -----------------------------------------------

    def run_async(
        self,
        prompt: str,
        sub_type: SubAgentType,
        parent_case: Case | None = None,
        toolsets: list[str] | None = None,
        parent_trace: Any | None = None,
    ) -> str:
        """Run a sub-agent asynchronously. Returns a future_id for polling."""
        future_id = f"sub_{uuid.uuid4().hex[:8]}"

        def _run():
            return self.run_sync(prompt, sub_type, parent_case, toolsets, parent_trace)

        with self._lock:
            self._futures[future_id] = _executor_pool.submit(_run)
        return future_id

    def poll(self, future_id: str, timeout: float | None = None) -> SubAgentResult | None:
        """Poll an async sub-agent. Returns None if still running."""
        future = self._futures.get(future_id)
        if not future:
            return SubAgentResult(
                subagent_type="", description="", status="error",
                content=None, confidence=None, evidence=[], sub_case_id="",
                error=f"No future found: {future_id}",
            )

        try:
            if timeout is not None:
                return future.result(timeout=timeout)
            if not future.done():
                return None
            return future.result()
        except concurrent.futures.TimeoutError:
            return None
        except Exception as e:
            return SubAgentResult(
                subagent_type="", description="", status="error",
                content=None, confidence=None, evidence=[], sub_case_id="",
                error=str(e),
            )

    # ---- Agent Builder -------------------------------------------------

    def _build_agent(
        self,
        prompt: str,
        sub_type: SubAgentType,
        sub_case: Case,
        toolsets: list[str] | None = None,
    ):
        """Build a minimal Agent configured for this sub-agent type."""
        from kairos.core.loop import Agent

        return Agent(
            model=ModelConfig(
                api_key=self._model.config.api_key,
                base_url=self._model.config.base_url,
                model=sub_type.model if sub_type.model != "inherit" else self._model.config.model,
            ),
            agent_name=f"SubAgent({sub_type.name})",
            role_description=sub_type.system_prompt or f"Sub-agent of type: {sub_type.name}",
            max_iterations=sub_type.max_turns,
        )

    # ---- Evidence ------------------------------------------------------

    @staticmethod
    def _merge_evidence(sub_case: Case, parent_case: Case) -> None:
        """Merge sub-agent evidence steps into the parent case."""
        for step in sub_case.steps:
            parent_case.steps.append(step)
        if sub_case.conclusion and not parent_case.conclusion:
            parent_case.conclusion = sub_case.conclusion

    # ---- Type Management -----------------------------------------------

    @staticmethod
    def get_type(name: str) -> SubAgentType | None:
        return BUILTIN_TYPES.get(name)

    @staticmethod
    def register_type(sub_type: SubAgentType) -> None:
        BUILTIN_TYPES[sub_type.name] = sub_type

    @staticmethod
    def list_types() -> list[str]:
        return list(BUILTIN_TYPES.keys())
