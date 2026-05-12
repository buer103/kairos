"""Sub-agent delegation — delegate tasks to child agents in isolated contexts.

Mirrors Hermes's delegate_task workflow:
  - Spawn one or more sub-agents concurrently
  - Each runs in its own isolated Agent instance
  - Collect and merge results
  - Timeout, concurrency limits, and error handling
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("kairos.delegate")


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

@dataclass
class DelegateTask:
    """A single task to delegate to a sub-agent."""

    id: str = ""
    goal: str = ""
    context: str = ""  # Background info injected into sub-agent
    tools: list[str] = field(default_factory=list)  # Tool names to enable
    timeout: float = 180.0
    model_override: str = ""  # Override model for this task
    role: str = "worker"  # worker | reviewer | researcher

    def __post_init__(self):
        if not self.id:
            self.id = f"subtask_{uuid.uuid4().hex[:8]}"


@dataclass
class DelegateResult:
    """Result from a delegated task."""

    task_id: str
    success: bool
    content: str = ""
    error: str = ""
    duration_ms: float = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0


# ═══════════════════════════════════════════════════════════════
# Sub-agent
# ═══════════════════════════════════════════════════════════════

class CancelEvent:
    """Lightweight cancellation token for sub-agent graceful shutdown.

    Hermes-aligned pattern: sub-agents check this event periodically
    during their execution loop. When set, they stop gracefully and
    return a partial result instead of crashing.

    Usage::

        cancel = CancelEvent()
        # In sub-agent loop:
        if cancel.is_set():
            return partial_result
        # From parent:
        cancel.set()  # signals all sub-agents to stop
    """

    def __init__(self, reason: str = "Cancelled"):
        self._event = threading.Event()
        self.reason = reason
        self._set_at: float = 0.0

    def set(self, reason: str | None = None) -> None:
        if reason:
            self.reason = reason
        self._set_at = time.time()
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def clear(self) -> None:
        self._event.clear()
        self._set_at = 0.0

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def set_at(self) -> float:
        return self._set_at

    def __repr__(self) -> str:
        state = "set" if self._event.is_set() else "clear"
        return f"CancelEvent({state}, reason={self.reason!r})"


class SubAgent:
    """A lightweight agent spawned to handle a single delegated task.

    Runs in its own context with a subset of tools and no middleware
    (or minimal middleware — just the essential layers).
    """

    def __init__(
        self,
        task: DelegateTask,
        model_provider: Any,
        tool_registry: Any = None,
        system_prompt: str = "",
        cancel_event: CancelEvent | None = None,
    ):
        self.task = task
        self.model = model_provider
        self.tool_registry = tool_registry
        self.cancel_event = cancel_event or CancelEvent()
        self.system_prompt = system_prompt or (
            "You are a focused sub-agent. Complete the assigned task precisely "
            "and return a concise result. Do NOT ask for clarification — make "
            "your best judgment and deliver the result.\n\n"
            f"TASK: {task.goal}\n\n"
            f"CONTEXT: {task.context or 'None provided.'}"
        )

    def run(self) -> DelegateResult:
        """Execute the delegated task and return the result.

        Checks cancel_event before execution — if cancelled, returns
        immediately with a cancellation result.
        """
        start = time.time()

        # Check cancellation before starting
        if self.cancel_event.is_set():
            elapsed = (time.time() - start) * 1000
            logger.info("Sub-agent %s cancelled before start: %s",
                        self.task.id, self.cancel_event.reason)
            return DelegateResult(
                task_id=self.task.id,
                success=False,
                error=f"Cancelled: {self.cancel_event.reason}",
                duration_ms=elapsed,
            )

        try:
            # Capture parent trace context for full-chain observability
            from kairos.core.tracing import get_current_trace
            parent_trace = get_current_trace()

            # Run agent loop
            from kairos import Agent
            agent = Agent(
                model=self.model,
                role_description=self.system_prompt,
                max_iterations=10,
            )

            # Check cancellation after agent creation (may have waited)
            if self.cancel_event.is_set():
                elapsed = (time.time() - start) * 1000
                logger.info("Sub-agent %s cancelled before run: %s",
                            self.task.id, self.cancel_event.reason)
                return DelegateResult(
                    task_id=self.task.id,
                    success=False,
                    error=f"Cancelled: {self.cancel_event.reason}",
                    duration_ms=elapsed,
                )

            result = agent.run(
                self.task.goal,
                parent_trace=parent_trace,
            )
            elapsed = (time.time() - start) * 1000

            return DelegateResult(
                task_id=self.task.id,
                success=True,
                content=result.get("content", ""),
                evidence=result.get("evidence", []),
                duration_ms=elapsed,
                tool_calls=len(result.get("evidence", [])),
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error("Sub-agent %s failed: %s", self.task.id, e)
            return DelegateResult(
                task_id=self.task.id,
                success=False,
                error=str(e),
                duration_ms=elapsed,
            )


# ═══════════════════════════════════════════════════════════════
# Delegation manager
# ═══════════════════════════════════════════════════════════════

@dataclass
class DelegateConfig:
    """Configuration for the delegation system."""

    max_concurrent: int = 3
    max_tasks: int = 3       # Hard cap — reject batches larger than this
    default_timeout: float = 180.0
    max_retries: int = 1
    verbose: bool = False


class DelegationManager:
    """Manages sub-agent delegation with concurrency control.

    Usage::

        mgr = DelegationManager(model=model_config, config=DelegateConfig(max_concurrent=5))

        # Single task
        result = mgr.delegate(DelegateTask(
            goal="Search for the latest Python 3.13 release notes",
            context="We need the changelog URL and key highlights.",
        ))

        # Parallel batch
        results = mgr.delegate_batch([
            DelegateTask(goal="Task A", context="..."),
            DelegateTask(goal="Task B", context="..."),
            DelegateTask(goal="Task C", context="..."),
        ])
    """

    def __init__(
        self,
        model: Any,
        config: DelegateConfig | None = None,
        system_prompt: str = "",
    ):
        self.model = model
        self.config = config or DelegateConfig()
        self.system_prompt = system_prompt
        self._lock = threading.Lock()
        self._active_count = 0
        self._active_cancel_events: list[CancelEvent] = []
        self._global_cancel = CancelEvent()

    def delegate(self, task: DelegateTask) -> DelegateResult:
        """Delegate a single task. Runs synchronously."""
        results = self.delegate_batch([task])
        return results[0] if results else DelegateResult(
            task_id=task.id, success=False, error="No result"
        )

    def delegate_batch(self, tasks: list[DelegateTask]) -> list[DelegateResult]:
        """Delegate multiple tasks in parallel (up to max_concurrent)."""
        if not tasks:
            return []

        # Hard cap enforcement
        if len(tasks) > self.config.max_tasks:
            return [DelegateResult(
                task_id=tasks[i].id if i < len(tasks) else f"overflow_{i}",
                success=False,
                error=f"Too many tasks: {len(tasks)} requested, max {self.config.max_tasks}",
            ) for i in range(len(tasks))]

        max_workers = min(len(tasks), self.config.max_concurrent)
        results: list[DelegateResult] = []

        # Track cancel events for active sub-agents
        cancel_events: list[CancelEvent] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="kairos-sub",
        ) as executor:
            futures: dict[concurrent.futures.Future, str] = {}

            for task in tasks:
                # Check global cancellation before spawning
                if self._global_cancel.is_set():
                    results.append(DelegateResult(
                        task_id=task.id,
                        success=False,
                        error=f"Batch cancelled: {self._global_cancel.reason}",
                    ))
                    continue

                cancel = CancelEvent()
                cancel_events.append(cancel)
                with self._lock:
                    self._active_cancel_events.append(cancel)

                sub = SubAgent(
                    task=task,
                    model_provider=self.model,
                    system_prompt=self.system_prompt,
                    cancel_event=cancel,
                )
                future = executor.submit(
                    self._run_with_timeout, sub, task.timeout or self.config.default_timeout
                )
                futures[future] = task.id

            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result(timeout=10)  # collect timeout
                    results.append(result)
                except Exception as e:
                    logger.error("Task %s failed to collect: %s", task_id, e)
                    results.append(DelegateResult(
                        task_id=task_id,
                        success=False,
                        error=f"Collection error: {e}",
                    ))

        # Clean up cancel events
        with self._lock:
            for ce in cancel_events:
                if ce in self._active_cancel_events:
                    self._active_cancel_events.remove(ce)

        # Sort by task order
        id_order = {t.id: i for i, t in enumerate(tasks)}
        results.sort(key=lambda r: id_order.get(r.task_id, 999))

        return results

    def delegate_dict(self, task_specs: list[dict[str, Any]]) -> list[DelegateResult]:
        """Convenience: delegate from a list of dicts (as LLM would provide)."""
        tasks = []
        for spec in task_specs:
            tasks.append(DelegateTask(
                id=spec.get("id", ""),
                goal=spec.get("goal", spec.get("task", "")),
                context=spec.get("context", ""),
                tools=spec.get("tools", []),
                timeout=spec.get("timeout", self.config.default_timeout),
            ))
        return self.delegate_batch(tasks)

    def _run_with_timeout(self, sub: SubAgent, timeout: float) -> DelegateResult:
        """Run a sub-agent with a timeout wrapper.

        On timeout, signals the sub-agent's cancel_event for graceful shutdown
        before returning the timeout error.
        """
        start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as inner:
            future = inner.submit(sub.run)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                # Signal cancellation so the sub-agent stops its LLM call
                sub.cancel_event.set(reason=f"Timeout after {timeout}s")
                elapsed = (time.time() - start) * 1000
                logger.warning(
                    "Sub-agent %s timed out after %.1fs — cancel event set",
                    sub.task.id, timeout,
                )
                return DelegateResult(
                    task_id=sub.task.id,
                    success=False,
                    error=f"Timed out after {timeout}s",
                    duration_ms=elapsed,
                )

    def cancel_all(self, reason: str = "Cancelled by user") -> int:
        """Cancel ALL running sub-agents immediately.

        Signals every active sub-agent's cancel_event.
        Returns the count of sub-agents cancelled.

        Thread-safe — can be called from any thread.
        """
        with self._lock:
            self._global_cancel.set(reason=reason)
            cancelled = len(self._active_cancel_events)
            for ce in self._active_cancel_events:
                ce.set(reason=reason)
            logger.info(
                "Cancelled %d active sub-agents: %s", cancelled, reason,
            )
            return cancelled

    @property
    def active_count(self) -> int:
        """Number of currently active sub-agents."""
        with self._lock:
            return len(self._active_cancel_events)

    @property
    def cancelled(self) -> bool:
        """Whether global cancellation has been requested."""
        return self._global_cancel.is_set()


# ═══════════════════════════════════════════════════════════════
# Tool binding
# ═══════════════════════════════════════════════════════════════

def register_delegate_tool(delegation_manager: DelegationManager) -> None:
    """Register delegate_task as a tool visible to the agent.

    The agent can then spawn sub-agents programmatically::

        delegate_task(goal="Research X", context="We need Y")
        delegate_task(tasks=[
            {"goal": "Task A", "context": "..."},
            {"goal": "Task B", "context": "..."},
        ])
    """

    from kairos.tools.registry import register_tool, execute_tool

    @register_tool(
        name="delegate_task",
        description=(
            "Delegate one or more tasks to sub-agents. Sub-agents work in parallel "
            "and return consolidated results. Use for complex multi-step research, "
            "parallel data gathering, or dividing a large task into independent sub-tasks."
        ),
        parameters={
            "goal": {
                "type": "string",
                "description": "Single task goal. Omit if using 'tasks' for batch mode.",
            },
            "context": {
                "type": "string",
                "description": "Background information the sub-agent needs to complete the task.",
            },
            "tasks": {
                "type": "array",
                "description": (
                    'Batch mode: list of {"goal": str, "context": str, "tools": [str]} objects. '
                    "Use this instead of 'goal' to run multiple tasks in parallel."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout per task in seconds (default: 180, max: 600)",
            },
        },
        timeout=600,  # Sub-agent makes LLM calls — needs generous timeout
    )
    def delegate_task(
        goal: str = "",
        context: str = "",
        tasks: list[dict[str, Any]] | None = None,
        timeout: int = 180,
    ) -> dict[str, Any]:
        """Delegate tool — called by the agent."""
        timeout = min(max(timeout, 10), 600)

        if tasks:
            # Batch mode — enforce hard cap
            if len(tasks) > delegation_manager.config.max_tasks:
                return {
                    "error": f"Too many tasks: {len(tasks)} requested, max {delegation_manager.config.max_tasks}",
                    "results": [],
                }
            batch: list[DelegateTask] = []
            for t in tasks:
                batch.append(DelegateTask(
                    goal=t.get("goal", t.get("task", "")),
                    context=t.get("context", ""),
                    timeout=t.get("timeout", timeout),
                ))
            results = delegation_manager.delegate_batch(batch)
        elif goal:
            # Single task
            task = DelegateTask(goal=goal, context=context, timeout=timeout)
            results = [delegation_manager.delegate(task)]
        else:
            return {"error": "Provide either 'goal' or 'tasks'.", "results": []}

        return {
            "count": len(results),
            "results": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "content": r.content,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for r in results
            ],
        }
