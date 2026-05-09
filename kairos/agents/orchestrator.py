"""Orchestrator sub-agent — plans work and spawns child sub-agents with delegation tree management.

Provides:
  - DelegationManager: singleton tracking the delegation tree, depth limits, concurrency
  - OrchestratorRole: a sub-agent role that CAN spawn more sub-agents recursively
  - Depth-aware turn limiting and timeout enforcement
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from kairos.agents.executor import SubAgentExecutor, SubAgentResult, TaskSpec
from kairos.agents.types import BUILTIN_TYPES, SubAgentType
from kairos.providers.base import ModelProvider

logger = logging.getLogger("kairos.orchestrator")


# ═══════════════════════════════════════════════════════════════════════════
# Delegation Tree Node
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DelegationNode:
    """A node in the delegation tree."""

    agent_id: str
    parent_id: str | None
    depth: int
    subagent_type: str
    status: str = "pending"  # pending | running | success | error | timeout | cancelled
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: SubAgentResult | None = None
    children: list[str] = field(default_factory=list)  # child agent_ids


# ═══════════════════════════════════════════════════════════════════════════
# DelegationManager — singleton tracking the full delegation tree
# ═══════════════════════════════════════════════════════════════════════════


class DelegationManager:
    """Singleton tracking all active delegations in a tree structure.

    Enforces:
      - max_depth: prevents infinite recursion (default 2)
      - max_concurrent_per_depth: limits parallelism at each depth level
      - Per-agent timeout and cancellation

    Usage::

        dm = DelegationManager(max_depth=3)
        ok = dm.register(parent_id="root", child_id="sub_1", depth=1)
        if not ok:
            print("Depth limit exceeded")

        tree = dm.get_tree("root")  # full subtree
    """

    _instance: DelegationManager | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> DelegationManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        max_depth: int = 2,
        max_concurrent_per_depth: int = 5,
        default_timeout: float = 300.0,
    ) -> None:
        # Only init once (singleton)
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.max_depth = max(1, max_depth)
        self.max_concurrent_per_depth = max(1, max_concurrent_per_depth)
        self.default_timeout = default_timeout

        self._tree: dict[str, DelegationNode] = {}
        self._tree_lock = threading.Lock()

        # Track running count per depth for concurrency control
        self._active_at_depth: dict[int, int] = {}
        self._depth_lock = threading.Lock()

        # Cancellation flags
        self._cancelled: set[str] = set()
        self._cancel_lock = threading.Lock()

        logger.info(
            "DelegationManager initialized: max_depth=%d, max_concurrent_per_depth=%d",
            self.max_depth,
            self.max_concurrent_per_depth,
        )

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        parent_id: str,
        child_id: str,
        depth: int,
        subagent_type: str = "general-purpose",
    ) -> bool:
        """Register a child agent in the delegation tree.

        Returns False if depth limit is exceeded or concurrency limit is hit.
        """
        if depth > self.max_depth:
            logger.warning(
                "Depth %d exceeds max_depth %d for child %s",
                depth, self.max_depth, child_id,
            )
            return False

        with self._depth_lock:
            current = self._active_at_depth.get(depth, 0)
            if current >= self.max_concurrent_per_depth:
                logger.warning(
                    "Concurrency limit reached at depth %d (%d active)",
                    depth, current,
                )
                return False
            self._active_at_depth[depth] = current + 1

        node = DelegationNode(
            agent_id=child_id,
            parent_id=parent_id,
            depth=depth,
            subagent_type=subagent_type,
        )

        with self._tree_lock:
            self._tree[child_id] = node
            if parent_id and parent_id in self._tree:
                self._tree[parent_id].children.append(child_id)
            # Ensure root exists even if not pre-registered
            elif parent_id and parent_id not in self._tree:
                self._tree[parent_id] = DelegationNode(
                    agent_id=parent_id,
                    parent_id=None,
                    depth=depth - 1,
                    subagent_type="root",
                )
                self._tree[parent_id].children.append(child_id)

        logger.debug("Registered %s (depth=%d, parent=%s)", child_id, depth, parent_id)
        return True

    def mark_running(self, agent_id: str) -> None:
        """Mark an agent as running."""
        with self._tree_lock:
            if agent_id in self._tree:
                self._tree[agent_id].status = "running"

    def mark_complete(self, agent_id: str, result: SubAgentResult) -> None:
        """Mark an agent as complete with its result."""
        with self._tree_lock:
            if agent_id in self._tree:
                node = self._tree[agent_id]
                node.status = result.status
                node.result = result
                node.finished_at = time.time()

        with self._depth_lock:
            depth = self._tree[agent_id].depth if agent_id in self._tree else 0
            self._active_at_depth[depth] = max(0, self._active_at_depth.get(depth, 1) - 1)

    # ── Query ─────────────────────────────────────────────────────────────

    def active_count(self, depth: int | None = None) -> int:
        """Count active agents, optionally filtered by depth."""
        with self._tree_lock:
            if depth is not None:
                return sum(
                    1 for n in self._tree.values()
                    if n.depth == depth and n.status in ("pending", "running")
                )
            return sum(
                1 for n in self._tree.values()
                if n.status in ("pending", "running")
            )

    def depth_count(self) -> dict[int, int]:
        """Return active count per depth level."""
        with self._tree_lock:
            counts: dict[int, int] = {}
            for n in self._tree.values():
                if n.status in ("pending", "running"):
                    counts[n.depth] = counts.get(n.depth, 0) + 1
            return counts

    def get_tree(self, root_id: str) -> dict[str, Any]:
        """Return the full delegation tree rooted at root_id.

        Returns a nested dict suitable for visualization or debugging.
        """
        with self._tree_lock:
            if root_id not in self._tree:
                return {"error": f"Unknown agent: {root_id}"}
            return self._build_tree_dict(root_id)

    def _build_tree_dict(self, agent_id: str) -> dict[str, Any]:
        """Recursively build a nested tree dict."""
        node = self._tree[agent_id]
        result_summary = None
        if node.result:
            result_summary = {
                "status": node.result.status,
                "content": (node.result.content or "")[:200],
                "confidence": node.result.confidence,
                "error": node.result.error,
            }

        return {
            "agent_id": agent_id,
            "depth": node.depth,
            "subagent_type": node.subagent_type,
            "status": node.status,
            "created_at": node.created_at,
            "finished_at": node.finished_at,
            "result": result_summary,
            "children": [
                self._build_tree_dict(child_id)
                for child_id in node.children
            ],
        }

    def get_node(self, agent_id: str) -> DelegationNode | None:
        """Get a single delegation tree node."""
        with self._tree_lock:
            return self._tree.get(agent_id)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def cancel_all(self, depth: int | None = None) -> int:
        """Cancel all agents at a given depth (or all depths if None).

        Returns the number of agents cancelled.
        """
        count = 0
        with self._cancel_lock:
            with self._tree_lock:
                for agent_id, node in self._tree.items():
                    if depth is not None and node.depth != depth:
                        continue
                    if node.status in ("pending", "running"):
                        self._cancelled.add(agent_id)
                        node.status = "cancelled"
                        node.finished_at = time.time()
                        count += 1
        if count:
            logger.info("Cancelled %d agents at depth=%s", count, depth or "all")
        return count

    def is_cancelled(self, agent_id: str) -> bool:
        """Check if an agent has been cancelled."""
        with self._cancel_lock:
            return agent_id in self._cancelled

    def reset(self) -> None:
        """Reset the entire delegation tree (for testing)."""
        with self._tree_lock:
            self._tree.clear()
        with self._depth_lock:
            self._active_at_depth.clear()
        with self._cancel_lock:
            self._cancelled.clear()
        logger.debug("DelegationManager reset")

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the delegation tree."""
        with self._tree_lock:
            total = len(self._tree)
            by_status: dict[str, int] = {}
            by_type: dict[str, int] = {}
            max_seen_depth = 0
            for n in self._tree.values():
                by_status[n.status] = by_status.get(n.status, 0) + 1
                by_type[n.subagent_type] = by_type.get(n.subagent_type, 0) + 1
                max_seen_depth = max(max_seen_depth, n.depth)

            return {
                "total_agents": total,
                "by_status": by_status,
                "by_type": by_type,
                "max_depth_seen": max_seen_depth,
                "max_depth_limit": self.max_depth,
                "active_by_depth": dict(self._active_at_depth),
            }

    def __repr__(self) -> str:
        return (
            f"DelegationManager(depth={self.max_depth}, "
            f"active={self.active_count()}, total_nodes={len(self._tree)})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# OrchestratorRole
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OrchestratorConfig:
    """Configuration for an OrchestratorRole."""

    max_depth: int = 2
    max_children_per_level: int = 5
    child_timeout: float = 300.0
    default_child_type: str = "general-purpose"
    # Turn scaling: each depth level gets fewer turns
    base_turns: int = 20
    depth_turn_divisor: int = 2  # turns = base_turns / (divisor ** depth)


@dataclass
class OrchestratorResult:
    """Result from an orchestrator run, including all child results."""

    agent_id: str
    status: str  # success | error | timeout
    content: str | None
    confidence: float | None
    evidence: list[dict[str, Any]]
    child_results: list[SubAgentResult]
    delegation_tree: dict[str, Any]
    error: str | None = None


class OrchestratorRole:
    """A sub-agent that can plan and delegate work to child sub-agents.

    Unlike leaf sub-agents, an orchestrator:
      - Breaks down a complex task into sub-tasks
      - Spawns child sub-agents to execute them
      - Collects and synthesizes results
      - Respects depth limits to prevent infinite recursion

    Usage::

        orch = OrchestratorRole(
            executor=sub_executor,
            config=OrchestratorConfig(max_depth=2),
            depth=0,
        )
        result = orch.run("Analyze the codebase and find all security issues")
    """

    def __init__(
        self,
        executor: SubAgentExecutor,
        config: OrchestratorConfig | None = None,
        depth: int = 0,
        parent_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._executor = executor
        self._config = config or OrchestratorConfig()
        self._depth = depth
        self._parent_id = parent_id
        self._agent_id = agent_id or f"orch_{uuid.uuid4().hex[:8]}"
        self._dm = DelegationManager()

        # Register self in the delegation tree
        if not self._dm.get_node(self._agent_id):
            self._dm.register(
                parent_id=self._parent_id or "root",
                child_id=self._agent_id,
                depth=self._depth,
                subagent_type="orchestrator",
            )

        # Compute max turns for this depth
        divisor = max(1, self._config.depth_turn_divisor ** self._depth)
        self._max_turns = max(1, self._config.base_turns // divisor)

        logger.debug(
            "OrchestratorRole %s created: depth=%d, max_turns=%d",
            self._agent_id, self._depth, self._max_turns,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def can_delegate(self) -> bool:
        """Can this orchestrator spawn more children?"""
        return self._depth < self._config.max_depth

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self, prompt: str) -> OrchestratorResult:
        """Execute the orchestrator: plan, delegate, synthesize."""
        self._dm.mark_running(self._agent_id)

        try:
            # Phase 1: Plan — break down the task
            child_tasks = self._plan(prompt)

            # Phase 2: Delegate — spawn child sub-agents
            child_results: list[SubAgentResult] = []
            if child_tasks and self.can_delegate:
                child_results = self._delegate_children(child_tasks)

            # Phase 3: Synthesize — combine results
            content, confidence = self._synthesize(prompt, child_results)

            result = OrchestratorResult(
                agent_id=self._agent_id,
                status="success",
                content=content,
                confidence=confidence,
                evidence=[],
                child_results=child_results,
                delegation_tree=self._dm.get_tree(self._agent_id),
            )

            self._dm.mark_complete(self._agent_id, SubAgentResult(
                subagent_type="orchestrator",
                description=prompt[:100],
                status="success",
                content=content,
                confidence=confidence,
                evidence=[],
                sub_case_id=self._agent_id,
            ))

            return result

        except Exception as e:
            logger.error("Orchestrator %s failed: %s", self._agent_id, e)
            error_result = OrchestratorResult(
                agent_id=self._agent_id,
                status="error",
                content=None,
                confidence=None,
                evidence=[],
                child_results=[],
                delegation_tree=self._dm.get_tree(self._agent_id),
                error=str(e),
            )
            self._dm.mark_complete(self._agent_id, SubAgentResult(
                subagent_type="orchestrator",
                description=prompt[:100],
                status="error",
                content=None,
                confidence=None,
                evidence=[],
                sub_case_id=self._agent_id,
                error=str(e),
            ))
            return error_result

    # ── Internal phases ───────────────────────────────────────────────────

    def _plan(self, prompt: str) -> list[TaskSpec]:
        """Break down a complex prompt into child tasks.

        Uses a lightweight LLM call for planning. Falls back to heuristic
        decomposition if the LLM call fails.
        """
        # Try LLM-based planning via a brief sub-agent
        plan_prompt = (
            f"You are a task planner. Break down the following task into "
            f"independent sub-tasks that can run in parallel. Return a JSON list "
            f"of objects with 'prompt' (detailed instruction), 'toolsets' (list of "
            f"toolset names: terminal, file, web, search), and 'type' "
            f"(sub-agent type like general-purpose, bash, research).\n\n"
            f"TASK: {prompt}\n\n"
            f"Respond ONLY with a valid JSON array. Maximum {self._config.max_children_per_level} sub-tasks."
        )

        try:
            planner_type = SubAgentType(
                name="planner",
                description="Task decomposition planner",
                tools=[],  # No tools needed — just reasoning
                max_turns=3,
                timeout_seconds=60,
            )
            plan_result = self._executor.run_sync(plan_prompt, planner_type)

            if plan_result.status == "success" and plan_result.content:
                import json
                raw = plan_result.content.strip()
                # Handle markdown code blocks
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:]) if len(lines) > 1 else raw
                    if raw.endswith("```"):
                        raw = raw[:-3]
                tasks_data = json.loads(raw)
                if isinstance(tasks_data, list):
                    return self._tasks_from_json(tasks_data)
        except Exception as e:
            logger.debug("LLM planning failed, using heuristic: %s", e)

        # Fallback: heuristic decomposition — one task per sentence/clause
        return self._heuristic_plan(prompt)

    def _heuristic_plan(self, prompt: str) -> list[TaskSpec]:
        """Simple heuristic: split by newlines or sentences for parallel work."""
        # Split by newlines first
        parts = [p.strip() for p in prompt.split("\n") if p.strip()]
        if len(parts) <= 1:
            # Split by sentences
            import re
            parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', prompt) if p.strip()]

        # Limit
        parts = parts[:self._config.max_children_per_level]
        if not parts:
            parts = [prompt]

        return [
            TaskSpec(
                prompt=p,
                sub_type=self._config.default_child_type,
                toolsets=[],
                role="leaf",
                timeout=self._config.child_timeout,
            )
            for p in parts
        ]

    def _tasks_from_json(self, data: list[dict[str, Any]]) -> list[TaskSpec]:
        """Convert JSON task descriptions to TaskSpec objects."""
        tasks = []
        for item in data[:self._config.max_children_per_level]:
            tasks.append(TaskSpec(
                prompt=item.get("prompt", item.get("task", "")),
                sub_type=item.get("type", item.get("subagent_type", self._config.default_child_type)),
                toolsets=item.get("toolsets", []),
                role="leaf" if self._depth + 1 >= self._config.max_depth else "leaf",
                timeout=item.get("timeout", self._config.child_timeout),
            ))
        return tasks

    def _delegate_children(self, tasks: list[TaskSpec]) -> list[SubAgentResult]:
        """Spawn child sub-agents and collect results."""
        if not tasks:
            return []

        child_depth = self._depth + 1

        # Register children in delegation tree before execution
        for i, task in enumerate(tasks):
            child_id = f"{self._agent_id}_child_{i}"
            task.parent_case = child_id  # Use as identifier
            ok = self._dm.register(
                parent_id=self._agent_id,
                child_id=child_id,
                depth=child_depth,
                subagent_type=task.sub_type,
            )
            if not ok:
                logger.warning("Failed to register child %s at depth %d", child_id, child_depth)
                task.parent_case = None
            else:
                task.parent_case = child_id

        # Filter out unregistered tasks
        valid_tasks = [t for t in tasks if t.parent_case is not None]

        if not valid_tasks:
            logger.warning("No valid child tasks to delegate at depth %d", child_depth)
            return []

        # Execute in parallel
        logger.info(
            "Orchestrator %s delegating %d tasks at depth %d",
            self._agent_id, len(valid_tasks), child_depth,
        )

        for t in valid_tasks:
            self._dm.mark_running(t.parent_case)

        results = self._executor.run_parallel(valid_tasks)

        for t, r in zip(valid_tasks, results):
            if t.parent_case:
                self._dm.mark_complete(t.parent_case, r)

        return results

    def _synthesize(
        self,
        original_prompt: str,
        child_results: list[SubAgentResult],
    ) -> tuple[str | None, float | None]:
        """Synthesize child results into a coherent response."""
        if not child_results:
            return "No child tasks were executed.", 0.0

        success_count = sum(1 for r in child_results if r.status == "success")
        total = len(child_results)

        # Build a synthesis of all child outputs
        parts: list[str] = []
        for i, r in enumerate(child_results):
            if r.status == "success" and r.content:
                parts.append(f"### Sub-task {i + 1} Result\n{r.content}")
            elif r.error:
                parts.append(f"### Sub-task {i + 1} Error\n{r.error}")

        synthesis = f"Orchestrator completed {success_count}/{total} sub-tasks successfully.\n\n"
        synthesis += "\n\n".join(parts)

        confidence = success_count / max(total, 1)
        return synthesis, confidence

    # ── Static helpers ────────────────────────────────────────────────────

    @staticmethod
    def depth_turns(base_turns: int, depth: int, divisor: int = 2) -> int:
        """Compute max turns for a given depth."""
        return max(1, base_turns // max(1, divisor ** depth))

    def __repr__(self) -> str:
        return (
            f"OrchestratorRole(id={self._agent_id}, depth={self._depth}, "
            f"max_turns={self._max_turns}, can_delegate={self.can_delegate})"
        )
