"""Sub-Agent executor — manages sub-agent lifecycle and evidence inheritance."""

from __future__ import annotations

import concurrent.futures
import uuid
from dataclasses import dataclass
from typing import Any

from kairos.agents.types import BUILTIN_TYPES, SubAgentType
from kairos.core.state import Case
from kairos.providers.base import ModelConfig, ModelProvider

# Thread pool for parallel sub-agent execution
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


@dataclass
class SubAgentResult:
    """Result of a sub-agent execution."""

    subagent_type: str
    description: str
    status: str  # "success", "error", "timeout"
    content: str | None
    confidence: float | None
    evidence: list[dict[str, Any]]
    sub_case_id: str
    error: str | None = None


class SubAgentExecutor:
    """Manages creation and execution of typed sub-agents.

    Handles:
      - Model inheritance from parent agent
      - Type-based tool filtering
      - Evidence chain inheritance (sub → parent)
      - Parallel execution via thread pool
    """

    def __init__(self, model_provider: ModelProvider):
        self._model = model_provider
        self._futures: dict[str, concurrent.futures.Future] = {}

    def run_sync(
        self,
        prompt: str,
        sub_type: SubAgentType,
        parent_case: Case | None = None,
    ) -> SubAgentResult:
        """Run a sub-agent synchronously, merging evidence into the parent case."""
        sub_case = Case(id=f"sub_{uuid.uuid4().hex[:8]}")

        try:
            agent = self._build_agent(prompt, sub_type, sub_case)
            result = agent.run(prompt)

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
            )

    def run_async(
        self,
        prompt: str,
        sub_type: SubAgentType,
        parent_case: Case | None = None,
    ) -> str:
        """Run a sub-agent asynchronously. Returns a future_id for polling."""
        future_id = f"sub_{uuid.uuid4().hex[:8]}"

        def _run():
            return self.run_sync(prompt, sub_type, parent_case)

        self._futures[future_id] = _executor.submit(_run)
        return future_id

    def poll(self, future_id: str, timeout: float | None = None) -> SubAgentResult | None:
        """Poll an async sub-agent. Returns None if still running."""
        future = self._futures.get(future_id)
        if not future:
            return SubAgentResult(
                subagent_type="",
                description="",
                status="error",
                content=None,
                confidence=None,
                evidence=[],
                sub_case_id="",
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
                subagent_type="",
                description="",
                status="error",
                content=None,
                confidence=None,
                evidence=[],
                sub_case_id="",
                error=str(e),
            )

    def _build_agent(
        self,
        prompt: str,
        sub_type: SubAgentType,
        sub_case: Case,
    ):
        """Build a minimal Agent configured for this sub-agent type."""
        from kairos.core.loop import Agent  # noqa: PLC0415 — avoid circular import

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

    @staticmethod
    def _merge_evidence(sub_case: Case, parent_case: Case) -> None:
        """Merge sub-agent evidence steps into the parent case."""
        for step in sub_case.steps:
            parent_case.steps.append(step)
        if sub_case.conclusion and not parent_case.conclusion:
            parent_case.conclusion = sub_case.conclusion

    @staticmethod
    def get_type(name: str) -> SubAgentType | None:
        """Get a sub-agent type by name."""
        return BUILTIN_TYPES.get(name)

    @staticmethod
    def register_type(sub_type: SubAgentType) -> None:
        """Register a custom sub-agent type."""
        BUILTIN_TYPES[sub_type.name] = sub_type

    @staticmethod
    def list_types() -> list[str]:
        """List available sub-agent type names."""
        return list(BUILTIN_TYPES.keys())
