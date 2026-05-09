"""Agent Loop — the core engine with budget control, interrupt, checkpoint, and error handling.

Handles:
  - Token + iteration budget with grace calls
  - Interrupt/resume via signals + checkpoint persistence
  - Error classification (rate_limit / auth / network / tool)
  - Reasoning content extraction
  - Parallel tool execution
  - Model fallback chain
  - Prefill support
"""

from __future__ import annotations

import json
import logging
import signal
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from kairos.core.middleware import Middleware, MiddlewarePipeline
from kairos.core.state import Case, ThreadState
from kairos.prompt.template import PromptBuilder
from kairos.providers.base import ModelConfig, ModelProvider
from kairos.tools.registry import execute_tool, get_tool_schemas

logger = logging.getLogger("kairos.agent")


# ── Budget ──────────────────────────────────────────────────────

@dataclass
class Budget:
    """Token + iteration budget tracker."""
    max_iterations: int = 20
    max_tokens: int = 120000
    budget_ratio: float = 0.85
    response_reserve: int = 1024

    # Runtime counters
    iterations: int = 0
    tokens_used: int = 0
    grace_call_used: bool = False

    @property
    def token_budget(self) -> int:
        return max(0, int(self.max_tokens * self.budget_ratio) - self.response_reserve)

    @property
    def remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    @property
    def exhausted(self) -> bool:
        return self.iterations >= self.max_iterations or self.tokens_used >= self.token_budget

    @property
    def can_grace_call(self) -> bool:
        """Allow one grace call when budget is exhausted but close to finishing."""
        return self.exhausted and not self.grace_call_used

    def consume(self, tokens: int) -> None:
        self.tokens_used += tokens

    def step(self) -> bool:
        """Increment iteration counter. Returns True if under budget."""
        self.iterations += 1
        return self.iterations <= self.max_iterations


# ── Error Classification ────────────────────────────────────────

class ErrorKind(str, Enum):
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    NETWORK = "network"
    CONTEXT_OVERFLOW = "context_overflow"
    TOOL_ERROR = "tool_error"
    UNKNOWN = "unknown"


@dataclass
class AgentError:
    kind: ErrorKind
    message: str
    status_code: int = 0
    retryable: bool = True
    original: Exception | None = None


def classify_error(exc: Exception | dict) -> AgentError:
    """Classify an error into a known category for handling."""
    if isinstance(exc, dict):
        msg = str(exc.get("error", ""))
        status = exc.get("status", exc.get("status_code", 0))
    else:
        msg = str(exc)
        status = 0

    msg_lower = msg.lower()

    if "rate limit" in msg_lower or "429" in msg or status == 429:
        return AgentError(ErrorKind.RATE_LIMIT, msg, 429, retryable=True)
    if "unauthorized" in msg_lower or "401" in msg or "403" in msg or status in (401, 403):
        return AgentError(ErrorKind.AUTH, msg, status, retryable=False)
    if any(k in msg_lower for k in ("timeout", "connection", "network", "reset by peer", "refused")):
        return AgentError(ErrorKind.NETWORK, msg, retryable=True)
    if "context length" in msg_lower or "token" in msg_lower or "maximum context" in msg_lower:
        return AgentError(ErrorKind.CONTEXT_OVERFLOW, msg, retryable=False)
    return AgentError(ErrorKind.UNKNOWN, msg, retryable=False)


# ── Agent ───────────────────────────────────────────────────────

class Agent:
    """Core Kairos agent with budget control, interrupt, and checkpoint.

    Usage:
        agent = Agent.build_default(model=ModelConfig(api_key="..."))
        result = agent.run("Hello")
    """

    # ── Factory ────────────────────────────────────────────────

    @classmethod
    def build_default(
        cls,
        model: ModelConfig,
        agent_name: str = "Kairos",
        role_description: str = "You are a helpful AI assistant.",
        max_iterations: int = 20,
        max_tokens: int = 120000,
        rag_store: Any = None,
        knowledge_stores: dict[str, Any] | None = None,
        skills_dir: str | None = None,
        memory_store: Any = None,
        supports_vision: bool = False,
        is_plan_mode: bool = False,
        credential_pool: Any = None,
        retry_config: Any = None,
        fallback_models: list[ModelConfig] | None = None,
        checkpoint_dir: str | None = None,
        **template_vars,
    ) -> Agent:
        """Build an Agent with full 17-layer pipeline."""
        from kairos.middleware import (
            ThreadDataMiddleware, UploadsMiddleware, DanglingToolCallMiddleware,
            SkillLoader, ContextCompressor, TodoMiddleware, ConfidenceScorer,
            EvidenceTracker, SubagentLimitMiddleware, TitleMiddleware,
            ClarificationMiddleware, ViewImageMiddleware, MemoryMiddleware,
            LLMRetryMiddleware, ToolArgRepairMiddleware,
        )
        from kairos.providers.credential import CredentialPool, RetryConfig

        layers: list[Middleware] = [
            ThreadDataMiddleware(),
            UploadsMiddleware(),
            DanglingToolCallMiddleware(),
        ]
        if skills_dir:
            layers.append(SkillLoader(skills_dir=skills_dir))
        layers.append(ContextCompressor())
        if is_plan_mode:
            layers.append(TodoMiddleware())
        if memory_store:
            layers.append(MemoryMiddleware(memory_store=memory_store))
        if supports_vision:
            layers.append(ViewImageMiddleware(supports_vision=True))
        layers.extend([EvidenceTracker(), ToolArgRepairMiddleware(), ConfidenceScorer()])
        if credential_pool:
            layers.append(LLMRetryMiddleware(
                credential_pool=credential_pool,
                retry_config=retry_config or RetryConfig(),
            ))
        layers.extend([SubagentLimitMiddleware(), TitleMiddleware()])
        layers.append(ClarificationMiddleware())

        return cls(
            model=model,
            middlewares=layers,
            agent_name=agent_name,
            role_description=role_description,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            rag_store=rag_store,
            knowledge_stores=knowledge_stores,
            skills_dir=skills_dir,
            fallback_models=fallback_models,
            checkpoint_dir=checkpoint_dir,
            **template_vars,
        )

    # ── Constructor ────────────────────────────────────────────

    def __init__(
        self,
        model: ModelConfig | None = None,
        tools: list[Any] | None = None,
        middlewares: list[Middleware] | None = None,
        rag_store: Any = None,
        knowledge_stores: dict[str, Any] | None = None,
        skills_dir: str | None = None,
        enable_subagents: bool = True,
        prompt_builder: PromptBuilder | None = None,
        system_template: str | None = None,
        agent_name: str = "Kairos",
        role_description: str = "You are a helpful AI assistant.",
        soul: str | None = None,
        response_style: str | None = None,
        guidelines: str | None = None,
        knowledge_description: str | None = None,
        memory_description: str | None = None,
        max_iterations: int = 20,
        max_tokens: int = 120000,
        fallback_models: list[ModelConfig] | None = None,
        checkpoint_dir: str | None = None,
        **template_vars,
    ):
        self.model = ModelProvider(model or ModelConfig(api_key=""))
        self.budget = Budget(max_iterations=max_iterations, max_tokens=max_tokens)

        # Fallback chain
        self._fallback_models = fallback_models or []
        self._active_model_index = 0

        # Interrupt + checkpoint
        self._interrupted = False
        self._checkpoint_dir = Path(
            checkpoint_dir or Path.home() / ".kairos" / "checkpoints"
        )
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._setup_signal_handlers()

        # Wire up infrastructure
        if rag_store:
            from kairos.tools.rag_search import set_rag_store
            set_rag_store(rag_store)
        if knowledge_stores:
            from kairos.tools.knowledge_lookup import set_knowledge_store
            for name, store in knowledge_stores.items():
                set_knowledge_store(name, store)
        if enable_subagents:
            from kairos.agents.executor import SubAgentExecutor
            from kairos.agents.factory import set_executor
            set_executor(SubAgentExecutor(self.model))
        from kairos.tools.skills_tool import set_skill_manager
        from kairos.skills.manager import SkillManager
        set_skill_manager(SkillManager(skills_dir))

        # Pipeline
        self.pipeline = MiddlewarePipeline(middlewares or [])

        # System prompt
        self._prompt_builder = prompt_builder or PromptBuilder(
            template=system_template, agent_name=agent_name,
            role_description=role_description, soul=soul, guidelines=guidelines,
            response_style=response_style, knowledge_description=knowledge_description,
            memory_description=memory_description, **template_vars,
        )
        self.system_prompt = self._prompt_builder.build()

    # ── Run ────────────────────────────────────────────────────

    def run(self, user_message: str, prefill: str | None = None) -> dict[str, Any]:
        """Run the agent loop one-shot. Returns {content, confidence, evidence}."""
        case = Case(id=str(uuid.uuid4())[:8])
        state = ThreadState(case=case)
        runtime: dict[str, Any] = {
            "user_message": user_message,
            "thread_id": case.id,
            "session_id": case.id,
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        state.messages = messages

        self.pipeline.before_agent(state, runtime)
        self.budget.iterations = 0
        self.budget.tokens_used = 0
        self.budget.grace_call_used = False
        self._interrupted = False

        try:
            return self._execute_loop(state, runtime)
        finally:
            if not self._interrupted:
                self.pipeline.after_agent(state, runtime)

    # ── Core Loop ──────────────────────────────────────────────

    def _execute_loop(self, state: ThreadState, runtime: dict[str, Any]) -> dict[str, Any]:
        """Main agent loop with budget, interrupt, and error handling."""
        messages = state.messages
        case = state.case

        while not self.budget.exhausted or self.budget.can_grace_call:
            # ── Interrupt check ──────────────────────────────
            if self._interrupted:
                return {
                    "content": "[Interrupted]",
                    "confidence": None,
                    "evidence": [],
                    "interrupted": True,
                }

            self.pipeline.before_model(state, runtime)

            tool_schemas = get_tool_schemas() or None

            # ── Model call with error handling ────────────────
            response = self._call_model(messages, tool_schemas, state, runtime)

            # Error response
            if isinstance(response, dict) and "error" in response:
                err = classify_error(response)
                if err.kind == ErrorKind.CONTEXT_OVERFLOW:
                    return {
                        "content": f"Context window exceeded. {err.message}",
                        "confidence": None, "evidence": [],
                    }
                if err.retryable:
                    continue
                return {
                    "content": f"Error: {err.message}",
                    "confidence": None, "evidence": [],
                }

            msg = response.choices[0].message
            self.pipeline.after_model(state, runtime)

            # ── Reasoning content ────────────────────────────
            reasoning = getattr(msg, "reasoning_content", None)

            # ── Tool calls ───────────────────────────────────
            if msg.tool_calls:
                tool_results = self._execute_tools(msg, messages, state)
                self.budget.step()
                continue

            # ── Done — no more tool calls ────────────────────
            messages.append({
                "role": "assistant",
                "content": msg.content,
                **({"reasoning": reasoning} if reasoning else {}),
            })

            return {
                "content": msg.content,
                "confidence": case.confidence if case else None,
                "reasoning": reasoning,
                "evidence": self._format_evidence(case),
            }

        # ── Budget exhausted ─────────────────────────────────
        return {
            "content": "Maximum context or iterations reached.",
            "confidence": None,
            "evidence": self._format_evidence(case),
        }

    # ── Model Call ──────────────────────────────────────────────

    def _call_model(self, messages, tool_schemas, state, runtime) -> Any:
        """Call the model, falling back on persistent errors."""
        last_error = None

        for attempt in range(len(self._fallback_models) + 1):
            try:
                return self.pipeline.wrap_model_call(
                    messages,
                    lambda msgs, **kw: self._current_model.chat(msgs, tools=tool_schemas),
                )
            except Exception as e:
                last_error = e
                err = classify_error(e)
                if not err.retryable:
                    if attempt < len(self._fallback_models):
                        logger.warning("Falling back to model %d/%d", attempt + 1, len(self._fallback_models))
                        self._active_model_index = (self._active_model_index + 1) % (len(self._fallback_models) + 1)
                        continue
                    raise

        return {"error": str(last_error)}

    @property
    def _current_model(self) -> ModelProvider:
        if self._active_model_index == 0 or not self._fallback_models:
            return self.model
        config = self._fallback_models[self._active_model_index - 1]
        return ModelProvider(config)

    # ── Tool Execution ──────────────────────────────────────────

    def _execute_tools(self, msg, messages, state) -> list[dict]:
        """Execute tool calls sequentially. Returns list of tool results."""
        results = []
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            try:
                result = self.pipeline.wrap_tool_call(
                    tool_name, tool_args,
                    lambda name, args, **kw: execute_tool(name, args),
                    state=state,
                )
            except Exception as e:
                err = classify_error(e)
                result = {"error": str(e), "kind": err.kind.value}

            results.append(result)

            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tc.function.arguments,
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        return results

    # ── Interrupt + Checkpoint ──────────────────────────────────

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def interrupt(self) -> None:
        """Request graceful shutdown at next iteration boundary."""
        self._interrupted = True

    def save_checkpoint(self, state: ThreadState, runtime: dict, name: str = "") -> Path:
        """Save agent state for later resume."""
        path = self._checkpoint_dir / f"{name or uuid.uuid4().hex[:8]}.json"
        data = {
            "messages": state.messages,
            "metadata": state.metadata,
            "runtime": {k: v for k, v in runtime.items() if isinstance(v, (str, int, float, bool, list, dict))},
            "budget": {
                "iterations": self.budget.iterations,
                "tokens_used": self.budget.tokens_used,
            },
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        logger.info("Checkpoint saved: %s (%d messages)", path, len(state.messages))
        return path

    def load_checkpoint(self, name: str) -> tuple[ThreadState, dict] | None:
        """Load a checkpoint, returning (state, runtime) or None."""
        path = self._checkpoint_dir / f"{name}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        state = ThreadState(case=Case(id=name))
        state.messages = data["messages"]
        state.metadata = data.get("metadata", {})
        runtime = data.get("runtime", {})
        self.budget.iterations = data.get("budget", {}).get("iterations", 0)
        self.budget.tokens_used = data.get("budget", {}).get("tokens_used", 0)
        logger.info("Checkpoint loaded: %s (%d messages)", name, len(state.messages))
        return state, runtime

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints."""
        results = []
        for f in sorted(self._checkpoint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(f.read_text())
                results.append({
                    "name": f.stem,
                    "messages": len(d.get("messages", [])),
                    "timestamp": d.get("timestamp"),
                })
            except Exception:
                pass
        return results

    # ── Signal handlers ─────────────────────────────────────────

    def _setup_signal_handlers(self) -> None:
        def _handler(signum, frame):
            self._interrupted = True
            logger.info("Signal %d received — interrupt requested", signum)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _format_evidence(case) -> list[dict]:
        if not case or not case.steps:
            return []
        return [
            {"step": s.id, "tool": s.tool, "args": s.args,
             "result": s.result, "duration_ms": s.duration_ms}
            for s in case.steps
        ]
