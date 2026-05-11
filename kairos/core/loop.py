"""Agent Loop — the core engine with budget control, interrupt, checkpoint,
credential rotation, model fallback, and error handling.

Handles:
  - Token + iteration budget with grace calls
  - Interrupt/resume via signals + checkpoint persistence
  - Error classification (rate_limit / auth / network / tool / context_overflow)
  - Reasoning content extraction
  - Parallel tool execution (sequential for now, parallel-ready)
  - Model fallback chain with health tracking
  - Credential pool rotation (acquire/release/mark_rate_limited)
  - Provider factory (OpenAI-compatible / Anthropic native / Gemini native)
  - Trajectory saving (JSONL)
  - Prefill support
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from kairos.core.middleware import Middleware, MiddlewarePipeline
from kairos.core.state import Case, ThreadState
from kairos.core.tracing import TraceContext, TraceRecorder, set_current_trace
from kairos.prompt.template import PromptBuilder
from kairos.providers.base import ModelConfig, ModelProvider
from kairos.tools.registry import execute_tool, get_tool_schemas

logger = logging.getLogger("kairos.agent")


# ============================================================================
# Budget
# ============================================================================


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
        return self.exhausted and not self.grace_call_used

    def consume(self, tokens: int) -> None:
        self.tokens_used += tokens

    def step(self) -> bool:
        self.iterations += 1
        return self.iterations <= self.max_iterations


# ============================================================================
# Error Classification
# ============================================================================


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
        status = getattr(exc, "status_code", 0) or getattr(exc, "status", 0)

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


# ============================================================================
# Model Health Tracker (for fallback decisions)
# ============================================================================


@dataclass
class ModelHealth:
    """Tracks health of a model/provider for fallback decisions."""

    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    last_failure_time: float = 0.0
    cooldown_until: float = 0.0

    @property
    def is_healthy(self) -> bool:
        if self.consecutive_failures >= 3:
            return False
        if self.cooldown_until and time.time() < self.cooldown_until:
            return False
        return True

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_calls += 1

    def record_failure(self, kind: ErrorKind) -> None:
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_calls += 1
        self.last_failure_time = time.time()
        if kind == ErrorKind.RATE_LIMIT:
            self.cooldown_until = time.time() + 30


# ============================================================================
# Provider Factory
# ============================================================================


class ProviderFactory:
    """Creates the right provider based on ModelConfig metadata.

    Detection logic:
      - config.provider == "anthropic" or base_url contains "anthropic" -> AnthropicProvider
      - config.provider == "gemini" or base_url contains "googleapis" -> GeminiProvider
      - Otherwise -> ModelProvider (OpenAI-compatible)
    """

    @staticmethod
    def create(config: ModelConfig):
        """Instantiate the correct provider for a ModelConfig."""
        provider_name = getattr(config, "provider", "").lower()
        base_url = getattr(config, "base_url", "").lower()

        if provider_name == "anthropic" or "anthropic" in base_url:
            from kairos.providers.anthropic_adapter import AnthropicProvider
            return AnthropicProvider(config)

        if provider_name == "gemini" or "googleapis" in base_url:
            from kairos.providers.gemini_adapter import GeminiProvider
            return GeminiProvider(config)

        return ModelProvider(config)


# ============================================================================
# Agent
# ============================================================================


class Agent:
    """Core Kairos agent with budget control, interrupt, checkpoint,
    credential rotation, and model fallback.

    Usage:
        agent = Agent.build_default(model=ModelConfig(api_key="..."))
        result = agent.run("Hello")
    """

    # ---- Factory ----------------------------------------------------------

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
        trajectory_dir: str | None = None,
        enable_security: bool = False,
        security_allowed_paths: list[str] | None = None,
        enable_insights: bool = False,
        **template_vars,
    ) -> Agent:
        """Build an Agent with full 17-layer pipeline."""
        from kairos.middleware import (
            ThreadDataMiddleware,
            UploadsMiddleware,
            DanglingToolCallMiddleware,
            SkillLoader,
            ContextCompressor,
            TodoMiddleware,
            ConfidenceScorer,
            EvidenceTracker,
            SubagentLimitMiddleware,
            TitleMiddleware,
            ClarificationMiddleware,
            ViewImageMiddleware,
            MemoryMiddleware,
            LLMRetryMiddleware,
            ToolArgRepairMiddleware,
            SecurityMiddleware,
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
            layers.append(MemoryMiddleware(backend=memory_store))
        if supports_vision:
            layers.append(ViewImageMiddleware(supports_vision=True))
        layers.extend([EvidenceTracker(), ToolArgRepairMiddleware()])
        if enable_security:
            layers.append(SecurityMiddleware(
                allowed_paths=security_allowed_paths,
                block_dangerous_files=True,
                block_internal_urls=True,
            ))
        layers.append(ConfidenceScorer())
        if credential_pool:
            layers.append(
                LLMRetryMiddleware(
                    credential_pool=credential_pool,
                    retry_config=retry_config or RetryConfig(),
                )
            )
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
            trajectory_dir=trajectory_dir,
            enable_insights=enable_insights,
            **template_vars,
        )

    # ---- Constructor ------------------------------------------------------

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
        trajectory_dir: str | None = None,
        credential_pool: Any = None,
        enable_insights: bool = False,
        **template_vars,
    ):
        # ---- Model chain ----
        self._primary_config = model or ModelConfig(api_key="")
        self._primary_provider = ProviderFactory.create(self._primary_config)
        self._fallback_configs = fallback_models or []
        self._fallback_providers = [ProviderFactory.create(c) for c in self._fallback_configs]
        self._provider_chain = [self._primary_provider] + self._fallback_providers
        self._active_provider_index = 0

        # ---- Model health ----
        self._health: dict[int, ModelHealth] = {
            i: ModelHealth() for i in range(len(self._provider_chain))
        }

        # ---- Credential pool ----
        self._credential_pool = credential_pool
        self._active_credential = None  # currently held credential

        # ---- Insights ----
        self._insights = None
        if enable_insights:
            from kairos.observability.insights import AgentInsights
            self._insights = AgentInsights()

        # ---- Budget ----
        self.budget = Budget(max_iterations=max_iterations, max_tokens=max_tokens)

        # ---- Interrupt + checkpoint ----
        self._interrupted = False
        self._interrupt_lock = threading.Lock()
        self._checkpoint_dir = Path(
            checkpoint_dir or Path.home() / ".kairos" / "checkpoints"
        )
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._setup_signal_handlers()

        # ---- Trajectory ----
        self._trajectory_dir = Path(
            trajectory_dir or Path.home() / ".kairos" / "trajectories"
        )
        self._trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._trajectory_path: Path | None = None
        self._trajectory_events: list[dict] = []

        # ---- Trace recorder (full-chain observability) ----
        self._trace_recorder = TraceRecorder(
            output_dir=self._trajectory_dir / "traces"
        )
        self._trace_events: list[TraceEvent] = []

        # ---- Wire up infrastructure ----
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
            set_executor(SubAgentExecutor(self._primary_provider))
        from kairos.tools.skills_tool import set_skill_manager
        from kairos.skills.manager import SkillManager
        set_skill_manager(SkillManager(skills_dir))

        # ---- Tool whitelist (for sub-agent toolsets isolation) ----
        self._tool_whitelist: set[str] | None = set(tools) if tools else None

        # ---- Pipeline ----
        self.pipeline = MiddlewarePipeline(middlewares or [])

        # ---- System prompt ----
        self._prompt_builder = prompt_builder or PromptBuilder(
            template=system_template,
            agent_name=agent_name,
            role_description=role_description,
            soul=soul,
            guidelines=guidelines,
            response_style=response_style,
            knowledge_description=knowledge_description,
            memory_description=memory_description,
            **template_vars,
        )
        self.system_prompt = self._prompt_builder.build()

    # ---- Properties -------------------------------------------------------

    @property
    def _active_provider(self):
        return self._provider_chain[self._active_provider_index]

    @property
    def model(self):
        """Backward-compatible alias for _primary_provider."""
        return self._primary_provider

    @model.setter
    def model(self, value):
        """Allow replacing the primary provider (e.g., for tests)."""
        self._primary_provider = value
        self._provider_chain[0] = value

    # ---- Run --------------------------------------------------------------

    def run(
        self,
        user_message: str,
        prefill: str | None = None,
        parent_trace: TraceContext | None = None,
    ) -> dict[str, Any]:
        """Run the agent loop one-shot. Returns {content, confidence, evidence, trace_context}.

        Args:
            user_message: The user's input message.
            prefill: Optional assistant prefill for steering.
            parent_trace: Trace context from a parent agent for sub-agent call chains.
                       If None, a new root trace is created.
        """
        import time as _time
        _t0 = _time.time()

        from kairos.observability.metrics import get_metrics
        metrics = get_metrics()

        case = Case(id=str(uuid.uuid4())[:8])
        state = ThreadState(case=case)

        # ---- Trace context ----
        if parent_trace:
            trace_ctx = parent_trace.child()
        else:
            trace_ctx = TraceContext.new_root()
        state.trace_context = trace_ctx
        set_current_trace(trace_ctx)  # For implicit propagation to sub-agents

        runtime: dict[str, Any] = {
            "user_message": user_message,
            "thread_id": case.id,
            "session_id": case.id,
            "trace_context": trace_ctx,
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
        self._active_provider_index = 0

        # Start trajectory + trace span
        self._start_trajectory(user_message)
        self._trace_events = []
        self._log_trace_event(trace_ctx, "span_start", {
            "model": getattr(self._primary_config, "model", "unknown"),
            "depth": trace_ctx.depth,
        })

        try:
            result = self._execute_loop(state, runtime)
            result["trace_context"] = trace_ctx
            self._log_trajectory_event("agent_done", {
                "content": result.get("content", "")[:200],
                "confidence": result.get("confidence"),
            })
            self._log_trace_event(trace_ctx, "span_end", {
                "status": "success",
                "iterations": self.budget.iterations,
                "tokens_used": self.budget.tokens_used,
            })
            return result
        except Exception as e:
            self._log_trace_event(trace_ctx, "span_end", {
                "status": "error",
                "error": str(e)[:200],
                "iterations": self.budget.iterations,
            })
            raise
        finally:
            if not self._interrupted:
                self.pipeline.after_agent(state, runtime)
            self._release_credential()
            self._finish_trajectory()
            # Flush trace events
            if self._trace_events:
                self._trace_recorder.flush_span(trace_ctx, self._trace_events)

    # ---- Core Loop --------------------------------------------------------

    def _execute_loop(self, state: ThreadState, runtime: dict[str, Any]) -> dict[str, Any]:
        """Main agent loop with budget, interrupt, credential rotation, and model fallback."""
        messages = state.messages
        case = state.case

        while not self.budget.exhausted or self.budget.can_grace_call:
            # ---- Interrupt check ----
            if self._interrupted:
                self._log_trajectory_event("interrupted", {})
                return {
                    "content": "[Interrupted]",
                    "confidence": None,
                    "evidence": [],
                    "interrupted": True,
                }

            self.pipeline.before_model(state, runtime)

            tool_schemas_raw = get_tool_schemas() or []
            if self._tool_whitelist is not None:
                tool_schemas = [
                    s for s in tool_schemas_raw
                    if s["function"]["name"] in self._tool_whitelist
                ] or None
            else:
                tool_schemas = tool_schemas_raw or None

            # ---- Credential acquire ----
            self._acquire_credential()

            # ---- Model call with error handling and fallback ----
            response = self._call_model_with_fallback(messages, tool_schemas, state, runtime)

            # Error response
            if isinstance(response, dict) and "error" in response:
                err = classify_error(response)
                self._log_trajectory_event("error", {
                    "kind": err.kind.value,
                    "message": err.message[:200],
                })
                if err.kind == ErrorKind.CONTEXT_OVERFLOW:
                    return {
                        "content": f"Context window exceeded. {err.message}",
                        "confidence": None,
                        "evidence": [],
                    }
                if err.retryable:
                    continue
                return {
                    "content": f"Error: {err.message}",
                    "confidence": None,
                    "evidence": [],
                }

            # ---- Extract usage ----
            self._extract_usage(response)
            self.pipeline.after_model(state, runtime)

            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None)

            # ---- Tool calls ----
            if msg.tool_calls:
                tool_results = self._execute_tools(msg, messages, state)
                self.budget.step()
                continue

            # ---- Done ----
            assistant_msg = {
                "role": "assistant",
                "content": msg.content,
            }
            if reasoning:
                assistant_msg["reasoning"] = reasoning
            messages.append(assistant_msg)

            return {
                "content": msg.content,
                "confidence": case.confidence if case else None,
                "reasoning": reasoning,
                "evidence": self._format_evidence(case),
            }

        # ---- Budget exhausted ----
        self._log_trace_event(
            runtime.get("trace_context") or getattr(state, "trace_context", None),
            "budget_exhausted",
            {"iterations": self.budget.iterations},
        )
        return {
            "content": "Maximum context or iterations reached.",
            "confidence": None,
            "evidence": self._format_evidence(case),
        }

    # ---- Model Call with Fallback -----------------------------------------

    def _call_model_with_fallback(
        self, messages, tool_schemas, state, runtime
    ) -> Any:
        """Call the model with credential rotation and provider fallback.

        Tries each provider in the chain until one succeeds. Rotates credentials
        on rate limits. Tracks health for future routing decisions.
        """
        last_error = None
        providers_tried = 0

        while providers_tried < len(self._provider_chain):
            provider_idx = self._active_provider_index
            health = self._health[provider_idx]

            if not health.is_healthy:
                self._switch_provider()
                providers_tried += 1
                continue

            try:
                result = self.pipeline.wrap_model_call(
                    messages,
                    lambda msgs, **kw: self._active_provider.chat(
                        msgs, tools=tool_schemas
                    ),
                )
                health.record_success()
                if self._active_credential:
                    self._release_credential(success=True)
                # Auto-track usage
                self._track_call_to_insights(result, success=True)
                return result
            except Exception as e:
                err = classify_error(e)
                health.record_failure(err.kind)
                last_error = e

                if err.kind == ErrorKind.RATE_LIMIT:
                    # Try rotating credential within the same provider
                    if self._credential_pool and self._active_credential:
                        self._credential_pool.mark_rate_limited(
                            self._active_credential, retry_after=30
                        )
                        self._active_credential = None
                        self._acquire_credential()
                        if self._active_credential:
                            providers_tried -= 1  # retry same provider with new key
                            continue

                if not err.retryable:
                    self._switch_provider()
                    providers_tried += 1
                    continue

                # Retryable, switch provider
                self._switch_provider()
                providers_tried += 1

        return {"error": str(last_error)}

    def _switch_provider(self) -> None:
        """Move to the next provider in the chain."""
        self._release_credential()
        self._active_provider_index = (self._active_provider_index + 1) % len(self._provider_chain)
        logger.warning(
            "Switched to provider %d/%d",
            self._active_provider_index + 1,
            len(self._provider_chain),
        )

    # ---- Credential Management --------------------------------------------

    def _acquire_credential(self) -> None:
        """Acquire a credential for the active provider if pool is configured."""
        if not self._credential_pool:
            return
        self._active_credential = self._credential_pool.acquire("default")

    def _release_credential(self, success: bool = True) -> None:
        """Release the currently held credential."""
        if self._credential_pool and self._active_credential:
            self._credential_pool.release(self._active_credential, success=success)
            self._active_credential = None

    # ---- Usage Extraction -------------------------------------------------

    def _extract_usage(self, response: Any) -> None:
        """Extract token usage from API response and update budget."""
        try:
            usage = getattr(response, "usage", None)
            if usage and hasattr(usage, "total_tokens"):
                self.budget.consume(usage.total_tokens)
            elif isinstance(usage, dict) and "total_tokens" in usage:
                self.budget.consume(usage["total_tokens"])
        except Exception:
            pass

    def _track_call_to_insights(self, response: Any, success: bool = True) -> None:
        """Record API call metrics to AgentInsights if enabled."""
        if not self._insights:
            return
        try:
            usage = {}
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                }
            provider = getattr(self._primary_config, "provider", "unknown")
            model = getattr(self._primary_config, "model", "unknown")
            self._insights.record_call(
                provider=provider,
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                duration_ms=0,
                success=success,
            )
        except Exception:
            pass

    # ---- Tool Execution ---------------------------------------------------

    def _execute_tools(self, msg, messages, state) -> list[dict]:
        """Execute tool calls with smart parallel dispatch.

        Uses execute_tools_smart() — parallelizes read-only and path-scoped
        tools automatically, falls back to serial for write/interactive tools.
        """
        from kairos.tools.registry import execute_tools_smart

        results = []
        trace_ctx = getattr(state, "trace_context", None)

        # Normalize tool calls from OpenAI format
        tool_calls = []
        for tc in msg.tool_calls:
            tool_calls.append({
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })

        # Smart dispatch: parallel when safe, serial otherwise
        smart_results = execute_tools_smart(tool_calls)

        for i, tc in enumerate(msg.tool_calls):
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            self._log_trajectory_event("tool_start", {
                "tool": tool_name,
                "args": {k: str(v)[:100] for k, v in tool_args.items()},
            })
            if trace_ctx:
                self._log_trace_event(trace_ctx, "tool_start", {
                    "tool": tool_name,
                })

            # Wrap through pipeline (permission check, audit, etc.)
            try:
                result = self.pipeline.wrap_tool_call(
                    tool_name,
                    tool_args,
                    lambda name, args, **kw: smart_results[i],
                    state=state,
                )
                self._log_trajectory_event("tool_done", {
                    "tool": tool_name,
                    "success": True,
                })
                if trace_ctx:
                    self._log_trace_event(trace_ctx, "tool_done", {
                        "tool": tool_name,
                        "success": True,
                    })
            except Exception as e:
                err = classify_error(e)
                result = {"error": str(e), "kind": err.kind.value}
                self._log_trajectory_event("tool_error", {
                    "tool": tool_name,
                    "error": str(e)[:200],
                })
                if trace_ctx:
                    self._log_trace_event(trace_ctx, "tool_error", {
                        "tool": tool_name,
                        "error": str(e)[:200],
                    })

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

    # ---- Interrupt + Checkpoint -------------------------------------------

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def interrupt(self) -> None:
        """Request graceful shutdown at next iteration boundary."""
        with self._interrupt_lock:
            self._interrupted = True
        logger.info("Interrupt requested")

    def save_checkpoint(
        self, state: ThreadState, runtime: dict, name: str = ""
    ) -> Path:
        """Save agent state for later resume."""
        path = self._checkpoint_dir / f"{name or uuid.uuid4().hex[:8]}.json"
        data = {
            "messages": state.messages,
            "metadata": state.metadata,
            "runtime": {
                k: v
                for k, v in runtime.items()
                if isinstance(v, (str, int, float, bool, list, dict))
            },
            "budget": {
                "iterations": self.budget.iterations,
                "tokens_used": self.budget.tokens_used,
            },
            "provider_index": self._active_provider_index,
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
        self._active_provider_index = data.get("provider_index", 0)
        logger.info("Checkpoint loaded: %s (%d messages)", name, len(state.messages))
        return state, runtime

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints."""
        results = []
        for f in sorted(
            self._checkpoint_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
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

    # ---- Trace Event Logging ------------------------------------------------

    def _log_trace_event(
        self, ctx: TraceContext, event_type: str, data: dict[str, Any] | None = None
    ) -> None:
        """Record a trace event for this span."""
        event = self._trace_recorder.record(
            ctx, event_type, data, iteration=self.budget.iterations
        )
        self._trace_events.append(event)

    # ---- Trajectory -------------------------------------------------------

    def _start_trajectory(self, user_message: str) -> None:
        """Begin a new trajectory recording."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_id = getattr(self, "_session_id", "oneshot")
        self._trajectory_path = self._trajectory_dir / f"{session_id}_{ts}.jsonl"
        self._trajectory_events = []
        self._log_trajectory_event("session_start", {
            "user_message": user_message[:200],
            "model": getattr(self._primary_config, "model", "unknown"),
        })

    def _log_trajectory_event(self, event_type: str, data: dict) -> None:
        """Record a trajectory event."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "iteration": self.budget.iterations,
            **data,
        }
        self._trajectory_events.append(event)

    def _finish_trajectory(self) -> None:
        """Write trajectory to disk."""
        if not self._trajectory_path or not self._trajectory_events:
            return
        try:
            with open(self._trajectory_path, "w") as f:
                for event in self._trajectory_events:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            logger.info("Trajectory saved: %s (%d events)", self._trajectory_path, len(self._trajectory_events))
        except Exception as e:
            logger.warning("Failed to save trajectory: %s", e)

    def list_trajectories(self) -> list[dict]:
        """List saved trajectories."""
        results = []
        for f in sorted(
            self._trajectory_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                lines = f.read_text().strip().split("\n")
                first = json.loads(lines[0]) if lines else {}
                results.append({
                    "name": f.stem,
                    "events": len(lines),
                    "timestamp": first.get("timestamp"),
                    "model": first.get("model"),
                })
            except Exception:
                pass
        return results

    # ---- Health / Status --------------------------------------------------

    def health_status(self) -> dict:
        """Return health status for all providers in the chain."""
        status = {
            "active_provider": self._active_provider_index,
            "budget": {
                "iterations": self.budget.iterations,
                "tokens_used": self.budget.tokens_used,
                "remaining": self.budget.remaining,
                "exhausted": self.budget.exhausted,
            },
            "providers": [
                {
                    "index": i,
                    "config_model": getattr(
                        (self._primary_config if i == 0 else self._fallback_configs[i - 1]),
                        "model", "unknown",
                    ),
                    "consecutive_failures": h.consecutive_failures,
                    "failure_rate": round(h.failure_rate, 3),
                    "is_healthy": h.is_healthy,
                }
                for i, h in self._health.items()
            ],
        }
        # Merge insights if available
        if self._insights:
            try:
                insights_report = self._insights.get_health_report()
                status["insights"] = insights_report
            except Exception:
                pass
        return status

    # ---- Trace Queries -----------------------------------------------------

    def get_trace(self, trace_id: str) -> dict | None:
        """Retrieve the full trace tree for a trace_id."""
        return self._trace_recorder.get_span_tree(trace_id)

    def list_traces(self, limit: int = 20) -> list[dict]:
        """List recent traces with summary info."""
        results = []
        trace_dir = self._trace_recorder._output_dir
        for f in sorted(
            trace_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]:
            try:
                lines = [l for l in f.read_text().strip().split("\n") if l]
                if not lines:
                    continue
                first = json.loads(lines[0])
                last = json.loads(lines[-1])
                trace_id = first.get("trace_id", "unknown")
                results.append({
                    "trace_id": trace_id,
                    "span_id": first.get("span_id"),
                    "depth": first.get("depth", 0),
                    "events": len(lines),
                    "started_at": first.get("timestamp"),
                    "ended_at": last.get("timestamp"),
                    "status": last.get("status", "unknown"),
                })
            except Exception:
                pass
        return results

    # ---- Helpers ----------------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        def _handler(signum, frame):
            with self._interrupt_lock:
                self._interrupted = True
            logger.info("Signal %d received — interrupt requested", signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    @staticmethod
    def _format_evidence(case) -> list[dict]:
        if not case or not case.steps:
            return []
        return [
            {
                "step": s.id,
                "tool": s.tool,
                "args": s.args,
                "result": s.result,
                "duration_ms": s.duration_ms,
            }
            for s in case.steps
        ]
