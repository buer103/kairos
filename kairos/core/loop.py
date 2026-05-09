"""Agent Loop — the core engine. Pure message → LLM → tool dispatch loop."""

from __future__ import annotations

import json
import uuid
from typing import Any

from kairos.agents.executor import SubAgentExecutor
from kairos.agents.factory import set_executor
from kairos.core.middleware import Middleware, MiddlewarePipeline
from kairos.core.state import Case, ThreadState
from kairos.prompt.template import PromptBuilder
from kairos.providers.base import ModelConfig, ModelProvider
from kairos.tools.registry import execute_tool, get_tool_schemas


class Agent:
    """The core Kairos agent.

    Usage:
        # Minimal
        agent = Agent(model=ModelConfig(api_key="..."))

        # Full pipeline (14 layers, DeerFlow-compatible)
        agent = Agent.build_default(
            model=ModelConfig(api_key="..."),
            agent_name="MyAgent",
        )
        result = agent.run("What's the schema for the users table?")
    """

    @classmethod
    def build_default(
        cls,
        model: ModelConfig,
        agent_name: str = "Kairos",
        role_description: str = "You are a helpful AI assistant.",
        max_iterations: int = 20,
        rag_store: Any = None,
        knowledge_stores: dict[str, Any] | None = None,
        skills_dir: str | None = None,
        memory_store: Any = None,
        supports_vision: bool = False,
        is_plan_mode: bool = False,
        credential_pool: Any = None,
        retry_config: Any = None,
        **template_vars,
    ) -> Agent:
        """Build an Agent with the full 14-layer DeerFlow-compatible pipeline.

        Middleware order (matching DeerFlow dependency chain + Kairos additions):
          1. ThreadData         — workspace dirs
          2. Uploads            — file injection
          3. DanglingToolCall   — fix broken tool calls
          4. SkillLoader        — load skills
          5. ContextCompressor  — token budget compression
          6. Todo               — todo persistence (if plan_mode)
          7. Memory             — persistent memory injection
          8. ViewImage          — vision model support (if supports_vision)
          9. EvidenceTracker    — record evidence steps
         10. ToolArgRepair      — repair broken JSON tool args
         11. ConfidenceScorer   — score output quality
         12. LLMRetry           — retry with credential rotation
         13. SubagentLimit      — cap concurrent sub-agents
         14. Title              — auto-generate title
         15. MemoryMiddleware   — submit to memory queue
         16. Clarification      — intercept ask_user (MUST be last)
        """
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

        layers.extend([
            EvidenceTracker(),
            ToolArgRepairMiddleware(),
            ConfidenceScorer(),
        ])

        if credential_pool:
            layers.append(LLMRetryMiddleware(
                credential_pool=credential_pool,
                retry_config=retry_config or RetryConfig(),
            ))

        layers.extend([
            SubagentLimitMiddleware(),
            TitleMiddleware(),
        ])

        if memory_store:
            layers.append(MemoryMiddleware(memory_store=memory_store))

        # Clarification MUST be last
        layers.append(ClarificationMiddleware())

        return cls(
            model=model,
            middlewares=layers,
            agent_name=agent_name,
            role_description=role_description,
            max_iterations=max_iterations,
            rag_store=rag_store,
            knowledge_stores=knowledge_stores,
            skills_dir=skills_dir,
            **template_vars,
        )

    def __init__(
        self,
        model: ModelConfig | None = None,
        tools: list[Any] | None = None,
        middlewares: list[Middleware] | None = None,
        # RAG / Knowledge infrastructure
        rag_store: Any = None,
        knowledge_stores: dict[str, Any] | None = None,
        skills_dir: str | None = None,
        evidence_db: str | None = None,
        # Sub-agent support
        enable_subagents: bool = True,
        # System prompt via PromptBuilder
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
        **template_vars,
    ):
        self.model = ModelProvider(model or ModelConfig(api_key=""))
        self.max_iterations = max_iterations

        # Wire up RAG stores
        if rag_store is not None:
            from kairos.tools.rag_search import set_rag_store
            set_rag_store(rag_store)

        if knowledge_stores:
            from kairos.tools.knowledge_lookup import set_knowledge_store
            for name, store in knowledge_stores.items():
                set_knowledge_store(name, store)

        # Wire up sub-agent executor
        if enable_subagents:
            executor = SubAgentExecutor(self.model)
            set_executor(executor)

        # Load tools (trigger @register_tool decorators)
        if tools:
            for t in tools:
                pass

        # Build middleware pipeline
        self.pipeline = MiddlewarePipeline(middlewares or [])

        # Build system prompt
        if prompt_builder:
            self._prompt_builder = prompt_builder
        else:
            self._prompt_builder = PromptBuilder(
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

    def run(self, user_message: str) -> dict[str, Any]:
        """Run the agent loop one-shot and return the result."""
        case = Case(id=str(uuid.uuid4())[:8])
        state = ThreadState(case=case)
        runtime: dict[str, Any] = {"user_message": user_message}

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        state.messages = messages

        self.pipeline.before_agent(state, runtime)
        return self._execute_loop(state, runtime)

    def _execute_loop(self, state: ThreadState, runtime: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent loop on an existing state (used by both run() and StatefulAgent)."""
        messages = state.messages
        case = state.case

        iterations = 0
        while iterations < self.max_iterations:
            self.pipeline.before_model(state, runtime)

            tool_schemas = get_tool_schemas() or None

            response = self.pipeline.wrap_model_call(
                messages,
                lambda msgs, **kw: self.model.chat(msgs, tools=tool_schemas),
            )

            # Check for error response
            if isinstance(response, dict) and "error" in response:
                return {
                    "content": f"Error: {response.get('error', 'Unknown error')}",
                    "confidence": None,
                    "evidence": [],
                }

            msg = response.choices[0].message
            self.pipeline.after_model(state, runtime)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    result = self.pipeline.wrap_tool_call(
                        tool_name,
                        tool_args,
                        lambda name, args, **kw: execute_tool(name, args),
                        state=state,
                    )

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

                iterations += 1
                continue

            # Done — no more tool calls
            messages.append({"role": "assistant", "content": msg.content})

            self.pipeline.after_agent(state, runtime)

            return {
                "content": msg.content,
                "confidence": case.confidence if case else None,
                "evidence": [
                    {
                        "step": s.id,
                        "tool": s.tool,
                        "args": s.args,
                        "result": s.result,
                        "duration_ms": s.duration_ms,
                    }
                    for s in (case.steps if case else [])
                ],
            }

        return {"content": "Max iterations reached.", "confidence": None, "evidence": []}
