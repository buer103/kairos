"""Agent Loop — the core engine. Pure message → LLM → tool dispatch loop."""

from __future__ import annotations

import json
import uuid
from typing import Any

from kairos.core.middleware import Middleware, MiddlewarePipeline
from kairos.core.state import Case, ThreadState
from kairos.prompt.template import PromptBuilder
from kairos.providers.base import ModelConfig, ModelProvider
from kairos.tools.registry import execute_tool, get_tool_schemas


class Agent:
    """The core Kairos agent."""

    def __init__(
        self,
        model: ModelConfig | None = None,
        tools: list[Any] | None = None,
        middlewares: list[Middleware] | None = None,
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

        # Load tools (trigger @register_tool decorators)
        if tools:
            for _ in tools:
                pass

        # Build middleware pipeline
        self.pipeline = MiddlewarePipeline(middlewares or [])

        # Build system prompt — use PromptBuilder if provided, else construct one
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

        # Render the prompt once at init (tools don't change mid-session)
        self.system_prompt = self._prompt_builder.build()

    def run(self, user_message: str) -> dict[str, Any]:
        """Run the agent loop and return the result."""
        case = Case(id=str(uuid.uuid4())[:8])
        state = ThreadState(case=case)
        runtime = {"user_message": user_message}

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        state.messages = messages

        self.pipeline.before_agent(state, runtime)

        iterations = 0
        while iterations < self.max_iterations:
            self.pipeline.before_model(state, runtime)

            tool_schemas = get_tool_schemas() or None

            response = self.pipeline.wrap_model_call(
                state.messages,
                lambda msgs, **kw: self.model.chat(msgs, tools=tool_schemas),
            )

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

                    state.messages.append({
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
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                iterations += 1
                continue

            # Done — no more tool calls
            state.messages.append({"role": "assistant", "content": msg.content})

            self.pipeline.after_agent(state, runtime)

            return {
                "content": msg.content,
                "confidence": case.confidence,
                "evidence": [
                    {
                        "step": s.id,
                        "tool": s.tool,
                        "args": s.args,
                        "result": s.result,
                        "duration_ms": s.duration_ms,
                    }
                    for s in case.steps
                ],
            }

        return {"content": "Max iterations reached.", "confidence": None, "evidence": []}
