"""Agent Loop — the core engine. Pure message → LLM → tool dispatch loop."""

from __future__ import annotations

import json
import uuid
from typing import Any

from kairos.core.middleware import Middleware, MiddlewarePipeline
from kairos.core.state import Case, ThreadState
from kairos.providers.base import ModelConfig, ModelProvider
from kairos.tools.registry import execute_tool, get_tool_schemas


DEFAULT_SYSTEM_PROMPT = """You are {agent_name}, an AI agent built on Kairos.

{role_description}

<tools>
{tools_section}
</tools>

{memory_section}

<response>
{response_style}
</response>
"""


class Agent:
    """The core Kairos agent."""

    def __init__(
        self,
        model: ModelConfig | None = None,
        tools: list[Any] | None = None,
        middlewares: list[Middleware] | None = None,
        agent_name: str = "Kairos",
        role_description: str = "You are a helpful AI assistant.",
        soul: str = "",
        response_style: str = "Be concise and helpful.",
        memory_section: str = "",
        system_template: str | None = None,
        max_iterations: int = 20,
        **template_vars,
    ):
        self.model = ModelProvider(model or ModelConfig(api_key=""))
        self.max_iterations = max_iterations

        # Load tools (trigger registry)
        if tools:
            for t in tools:
                pass  # Tools self-register via @register_tool decorator

        # Build middleware pipeline
        self.pipeline = MiddlewarePipeline(middlewares or [])

        # Build system prompt
        tool_descs = "\n".join(
            f"- {name}: {info['schema']['function']['description']}"
            for name, info in get_tool_schemas().__dict__.items()
            if isinstance(info, dict)
        )
        template = system_template or DEFAULT_SYSTEM_PROMPT
        self.system_prompt = template.format(
            agent_name=agent_name,
            role_description=role_description + ("\n\n" + soul if soul else ""),
            tools_section=tool_descs,
            memory_section=memory_section,
            response_style=response_style,
            **template_vars,
        )

    def run(self, user_message: str) -> dict[str, Any]:
        """Run the agent loop and return the result."""
        # Initialize state
        case = Case(id=str(uuid.uuid4())[:8])
        state = ThreadState(case=case)
        runtime = {"user_message": user_message}

        # Build initial messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        state.messages = messages

        # Middleware: before_agent
        self.pipeline.before_agent(state, runtime)

        # Agent loop
        iterations = 0
        while iterations < self.max_iterations:
            # Middleware: before_model
            self.pipeline.before_model(state, runtime)

            # Get tool schemas
            tool_schemas = get_tool_schemas() or None

            # Call LLM (with middleware wrapping)
            response = self.pipeline.wrap_model_call(
                state.messages,
                lambda msgs, **kw: self.model.chat(msgs, tools=tool_schemas),
            )

            msg = response.choices[0].message

            # Middleware: after_model
            self.pipeline.after_model(state, runtime)

            # Check for tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    # Execute tool (with middleware wrapping)
                    result = self.pipeline.wrap_tool_call(
                        tool_name,
                        tool_args,
                        lambda name, args, **kw: execute_tool(name, args),
                        state=state,
                    )

                    # Append to messages
                    state.messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                        ],
                    })
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                iterations += 1
                continue

            # No tool calls — agent is done
            state.messages.append({
                "role": "assistant",
                "content": msg.content,
            })

            # Middleware: after_agent
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
