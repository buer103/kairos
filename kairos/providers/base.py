"""Model provider abstraction — OpenAI-compatible interface with streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generator

from openai import OpenAI


@dataclass
class ModelConfig:
    """Configuration for a model provider."""

    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 4096
    temperature: float = 0.0
    extra_headers: dict[str, str] = field(default_factory=dict)


class ModelProvider:
    """Unified interface for any OpenAI-compatible LLM provider."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=config.extra_headers,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        """Send a chat completion request. Returns the OpenAI response object."""
        params = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
        return self.client.chat.completions.create(**params)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream chat completion tokens from the provider.

        Yields dicts with type: 'token', 'tool_call', 'tool_delta', or 'done'.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs,
        }
        if tools:
            params["tools"] = tools

        stream = self.client.chat.completions.create(**params)

        content = ""
        tool_calls: dict[int, dict] = {}
        usage = {}

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Text content
            if delta.content:
                content += delta.content
                yield {"type": "token", "content": delta.content}

            # Tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "arguments": "",
                        }
                        yield {"type": "tool_call", "index": idx, "name": tool_calls[idx]["name"]}

                    if tc.function and tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments
                        yield {"type": "tool_delta", "index": idx, "arguments": tc.function.arguments}

            # Finish reason
            if chunk.choices[0].finish_reason:
                pass  # Handled at stream end

            # Usage stats (usually in final chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

        # Emit final done event
        yield {
            "type": "done",
            "content": content,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
                for tc in tool_calls.values()
            ] if tool_calls else None,
            "usage": usage,
        }
