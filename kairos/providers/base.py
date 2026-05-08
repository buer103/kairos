"""Model provider abstraction — OpenAI-compatible interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
