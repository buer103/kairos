"""Gemini native API provider adapter.

Wraps the google-genai SDK and exposes the same interface as
ModelProvider (OpenAI-compatible), converting between OpenAI-format
messages/tools and Gemini's native format automatically.

Key conversions:
  - Messages: OpenAI roles -> Gemini roles (system->user with prefix,
    assistant->model, tool->user with functionResponse)
  - Tools: OpenAI function.parameters -> Gemini functionDeclarations
  - Response: Gemini functionCall / functionResponse parts ->
    OpenAI tool_calls array
  - Reasoning: Gemini 2.5 thinking model -> reasoning_content
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Generator

from kairos.providers.base import ModelConfig

logger = logging.getLogger("kairos.provider")

# ---- Lazy SDK imports (allow module import without SDK installed) ----

_genai = None
_genai_types = None


def _get_genai():
    """Lazy-load google-genai. Raises ImportError if not installed."""
    global _genai, _genai_types
    if _genai is None:
        try:
            from google import genai as _g
            from google.genai import types as _t

            _genai = _g
            _genai_types = _t
        except ImportError:
            raise ImportError(
                "google-genai package is required for GeminiProvider. "
                "Install it with: pip install google-genai"
            )
    return _genai, _genai_types


# ---- OpenAI-shaped response helpers ----------------------------------


class _Choice:
    def __init__(self, message: "_Message"):
        self.message = message
        self.finish_reason = "stop"


class _Message:
    def __init__(
        self, content: str = "", tool_calls=None, reasoning_content: str | None = None
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class GeminiResponse:
    """Minimal OpenAI-compatible response wrapper."""

    def __init__(
        self,
        content: str = "",
        tool_calls: list | None = None,
        reasoning: str | None = None,
        usage: dict | None = None,
    ):
        self.choices = [_Choice(_Message(content, tool_calls, reasoning))]
        self.usage = usage or {}


# ---- Format Converters ------------------------------------------------


def _convert_tools(tool_schemas: list[dict] | None) -> list | None:
    """Convert OpenAI tool schemas to Gemini Tool objects."""
    if not tool_schemas:
        return None
    _genai, _types = _get_genai()
    declarations = []
    for ts in tool_schemas:
        func = ts.get("function", ts)
        declarations.append(
            _types.FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters=func.get("parameters"),
            )
        )
    return [_types.Tool(function_declarations=declarations)]


def _convert_messages(messages: list[dict]):
    """Convert OpenAI-format messages to Gemini Content list.
    Returns (system_instruction: str|None, contents: list[Content]).
    """
    _genai, _types = _get_genai()

    system_instruction = None
    gemini_contents: list = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            system_instruction = msg.get("content", "")
            continue

        if role == "tool":
            # Tool result -> user Content with functionResponse
            gemini_contents.append(
                _types.Content(
                    role="user",
                    parts=[
                        _types.Part(
                            function_response=_types.FunctionResponse(
                                name=msg.get("name", ""),
                                response={"output": msg.get("content", "")},
                            )
                        )
                    ],
                )
            )
            continue

        # assistant or user
        gemini_role = "model" if role == "assistant" else "user"
        parts = []

        content_text = msg.get("content", "")
        if isinstance(content_text, str) and content_text:
            parts.append(_types.Part(text=content_text))
        elif isinstance(content_text, list):
            # Anthropic-style content blocks
            for block in content_text:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(_types.Part(text=block.get("text", "")))
                    elif block.get("type") == "tool_use":
                        parts.append(
                            _types.Part(
                                function_call=_types.FunctionCall(
                                    name=block.get("name", ""),
                                    args=block.get("input", {}),
                                )
                            )
                        )

        # Tool calls from assistant (OpenAI format)
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", tc)
            try:
                args = (
                    json.loads(fn["arguments"])
                    if isinstance(fn.get("arguments"), str)
                    else fn.get("arguments", {})
                )
            except (json.JSONDecodeError, TypeError):
                args = {}
            parts.append(
                _types.Part(
                    function_call=_types.FunctionCall(
                        name=fn.get("name", ""), args=args
                    )
                )
            )

        gemini_contents.append(_types.Content(role=gemini_role, parts=parts))

    return system_instruction, gemini_contents


def _extract_response(response) -> GeminiResponse:
    """Convert Gemini GenerateContentResponse to OpenAI-compatible shape."""
    _genai, _types = _get_genai()

    content = ""
    tool_calls = []
    reasoning = None

    if not response.candidates:
        return GeminiResponse(content="")

    candidate = response.candidates[0]
    if candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            if part.text:
                content += part.text

            if hasattr(part, "thought") and part.thought:
                reasoning = str(part.thought)

            if part.function_call:
                fc = part.function_call
                tool_calls.append(
                    {
                        "id": f"call_{hash(fc.name)}_{int(time.time()*1000)}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(
                                dict(fc.args) if fc.args else {}
                            ),
                        },
                    }
                )

    usage_dict = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        usage_dict = {
            "prompt_tokens": getattr(um, "prompt_token_count", 0),
            "completion_tokens": getattr(um, "candidates_token_count", 0),
            "total_tokens": getattr(um, "total_token_count", 0),
        }

    return GeminiResponse(
        content=content,
        tool_calls=tool_calls if tool_calls else None,
        reasoning=reasoning,
        usage=usage_dict,
    )


# ---- GeminiProvider ---------------------------------------------------


class GeminiProvider:
    """Provider for Google Gemini models via native API.

    Usage:
        config = ModelConfig(
            api_key="...",
            model="gemini-2.5-pro",
            base_url="https://generativelanguage.googleapis.com"
        )
        provider = GeminiProvider(config)
        response = provider.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            _genai, _ = _get_genai()
            self._client = _genai.Client(api_key=self.config.api_key)
        return self._client

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> GeminiResponse:
        """Send a non-streaming chat completion request."""
        _genai, _types = _get_genai()
        system_instruction, contents = _convert_messages(messages)
        gemini_tools = _convert_tools(tools)

        config_params = {
            "temperature": self.config.temperature or 0.0,
            "max_output_tokens": self.config.max_tokens or 4096,
        }
        if system_instruction:
            config_params["system_instruction"] = system_instruction
        if gemini_tools:
            config_params["tools"] = gemini_tools

        generate_config = _types.GenerateContentConfig(**config_params)

        response = self.client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=generate_config,
        )
        return _extract_response(response)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream chat completion tokens from Gemini."""
        _genai, _types = _get_genai()
        system_instruction, contents = _convert_messages(messages)
        gemini_tools = _convert_tools(tools)

        config_params = {
            "temperature": self.config.temperature or 0.0,
            "max_output_tokens": self.config.max_tokens or 4096,
        }
        if system_instruction:
            config_params["system_instruction"] = system_instruction
        if gemini_tools:
            config_params["tools"] = gemini_tools

        generate_config = _types.GenerateContentConfig(**config_params)

        stream = self.client.models.generate_content_stream(
            model=self.config.model,
            contents=contents,
            config=generate_config,
        )

        content = ""
        tool_calls: dict[int, dict] = {}
        reasoning = None

        for chunk in stream:
            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        content += part.text
                        yield {"type": "token", "content": part.text}

                    if hasattr(part, "thought") and part.thought:
                        reasoning = str(part.thought)

                    if part.function_call:
                        fc = part.function_call
                        idx = hash(fc.name) % 100
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": f"call_{hash(fc.name)}",
                                "name": fc.name,
                                "arguments": "",
                            }
                            yield {
                                "type": "tool_call",
                                "index": idx,
                                "name": fc.name,
                            }
                        args_str = json.dumps(dict(fc.args) if fc.args else {})
                        tool_calls[idx]["arguments"] = args_str

        # Done event
        tc_list = (
            [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
                for tc in tool_calls.values()
            ]
            if tool_calls
            else None
        )

        yield {
            "type": "done",
            "content": content,
            "tool_calls": tc_list,
            "usage": {},
        }
