"""Anthropic native API provider adapter.

Wraps the anthropic Python SDK and exposes the same interface as
ModelProvider (OpenAI-compatible), converting between OpenAI-format
messages/tools and Anthropic's native format automatically.

Key conversions:
  - System prompt: extracted from messages → passed as ``system`` param
  - Tool calls: OpenAI ``tool_calls`` array ↔ Anthropic ``tool_use`` / ``tool_result`` blocks
  - Tools: OpenAI ``function.parameters`` → Anthropic ``input_schema``
  - Response: Anthropic content blocks → OpenAI-shaped ``choices[0].message``
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Generator

from kairos.providers.base import ModelConfig

logger = logging.getLogger("kairos.provider")

# ── Lazy SDK imports (to allow module import without SDK installed) ─

_anthropic = None
_anthropic_types = None


def _get_anthropic():
    """Lazy-load the anthropic SDK. Raises ImportError if not installed."""
    global _anthropic, _anthropic_types
    if _anthropic is None:
        try:
            import anthropic as _ant
            from anthropic import APIStatusError, RateLimitError
            from anthropic.types import TextBlock, ToolUseBlock
            _anthropic = _ant
            _anthropic_types = (TextBlock, ToolUseBlock)
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            )
    return _anthropic

logger = logging.getLogger("kairos.provider")


# ── OpenAI-shaped response helpers ──────────────────────────────

class _Message:
    """Minimal OpenAI-compatible message wrapper."""

    __slots__ = ("content", "tool_calls", "reasoning_content")

    def __init__(
        self,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = reasoning_content


class _Choice:
    """Minimal OpenAI-compatible choice wrapper."""

    __slots__ = ("message", "finish_reason")

    def __init__(self, message: _Message, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    """Minimal OpenAI-compatible response wrapper."""

    __slots__ = ("choices", "usage")

    def __init__(
        self,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning: str | None = None,
        usage: dict[str, int] | None = None,
        finish_reason: str = "stop",
    ):
        self.choices = [_Choice(_Message(content, tool_calls, reasoning), finish_reason)]
        self.usage = usage or {}


# ── Conversion helpers ──────────────────────────────────────────

def _convert_openai_tools_to_anthropic(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert OpenAI-format tools to Anthropic format.

    OpenAI:  [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]
    Anthropic: [{"name": ..., "description": ..., "input_schema": {...}}]
    """
    if not tools:
        return None

    converted = []
    for tool in tools:
        func = tool.get("function", tool)
        converted.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


def _convert_openai_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).

    Key conversions:
      - ``system`` role → extracted as system prompt string
      - ``assistant`` with ``tool_calls`` → ``tool_use`` content blocks
      - ``tool`` role → ``tool_result`` content blocks in a user message
    """
    system_parts: list[str] = []
    anthropic_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part["text"])
                    elif isinstance(part, str):
                        system_parts.append(part)
            continue

        if role == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                # Build content blocks: text block(s) + tool_use blocks
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for tc in tool_calls:
                    func = tc.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    })
                anthropic_msgs.append({"role": "assistant", "content": blocks})
            else:
                anthropic_msgs.append({"role": "assistant", "content": str(content)})

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = msg.get("content", "")
            # Tool results go in a user message with tool_result blocks
            anthropic_msgs.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(tool_content),
                }],
            })

        elif role == "user":
            content = msg.get("content", "")
            anthropic_msgs.append({"role": "user", "content": str(content)})

        else:
            # Unknown role — treat as user
            logger.debug("Unknown message role '%s', treating as user", role)
            content = msg.get("content", "")
            anthropic_msgs.append({"role": "user", "content": str(content)})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, anthropic_msgs


def _extract_content_and_tools_from_anthropic_response(
    response: Message,
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Extract text, tool calls, and reasoning from an Anthropic response.

    Returns (text_content, tool_calls_list, reasoning_content_or_none).
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []

    for block in response.content:
        if isinstance(block, TextBlock) or (
            hasattr(block, "type") and block.type == "text"
        ):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock) or (
            hasattr(block, "type") and block.type == "tool_use"
        ):
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input, ensure_ascii=False),
                },
            })
        elif hasattr(block, "type") and block.type == "thinking":
            if hasattr(block, "thinking") and block.thinking:
                reasoning_parts.append(block.thinking)
        elif hasattr(block, "type") and block.type == "redacted_thinking":
            reasoning_parts.append("[redacted thinking]")

    text = "".join(text_parts)
    reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    return text, tool_calls, reasoning


def _retry_after_from_error(error: Exception) -> float:
    """Extract Retry-After seconds from an Anthropic API error."""
    if hasattr(error, "response") and error.response is not None:
        headers = getattr(error.response, "headers", {})
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
    return 30.0  # default cooldown


# ── Provider ────────────────────────────────────────────────────

class AnthropicProvider:
    """Model provider wrapping the Anthropic native API.

    Accepts a :class:`ModelConfig` and exposes ``chat()`` and ``chat_stream()``
    with the same signatures and return shapes as :class:`ModelProvider`.

    Parameters
    ----------
    config : ModelConfig
        Configuration.  Uses ``config.api_key``, ``config.model``,
        ``config.max_tokens``, and ``config.temperature``.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._api_key = config.api_key
        _ant = _get_anthropic()
        self._client = _ant.Anthropic(api_key=config.api_key)
        logger.info(
            "AnthropicProvider initialized model=%s max_tokens=%d",
            config.model,
            config.max_tokens,
        )

    # ── Non-streaming ────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_cache: bool = False,
        **kwargs,
    ) -> _Response:
        """Send a chat completion request. Returns an OpenAI-shaped response.

        Args:
            messages: OpenAI-format messages.
            tools: OpenAI-format tool schemas.
            enable_cache: Enable Anthropic prompt caching (~90% cost reduction
                on cached tokens). Marks system + last 2 messages as cacheable.
        """
        system, anthropic_msgs = _convert_openai_messages_to_anthropic(messages)
        anthropic_tools = _convert_openai_tools_to_anthropic(tools)

        # ── Prompt Caching ──────────────────────────────────
        if enable_cache:
            # Mark system prompt as cacheable (largest static block)
            if isinstance(system, list):
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["cache_control"] = {"type": "ephemeral"}
            elif system:
                system = [
                    {"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}},
                ]

            # Mark last 2 messages as cacheable (recent context)
            cache_count = min(2, len(anthropic_msgs))
            for i in range(len(anthropic_msgs) - cache_count, len(anthropic_msgs)):
                anthropic_msgs[i]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": anthropic_msgs,
            **kwargs,
        }
        if system:
            params["system"] = system
        if anthropic_tools:
            params["tools"] = anthropic_tools

        logger.debug("Anthropic chat: model=%s, %d messages, %d tools",
                     self.config.model, len(anthropic_msgs),
                     len(anthropic_tools) if anthropic_tools else 0)

        try:
            response = self._client.messages.create(**params)
        except RateLimitError as e:
            retry_after = _retry_after_from_error(e)
            logger.warning(
                "Anthropic rate-limited (429). Retry-After: %.1fs. "
                "Error: %s", retry_after, e,
            )
            time.sleep(retry_after)
            response = self._client.messages.create(**params)
        except APIStatusError as e:
            logger.error("Anthropic API error %d: %s",
                         e.status_code if hasattr(e, "status_code") else 0, e)
            raise

        text, tool_calls, reasoning = _extract_content_and_tools_from_anthropic_response(
            response
        )

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens or 0,
                "completion_tokens": response.usage.output_tokens or 0,
                "total_tokens": (response.usage.input_tokens or 0)
                + (response.usage.output_tokens or 0),
            }

        finish_reason = "tool_calls" if tool_calls else "stop"
        if hasattr(response, "stop_reason") and response.stop_reason:
            finish_reason = response.stop_reason

        logger.debug(
            "Anthropic response: content=%d chars, %d tool_calls, "
            "reasoning=%s, stop_reason=%s",
            len(text), len(tool_calls),
            "present" if reasoning else "none",
            finish_reason,
        )

        return _Response(
            content=text,
            tool_calls=tool_calls if tool_calls else None,
            reasoning=reasoning,
            usage=usage,
            finish_reason=finish_reason,
        )

    # ── Streaming ────────────────────────────────────────────

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream chat completion tokens from Anthropic.

        Yields dicts with ``type``: ``'token'``, ``'tool_call'``,
        ``'tool_delta'``, or ``'done'`` — matching the
        :class:`ModelProvider` stream format.
        """
        system, anthropic_msgs = _convert_openai_messages_to_anthropic(messages)
        anthropic_tools = _convert_openai_tools_to_anthropic(tools)

        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": anthropic_msgs,
            **kwargs,
        }
        if system:
            params["system"] = system
        if anthropic_tools:
            params["tools"] = anthropic_tools

        logger.debug("Anthropic chat_stream: model=%s", self.config.model)

        content = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        # Map Anthropic tool_use index → our sequential index
        _tool_idx_map: dict[int, int] = {}
        _next_tool_idx = 0
        usage: dict[str, int] = {}
        finish_reason = "stop"

        try:
            with self._client.messages.stream(**params) as stream:
                for event in stream:
                    event_type = getattr(event, "type", None)

                    # ── text delta ──────────────────────────
                    if event_type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text_delta") and delta.text_delta:
                            text = delta.text_delta.text
                            content += text
                            yield {"type": "token", "content": text}

                        elif hasattr(delta, "input_json_delta") and delta.input_json_delta:
                            # Partial JSON for tool arguments
                            json_fragment = delta.input_json_delta.partial_json
                            # Map Anthropic content block index to our index
                            block_idx = getattr(event, "index", _next_tool_idx - 1)
                            our_idx = _tool_idx_map.get(block_idx)
                            if our_idx is not None and our_idx in tool_calls:
                                tool_calls[our_idx]["arguments"] += json_fragment
                                yield {
                                    "type": "tool_delta",
                                    "index": our_idx,
                                    "arguments": json_fragment,
                                }

                    # ── content block start ──────────────────
                    elif event_type == "content_block_start":
                        block = event.content_block
                        block_type = getattr(block, "type", None)

                        if block_type == "tool_use":
                            our_idx = _next_tool_idx
                            _next_tool_idx += 1
                            block_idx = getattr(event, "index", our_idx)
                            _tool_idx_map[block_idx] = our_idx

                            tool_calls[our_idx] = {
                                "id": getattr(block, "id", ""),
                                "name": getattr(block, "name", ""),
                                "arguments": "",
                            }
                            yield {
                                "type": "tool_call",
                                "index": our_idx,
                                "name": tool_calls[our_idx]["name"],
                            }

                    # ── message delta (stop reason, usage) ──
                    elif event_type == "message_delta":
                        delta = event.delta
                        if hasattr(delta, "stop_reason") and delta.stop_reason:
                            finish_reason = delta.stop_reason
                        if hasattr(event, "usage") and event.usage:
                            usage = {
                                "prompt_tokens": event.usage.input_tokens or 0,
                                "completion_tokens": getattr(event.usage, "output_tokens", 0),
                            }
                            usage["total_tokens"] = (
                                usage["prompt_tokens"] + usage["completion_tokens"]
                            )

                    # ── message stop ─────────────────────────
                    elif event_type == "message_stop":
                        pass  # final event, handled below

        except RateLimitError as e:
            retry_after = _retry_after_from_error(e)
            logger.warning(
                "Anthropic stream rate-limited. Retry-After: %.1fs", retry_after
            )
            time.sleep(retry_after)
            # Retry once
            with self._client.messages.stream(**params) as stream:
                for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text_delta") and delta.text_delta:
                            text = delta.text_delta.text
                            content += text
                            yield {"type": "token", "content": text}
                        elif hasattr(delta, "input_json_delta") and delta.input_json_delta:
                            json_fragment = delta.input_json_delta.partial_json
                            block_idx = getattr(event, "index", _next_tool_idx - 1)
                            our_idx = _tool_idx_map.get(block_idx)
                            if our_idx is not None and our_idx in tool_calls:
                                tool_calls[our_idx]["arguments"] += json_fragment
                                yield {
                                    "type": "tool_delta",
                                    "index": our_idx,
                                    "arguments": json_fragment,
                                }
                    elif event_type == "content_block_start":
                        block = event.content_block
                        if getattr(block, "type", None) == "tool_use":
                            our_idx = _next_tool_idx
                            _next_tool_idx += 1
                            block_idx = getattr(event, "index", our_idx)
                            _tool_idx_map[block_idx] = our_idx
                            tool_calls[our_idx] = {
                                "id": getattr(block, "id", ""),
                                "name": getattr(block, "name", ""),
                                "arguments": "",
                            }
                            yield {
                                "type": "tool_call",
                                "index": our_idx,
                                "name": tool_calls[our_idx]["name"],
                            }
                    elif event_type == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            usage = {
                                "prompt_tokens": event.usage.input_tokens or 0,
                                "completion_tokens": getattr(event.usage, "output_tokens", 0),
                            }
                            usage["total_tokens"] = (
                                usage["prompt_tokens"] + usage["completion_tokens"]
                            )

        except APIStatusError as e:
            logger.error("Anthropic stream API error %d: %s",
                         e.status_code if hasattr(e, "status_code") else 0, e)
            yield {
                "type": "done",
                "content": content,
                "tool_calls": None,
                "usage": usage,
            }
            return

        # Emit final done event
        formatted_tool_calls = None
        if tool_calls:
            formatted_tool_calls = [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
                for tc in tool_calls.values()
            ]

        yield {
            "type": "done",
            "content": content,
            "tool_calls": formatted_tool_calls,
            "usage": usage,
        }
