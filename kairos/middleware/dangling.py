"""Dangling tool call middleware — fixes broken tool calls in message history.

DeerFlow layer 3 — must run before any model call via wrap_model_call.

Scenarios creating dangling calls:
  1. User interrupts mid-tool-execution (SIGINT)
  2. Tool execution times out
  3. Server restart / crash recovery
  4. Malformed provider responses

This middleware normalizes message formats, detects gaps, and injects
synthetic ToolMessages with error status so the LLM receives well-formed input.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kairos.core.middleware import Middleware

logger = logging.getLogger("kairos.middleware.dangling")


class DanglingToolCallMiddleware(Middleware):
    """Fixes dangling tool calls by injecting synthetic ToolMessage placeholders.

    Hook: wrap_model_call — must insert at specific positions in the message
    list, not append at the end (which before_model would do).

    Message format normalization:
      Handles both structured dict tool_calls and raw provider payloads
      in additional_kwargs, parsing JSON string arguments when needed.
    """

    def wrap_model_call(self, messages: list[dict], handler, **kwargs) -> Any:
        patched = self._build_patched_messages(messages)
        return handler(patched, **kwargs)

    # ── Normalization ────────────────────────────────────────────

    @staticmethod
    def _normalize_tool_calls(msg: dict) -> list[dict]:
        """Extract and normalize tool calls from any message format.

        Handles:
          - msg["tool_calls"] — structured dict format
          - msg["additional_kwargs"]["tool_calls"] — raw provider payload
          - function.name + function.arguments (string JSON)
          - flat name + args (dict)
        """
        # Structured format (primary)
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            return [
                {
                    "id": tc.get("id", ""),
                    "name": DanglingToolCallMiddleware._extract_name(tc),
                    "arguments": DanglingToolCallMiddleware._extract_args(tc),
                }
                for tc in tool_calls
            ]

        # Raw provider payload (additional_kwargs)
        raw = (msg.get("additional_kwargs") or {}).get("tool_calls") or []
        if not raw:
            return []

        normalized = []
        for raw_tc in raw:
            if not isinstance(raw_tc, dict):
                continue

            tc_id = raw_tc.get("id", "")
            fn = raw_tc.get("function") or {}
            name = raw_tc.get("name") or fn.get("name", "unknown")
            args = raw_tc.get("args", {})
            if not args and isinstance(fn, dict):
                args = DanglingToolCallMiddleware._parse_json_args(
                    fn.get("arguments", "")
                )

            normalized.append({
                "id": tc_id,
                "name": name,
                "arguments": args if isinstance(args, dict) else {},
            })

        return normalized

    @staticmethod
    def _extract_name(tc: dict) -> str:
        """Extract tool name from structured or function-nested formats."""
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            return fn.get("name", "unknown")
        return tc.get("name", "unknown")

    @staticmethod
    def _extract_args(tc: dict) -> dict:
        """Extract tool arguments, parsing JSON strings if needed."""
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            raw = fn.get("arguments", {})
            if isinstance(raw, str):
                return DanglingToolCallMiddleware._parse_json_args(raw)
            return raw if isinstance(raw, dict) else {}
        args = tc.get("arguments", {})
        if isinstance(args, str):
            return DanglingToolCallMiddleware._parse_json_args(args)
        return args if isinstance(args, dict) else {}

    @staticmethod
    def _parse_json_args(raw: str) -> dict:
        """Safely parse JSON string arguments, returning {} on failure."""
        if not raw or not isinstance(raw, str):
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}

    # ── Patching ─────────────────────────────────────────────────

    def _build_patched_messages(self, messages: list[dict]) -> list[dict]:
        """Detect and fix dangling tool calls in the message history.

        Two-pass approach:
          1. Collect all existing ToolMessage tool_call_ids
          2. Build patched list, inserting synthetic ToolMessages
             immediately after each AIMessage with dangling calls.

        Returns the original list unchanged if nothing needs patching.
        """
        if not messages:
            return messages

        # Pass 1: collect existing ToolMessage IDs
        existing_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                existing_ids.add(msg["tool_call_id"])

        # Pass 2: build patched list
        patched: list[dict] = []
        patched_ids: set[str] = set()
        patch_count = 0

        for msg in messages:
            patched.append(msg)

            if msg.get("role") != "assistant":
                continue

            for tc in self._normalize_tool_calls(msg):
                tc_id = tc["id"]
                if not tc_id:
                    continue
                if tc_id in existing_ids or tc_id in patched_ids:
                    continue

                # Inject synthetic error ToolMessage
                patched.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc["name"],
                    "content": "[Tool call was interrupted and did not return a result.]",
                    "status": "error",
                })
                patched_ids.add(tc_id)
                patch_count += 1

        if patch_count > 0:
            logger.warning(
                "Injected %d placeholder ToolMessage(s) for dangling tool calls",
                patch_count,
            )

        return patched

    def __repr__(self) -> str:
        return "DanglingToolCallMiddleware()"
