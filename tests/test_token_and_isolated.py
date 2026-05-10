"""Tests for TokenUsageMiddleware + IsolatedLoop."""

from __future__ import annotations

import asyncio
import time
import pytest

from kairos.middleware.token_usage import TokenUsageMiddleware, TokenUsage


# ============================================================================
# TokenUsageMiddleware
# ============================================================================


class TestTokenUsageAttribution:
    def test_injects_attribution(self):
        mw = TokenUsageMiddleware()
        state = type("S", (), {"messages": [
            {"role": "assistant", "content": "Hello!"},
        ]})()
        mw.after_model(state, {})
        akw = state.messages[-1].get("additional_kwargs", {})
        attr = akw.get("token_usage_attribution", {})
        assert attr["kind"] == "final_answer"
        assert "turn_index" in attr

    def test_tool_call_kind(self):
        mw = TokenUsageMiddleware()
        state = type("S", (), {"messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    },
                ],
            },
        ]})()
        mw.after_model(state, {})
        attr = state.messages[-1]["additional_kwargs"]["token_usage_attribution"]
        assert attr["kind"] == "tool_batch"
        assert "read_file" in attr["tool_names"]

    def test_subagent_dispatch_kind(self):
        mw = TokenUsageMiddleware()
        state = type("S", (), {"messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "delegate_task", "arguments": "{}"},
                    },
                ],
            },
        ]})()
        mw.after_model(state, {})
        attr = state.messages[-1]["additional_kwargs"]["token_usage_attribution"]
        assert attr["kind"] == "subagent_dispatch"

    def test_search_kind(self):
        mw = TokenUsageMiddleware()
        state = type("S", (), {"messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    },
                ],
            },
        ]})()
        mw.after_model(state, {})
        attr = state.messages[-1]["additional_kwargs"]["token_usage_attribution"]
        assert attr["kind"] == "search"

    def test_todo_kind(self):
        mw = TokenUsageMiddleware()
        state = type("S", (), {"messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "todo_update", "arguments": "{}"},
                    },
                ],
            },
        ]})()
        mw.after_model(state, {})
        attr = state.messages[-1]["additional_kwargs"]["token_usage_attribution"]
        assert attr["kind"] == "todo_update"


class TestTokenUsageHistory:
    def test_accumulates_turns(self):
        mw = TokenUsageMiddleware()
        for i in range(3):
            state = type("S", (), {"messages": [
                {"role": "assistant", "content": f"msg {i}"},
            ]})()
            mw.after_model(state, {})
        assert mw.turn_count == 3

    def test_cost_calculation(self):
        mw = TokenUsageMiddleware(price_per_1k_input=0.01, price_per_1k_output=0.02)
        state = type("S", (), {"messages": [
            {
                "role": "assistant",
                "content": "ok",
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            },
        ]})()
        mw.after_model(state, {})
        # 1k input = $0.01, 0.5k output = $0.01 → $0.02
        assert mw.total_cost > 0.01

    def test_price_from_runtime(self):
        mw = TokenUsageMiddleware()
        state = type("S", (), {"messages": [
            {"role": "assistant", "content": "ok"},
        ]})()
        mw.after_model(state, {"last_usage": {"prompt_tokens": 100, "completion_tokens": 50}})
        assert mw.total_input == 100
        assert mw.total_output == 50


class TestTokenUsageEdgeCases:
    def test_non_assistant_last_message(self):
        mw = TokenUsageMiddleware()
        state = type("S", (), {"messages": [{"role": "user", "content": "hi"}]})()
        mw.after_model(state, {})
        assert mw.turn_count == 0

    def test_empty_messages(self):
        mw = TokenUsageMiddleware()
        state = type("S", (), {"messages": []})()
        mw.after_model(state, {})
        assert mw.turn_count == 0

    def test_no_tool_calls_no_content_thinking(self):
        mw = TokenUsageMiddleware()
        state = type("S", (), {"messages": [
            {"role": "assistant", "content": "", "tool_calls": []},
        ]})()
        mw.after_model(state, {})
        attr = state.messages[-1]["additional_kwargs"]["token_usage_attribution"]
        assert attr["kind"] == "thinking"

    def test_repr(self):
        mw = TokenUsageMiddleware()
        r = repr(mw)
        assert "tokens" in r


# ============================================================================
# Isolated Loop
# ============================================================================


class TestIsolatedLoop:
    def test_create_and_run_sync(self):
        from kairos.agents.isolated_loop import run_in_isolated_loop

        async def add(a, b):
            return a + b

        result = run_in_isolated_loop(add(3, 4))
        assert result == 7

    def test_multiple_calls_reuse_loop(self):
        from kairos.agents.isolated_loop import run_in_isolated_loop

        async def identity(x):
            return x

        for i in range(5):
            result = run_in_isolated_loop(identity(i))
            assert result == i

    def test_exception_propagates(self):
        from kairos.agents.isolated_loop import run_in_isolated_loop

        async def fail():
            raise ValueError("test failure")

        with pytest.raises(ValueError, match="test failure"):
            run_in_isolated_loop(fail())

    def test_timeout(self):
        from kairos.agents.isolated_loop import run_in_isolated_loop

        async def slow():
            await asyncio.sleep(10)

        with pytest.raises(TimeoutError):
            run_in_isolated_loop(slow(), timeout=0.1)

    def test_needs_isolated_loop_detection(self):
        from kairos.agents.isolated_loop import needs_isolated_loop
        # In test runner, no running loop → False
        assert not needs_isolated_loop()

    @pytest.mark.asyncio
    async def test_needs_isolated_loop_true(self):
        """Inside an async test, needs_isolated_loop should detect loop."""
        from kairos.agents.isolated_loop import needs_isolated_loop
        assert needs_isolated_loop()

    @pytest.mark.asyncio
    async def test_run_async_in_isolated(self):
        from kairos.agents.isolated_loop import run_in_isolated_loop_async

        async def compute():
            return 42

        result = await run_in_isolated_loop_async(compute())
        assert result == 42
