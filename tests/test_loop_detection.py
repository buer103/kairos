"""Tests for LoopDetectionMiddleware — DeerFlow-compatible loop breaking."""

from __future__ import annotations

from kairos.middleware.loop_detection import LoopDetectionMiddleware


# ============================================================================
# Helpers
# ============================================================================


def make_state(messages: list[dict]) -> object:
    """Create a simple state object with messages."""
    return type("State", (), {"messages": messages})()


def make_tool_call(name: str, args: dict | None = None) -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": str(args or {}),
        },
    }


def make_assistant(tool_calls: list[dict] | None = None, content: str = "") -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


# ============================================================================
# Tests
# ============================================================================


class TestExactRepeatDetection:
    """Detect identical tool calls repeated N times."""

    def test_single_call_not_detected(self):
        mw = LoopDetectionMiddleware(max_repeats=3)
        state = make_state([make_assistant([make_tool_call("read_file", {"path": "/tmp/a"})])])
        result = mw.after_model(state, {})
        assert result is None
        assert not mw.loop_broken

    def test_two_calls_not_detected(self):
        mw = LoopDetectionMiddleware(max_repeats=3)
        tc = make_tool_call("read_file", {"path": "/tmp/a"})
        for _ in range(2):
            state = make_state([make_assistant([tc])])
            mw.after_model(state, {})
        assert not mw.loop_broken

    def test_three_identical_calls_detected(self):
        mw = LoopDetectionMiddleware(max_repeats=3)
        tc = make_tool_call("read_file", {"path": "/tmp/a"})
        for _ in range(2):
            state = make_state([make_assistant([tc])])
            mw.after_model(state, {})
        # Third time triggers
        state = make_state([make_assistant([tc])])
        result = mw.after_model(state, {})
        assert result is not None
        # Tool calls should be removed
        assert state.messages[-1].get("tool_calls") == [] or len(state.messages[-1].get("tool_calls", [])) == 0

    def test_different_args_not_detected(self):
        mw = LoopDetectionMiddleware(max_repeats=3)
        calls = [
            make_tool_call("read_file", {"path": "/tmp/a"}),
            make_tool_call("read_file", {"path": "/tmp/b"}),
            make_tool_call("read_file", {"path": "/tmp/c"}),
        ]
        for tc in calls:
            state = make_state([make_assistant([tc])])
            mw.after_model(state, {})
        assert not mw.loop_broken

    def test_different_tools_not_detected(self):
        mw = LoopDetectionMiddleware(max_repeats=3)
        calls = [
            make_tool_call("read_file"),
            make_tool_call("write_file"),
            make_tool_call("web_search"),
        ]
        for tc in calls:
            state = make_state([make_assistant([tc])])
            mw.after_model(state, {})
        assert not mw.loop_broken

    def test_break_injects_message(self):
        mw = LoopDetectionMiddleware(max_repeats=2)
        tc = make_tool_call("web_search", {"query": "x"})
        # First call
        state = make_state([make_assistant([tc])])
        mw.after_model(state, {})
        # Second call — should break
        state = make_state([make_assistant([tc])])
        mw.after_model(state, {})
        content = state.messages[-1].get("content", "")
        assert "Loop detected" in content

    def test_break_clears_tool_calls(self):
        mw = LoopDetectionMiddleware(max_repeats=2)
        tc = make_tool_call("web_search", {"query": "x"})
        # First
        mw.after_model(make_state([make_assistant([tc])]), {})
        # Second — break
        state = make_state([make_assistant([tc])])
        mw.after_model(state, {})
        assert state.messages[-1].get("tool_calls") == []


class TestSequenceRepeatDetection:
    """Detect sequence repetition within a window."""

    def test_repeating_sequence_detected(self):
        mw = LoopDetectionMiddleware(max_repeats=10, sequence_window=4)
        # Build pattern: A, B, A, B
        tc_a = make_tool_call("read_file", {"path": "/tmp/a"})
        tc_b = make_tool_call("write_file", {"path": "/tmp/b"})
        pattern = [tc_a, tc_b, tc_a, tc_b]

        for tc in pattern:
            state = make_state([make_assistant([tc])])
            mw.after_model(state, {})

        assert mw.loop_broken

    def test_non_repeating_sequence_not_detected(self):
        mw = LoopDetectionMiddleware(max_repeats=10, sequence_window=4)
        tc_a = make_tool_call("read_file")
        tc_b = make_tool_call("write_file")
        tc_c = make_tool_call("web_search")
        tc_d = make_tool_call("terminal")
        pattern = [tc_a, tc_b, tc_c, tc_d]

        for tc in pattern:
            state = make_state([make_assistant([tc])])
            mw.after_model(state, {})

        assert not mw.loop_broken


class TestLoopDetectionEdgeCases:
    """Edge case handling."""

    def test_empty_messages(self):
        mw = LoopDetectionMiddleware()
        state = make_state([])
        result = mw.after_model(state, {})
        assert result is None

    def test_last_message_not_assistant(self):
        mw = LoopDetectionMiddleware()
        state = make_state([{"role": "user", "content": "hello"}])
        result = mw.after_model(state, {})
        assert result is None

    def test_no_tool_calls_resets_history(self):
        mw = LoopDetectionMiddleware(max_repeats=2)
        tc = make_tool_call("web_search")
        # Build up one call
        mw.after_model(make_state([make_assistant([tc])]), {})
        assert len(mw._history) == 1
        # No tool calls → reset
        mw.after_model(make_state([make_assistant(content="done")]), {})
        assert len(mw._history) == 0

    def test_reset_clears_history(self):
        mw = LoopDetectionMiddleware()
        mw._history = ["a", "b", "c"]
        mw.reset()
        assert mw._history == []

    def test_multiple_tool_calls_per_turn(self):
        """Multiple tool calls in one assistant message."""
        mw = LoopDetectionMiddleware(max_repeats=9)
        tc1 = make_tool_call("read_file", {"path": "/tmp/a"})
        tc2 = make_tool_call("write_file", {"path": "/tmp/b"})
        tc3 = make_tool_call("web_search", {"query": "test"})
        # 3 calls at once
        mw.after_model(make_state([make_assistant([tc1])]), {})
        mw.after_model(make_state([make_assistant([tc2])]), {})
        mw.after_model(make_state([make_assistant([tc3])]), {})
        assert len(mw._history) == 3

    def test_json_args_normalized(self):
        """Args with different key order should match."""
        import json
        mw = LoopDetectionMiddleware(max_repeats=2)
        tc1 = {
            "id": "c1", "type": "function",
            "function": {"name": "t", "arguments": json.dumps({"b": 1, "a": 2})},
        }
        tc2 = {
            "id": "c2", "type": "function",
            "function": {"name": "t", "arguments": json.dumps({"a": 2, "b": 1})},
        }
        mw.after_model(make_state([make_assistant([tc1])]), {})
        state = make_state([make_assistant([tc2])])
        mw.after_model(state, {})
        # Should match (same args, different key order)
        assert "Loop detected" in state.messages[-1].get("content", "")

    def test_repr(self):
        mw = LoopDetectionMiddleware(max_repeats=5, sequence_window=10)
        r = repr(mw)
        assert "max_repeats=5" in r
        assert "window=10" in r
