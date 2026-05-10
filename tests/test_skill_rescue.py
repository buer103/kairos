"""Tests for ContextCompressor skill rescue (DeerFlow-compatible).

Verifies that skill bundles (skill_view/skills_list calls + results)
are preserved during compression even when tokens exceed budget.
"""

from __future__ import annotations

from kairos.middleware.compress import ContextCompressor


# ============================================================================
# Helpers
# ============================================================================


def make_msg(role: str, content: str = "", **kw) -> dict:
    """Build a message dict."""
    msg: dict = {"role": role, "content": content}
    msg.update(kw)
    return msg


def make_skill_call(tool_call_id: str = "call_1") -> dict:
    """Build an AIMessage that calls skill_view."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "skill_view",
                    "arguments": '{"name": "code-review"}',
                },
            }
        ],
    }


def make_skill_result(tool_call_id: str = "call_1", content: str = "# Code Review Skill\n\nLong content...") -> dict:
    """Build a ToolMessage with skill content."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": "skill_view",
        "content": content,
    }


# ============================================================================
# Tests
# ============================================================================


class TestSkillRescueBundles:
    """_rescue_skill_bundles extracts skill messages."""

    def test_no_skill_calls_returns_empty(self):
        compressor = ContextCompressor(max_tokens=4000, keep_recent=2)
        middle = [
            make_msg("user", "hello"),
            make_msg("assistant", "hi there"),
        ]
        rescued, remaining = compressor._rescue_skill_bundles(middle)
        assert rescued == []
        assert remaining == middle

    def test_rescues_skill_view_bundle(self):
        compressor = ContextCompressor(max_tokens=4000, keep_recent=2)
        middle = [
            make_msg("user", "review my code"),
            make_skill_call("call_1"),
            make_skill_result("call_1", "# Code Review\n\nThis is skill content."),
            make_msg("assistant", "OK I'll review"),
        ]
        rescued, remaining = compressor._rescue_skill_bundles(middle)
        assert len(rescued) == 1
        assert len(rescued[0]) == 2  # AI call + tool result
        assert rescued[0][0]["role"] == "assistant"
        assert rescued[0][1]["role"] == "tool"
        # Remaining should be the user + final assistant message
        assert len(remaining) == 2
        assert remaining[0]["content"] == "review my code"

    def test_rescues_skills_list(self):
        compressor = ContextCompressor(max_tokens=4000, keep_recent=2)
        middle = [
            make_msg("user", "what skills?"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "skills_list", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "skills_list",
                "content": "Available: code-review, debugging",
            },
        ]
        rescued, remaining = compressor._rescue_skill_bundles(middle)
        assert len(rescued) == 1

    def test_multiple_skill_bundles(self):
        compressor = ContextCompressor(max_tokens=4000, keep_recent=2)
        middle = [
            make_skill_call("c1"),
            make_skill_result("c1"),
            make_msg("user", "another"),
            make_skill_call("c2"),
            make_skill_result("c2"),
        ]
        rescued, remaining = compressor._rescue_skill_bundles(middle)
        assert len(rescued) == 2

    def test_max_rescued_bundles_cap(self):
        compressor = ContextCompressor(max_tokens=4000, keep_recent=2)
        compressor._max_rescued_bundles = 1
        middle = [
            make_skill_call("c1"),
            make_skill_result("c1"),
            make_skill_call("c2"),
            make_skill_result("c2"),
        ]
        rescued, remaining = compressor._rescue_skill_bundles(middle)
        assert len(rescued) == 1
        # Second bundle should be in remaining
        assert any(
            m.get("tool_call_id") == "c2" for m in remaining
        )

    def test_non_skill_tools_not_rescued(self):
        """test_only non-skill tool calls (read_file, web_search) are not rescued."""
        compressor = ContextCompressor(max_tokens=4000, keep_recent=2)
        middle = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"/tmp/test"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_3",
                "name": "read_file",
                "content": "file content",
            },
        ]
        rescued, remaining = compressor._rescue_skill_bundles(middle)
        assert rescued == []

    def test_empty_middle(self):
        compressor = ContextCompressor(max_tokens=4000, keep_recent=2)
        rescued, remaining = compressor._rescue_skill_bundles([])
        assert rescued == []
        assert remaining == []


class TestSkillRescueIntegration:
    """Full compression with skill rescue — skill bundles survive."""

    def _build_messages_with_skill(self, filler_count: int = 20) -> list[dict]:
        """Build a message list with skill calls in the middle + filler."""
        msgs = [make_msg("system", "You are helpful")]
        # Filler messages to push us over budget
        filler = "x" * 400  # ~100 tokens per message
        for i in range(filler_count):
            msgs.append(make_msg("user", f"msg {i}: {filler}"))
            msgs.append(make_msg("assistant", f"response {i}: {filler}"))
        # Skill bundle
        msgs.append(make_skill_call("sk1"))
        msgs.append(make_skill_result("sk1", "# Important Skill\n\nCritical instructions here." * 50))
        # Recent messages
        msgs.append(make_msg("user", "do the thing"))
        msgs.append(make_msg("assistant", "using skill instructions, I will..."))
        return msgs

    def test_skill_bundle_preserved_after_compression(self):
        """After compression, skill bundle messages still exist."""
        compressor = ContextCompressor(max_tokens=4000, keep_recent=2)
        msgs = self._build_messages_with_skill(filler_count=15)

        # Verify skill messages exist before compression
        skill_msgs = [m for m in msgs if _is_skill_message(m)]
        assert len(skill_msgs) == 2  # call + result

        # Setup runtime
        runtime = {"thread_id": "test"}
        state = type("State", (), {"messages": msgs})()

        compressor.before_model(state, runtime)

        # After compression, skill messages should still be present
        skill_after = [m for m in msgs if _is_skill_message(m)]
        assert len(skill_after) == 2, f"Skill messages lost: {len(skill_after)} found"

    def test_skill_content_intact(self):
        """Skill content text is preserved (not truncated or summarized away)."""
        compressor = ContextCompressor(max_tokens=4000, keep_recent=2)
        skill_text = "# My Skill\n\nImportant: always do X first, then Y."
        msgs = [
            make_msg("system", "helper"),
        ]
        filler = "x" * 500
        for i in range(15):
            msgs.append(make_msg("user", f"msg {i}: {filler}"))
            msgs.append(make_msg("assistant", f"rsp {i}: {filler}"))
        msgs.append(make_skill_call("sk1"))
        msgs.append(make_skill_result("sk1", skill_text))
        msgs.append(make_msg("user", "execute"))
        msgs.append(make_msg("assistant", "done"))

        runtime = {"thread_id": "t1"}
        state = type("State", (), {"messages": msgs})()

        compressor.before_model(state, runtime)

        # Find skill content in messages
        found = False
        for m in msgs:
            if isinstance(m.get("content"), str) and skill_text in m["content"]:
                found = True
                break
        assert found, "Skill content not found after compression"

    def test_no_skill_compression_still_works(self):
        """Compression without skills still works normally."""
        compressor = ContextCompressor(max_tokens=1000, keep_recent=2)
        filler = "x" * 500
        msgs = [make_msg("system", "helper")]
        for i in range(10):
            msgs.append(make_msg("user", f"msg {i}: {filler}"))
            msgs.append(make_msg("assistant", f"rsp {i}: {filler}"))
        msgs.append(make_msg("user", "last"))
        msgs.append(make_msg("assistant", "done"))

        runtime = {"thread_id": "t2"}
        state = type("State", (), {"messages": msgs})()

        result = compressor.before_model(state, runtime)
        # Should compress (no error, no crash)
        assert result is not None
        assert "compressed_after" in result


class TestSkillRescueImportance:
    """Skill tool calls get max importance score."""

    def test_skill_tool_call_max_importance(self):
        compressor = ContextCompressor(max_tokens=4000, importance_scoring=True)
        block = [make_skill_call("c1"), make_skill_result("c1")]
        score = compressor._importance_score(block)
        assert score == 1.0

    def test_normal_block_not_max(self):
        compressor = ContextCompressor(max_tokens=4000, importance_scoring=True)
        block = [
            make_msg("user", "hello"),
            make_msg("assistant", "hi, what can I do?"),
        ]
        score = compressor._importance_score(block)
        assert score < 1.0


# ============================================================================
# Helper
# ============================================================================


def _is_skill_message(m: dict) -> bool:
    """Check if a message is part of a skill tool call/response."""
    if m.get("role") == "assistant":
        for tc in m.get("tool_calls", []):
            if tc.get("function", {}).get("name") in ("skill_view", "skills_list"):
                return True
    if m.get("role") == "tool" and m.get("name") in ("skill_view", "skills_list"):
        return True
    return False
