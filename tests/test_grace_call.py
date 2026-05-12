"""Tests for _grace_call() — graceful tool execution with retry and arg repair."""

import pytest
from unittest.mock import MagicMock, patch

from kairos.core.loop import Agent, ErrorKind, AgentError, classify_error
from kairos.providers.base import ModelConfig
from kairos.core.state import Case, ThreadState


@pytest.fixture
def agent():
    return Agent(model=ModelConfig(api_key="test"))


@pytest.fixture
def state():
    return ThreadState(case=Case(id="test-case"))


class TestGraceCall:
    """Tests for the _grace_call method."""

    def test_success_first_attempt(self, agent, state):
        """Successful first attempt should not retry."""
        result = agent._grace_call(
            tool_name="read_file",
            tool_args={"path": "test.py"},
            result_getter=lambda: {"content": "ok"},
            state=state,
        )
        assert "content" in result
        assert result["_grace"] == {"retried": False, "attempts": 1}

    def test_retry_on_network_error(self, agent, state):
        """Network error should trigger retry."""
        call_count = [0]

        def flaky_getter():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("network timeout")
            return {"content": "recovered"}

        result = agent._grace_call(
            tool_name="web_search",
            tool_args={"query": "test"},
            result_getter=flaky_getter,
            state=state,
        )
        assert result["content"] == "recovered"
        assert result["_grace"]["retried"] is True
        assert result["_grace"]["attempts"] == 2
        assert call_count[0] == 2

    def test_non_retryable_fails_immediately(self, agent, state):
        """Non-retryable errors should fail on first attempt."""
        result = agent._grace_call(
            tool_name="read_file",
            tool_args={"path": "test.py"},
            result_getter=lambda: (_ for _ in ()).throw(
                Exception("401 Unauthorized")
            ),
            state=state,
        )
        assert "error" in result
        assert result["kind"] == "auth"
        assert result["_grace"]["retried"] is True
        assert result["_grace"]["attempts"] == 1  # non-retryable breaks immediately

    def test_all_retries_exhausted(self, agent, state):
        """After max_retries, return graceful error."""
        def always_fails():
            raise TimeoutError("connection timed out")

        result = agent._grace_call(
            tool_name="terminal",
            tool_args={"command": "ls"},
            result_getter=always_fails,
            state=state,
            max_retries=3,
        )
        assert "error" in result
        assert result["kind"] == "network"
        assert result["_grace"]["retried"] is True
        assert result["_grace"]["attempts"] == 3
        assert result["_grace"]["max_retries"] == 3
        assert "timed out" in result["_grace"]["final_error"]

    def test_grace_metadata_on_success(self, agent, state):
        """Even on success, result gets _grace metadata."""
        result = agent._grace_call(
            tool_name="read_file",
            tool_args={"path": "ok.py"},
            result_getter=lambda: {"content": "hello"},
            state=state,
        )
        assert "_grace" in result
        assert result["_grace"]["retried"] is False
        assert result["_grace"]["attempts"] == 1

    def test_rate_limit_triggers_retry(self, agent, state):
        """Rate limit errors should retry."""
        call_count = [0]

        def rate_limited_getter():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Rate limit exceeded. Try again in 10s.")
            return {"content": "ok after retry"}

        result = agent._grace_call(
            tool_name="read_file",
            tool_args={"path": "test.py"},
            result_getter=rate_limited_getter,
            state=state,
        )
        assert result["content"] == "ok after retry"
        assert result["_grace"]["retried"] is True
        assert call_count[0] == 2


class TestRepairToolArgs:
    """Tests for _repair_tool_args."""

    def test_truncate_oversized_string_arg(self, agent):
        err = AgentError(ErrorKind.TOOL_ERROR, "content too large")
        args = {"content": "x" * 15000, "path": "test.py"}
        repaired = agent._repair_tool_args("write_file", args, err, "too large")
        assert len(repaired["content"]) == 8000 + len("...[truncated by grace_call]")
        assert "path" in repaired

    def test_remove_null_values(self, agent):
        err = AgentError(ErrorKind.TOOL_ERROR, "null argument")
        args = {"path": "test.py", "extra": None, "keep": "value"}
        repaired = agent._repair_tool_args("read_file", args, err, "null")
        assert "extra" not in repaired
        assert "path" in repaired
        assert "keep" in repaired

    def test_strip_control_chars_on_json_error(self, agent):
        err = AgentError(ErrorKind.UNKNOWN, "json decode error: invalid \\x00")
        args = {"text": "hello\x00world\r\n", "path": "/tmp"}
        repaired = agent._repair_tool_args("write_file", args, err, "json decode error")
        assert "\x00" not in repaired["text"]
        assert "\r" not in repaired["text"]
        assert "hello" in repaired["text"]

    def test_remove_empty_string_args(self, agent):
        err = AgentError(ErrorKind.TOOL_ERROR, "bad arg")
        args = {"path": "/tmp", "empty": "", "valid": "ok"}
        repaired = agent._repair_tool_args("read_file", args, err, "bad")
        assert "empty" not in repaired
        assert "path" in repaired
        assert "valid" in repaired

    def test_no_change_on_network_error(self, agent):
        err = AgentError(ErrorKind.NETWORK, "timeout")
        args = {"query": "test", "limit": 10}
        repaired = agent._repair_tool_args("web_search", args, err, "timeout")
        assert repaired == args  # network errors just retry, no repair


class TestClassifyError:
    """Tests for the classify_error helper."""

    def test_rate_limit_429(self):
        err = classify_error(Exception("429 Rate limit exceeded"))
        assert err.kind == ErrorKind.RATE_LIMIT
        assert err.retryable is True

    def test_auth_401(self):
        err = classify_error(Exception("401 Unauthorized"))
        assert err.kind == ErrorKind.AUTH
        assert err.retryable is False

    def test_network_timeout(self):
        err = classify_error(TimeoutError("connection timed out"))
        assert err.kind == ErrorKind.NETWORK
        assert err.retryable is True

    def test_context_overflow(self):
        err = classify_error(Exception("maximum context length exceeded"))
        assert err.kind == ErrorKind.CONTEXT_OVERFLOW
        assert err.retryable is False

    def test_unknown_error(self):
        err = classify_error(ValueError("something weird"))
        assert err.kind == ErrorKind.UNKNOWN
        assert err.retryable is False

    def test_chinese_rate_limit(self):
        err = classify_error(Exception("请求频率超限"))
        assert err.kind in (ErrorKind.UNKNOWN, ErrorKind.RATE_LIMIT)
