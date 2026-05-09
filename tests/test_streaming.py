"""Tests for streaming: ModelProvider.chat_stream + StatefulAgent + gateway SSE."""

from unittest.mock import Mock, patch, MagicMock
import json

import pytest

from kairos.providers.base import ModelProvider, ModelConfig


class TestModelProviderStreaming:
    """Real provider-level streaming with chat_stream()."""

    @pytest.fixture
    def provider(self):
        """Create a ModelProvider with a mocked OpenAI client."""
        config = ModelConfig(api_key="test-key")
        prov = ModelProvider(config)
        prov.client = MagicMock()
        return prov

    def test_chat_stream_yields_tokens(self, provider):
        """chat_stream yields token events for text content."""
        # Mock streaming chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.tool_calls = None
        chunk1.choices[0].finish_reason = None
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].delta.tool_calls = None
        chunk2.choices[0].finish_reason = "stop"
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 10
        chunk2.usage.completion_tokens = 5
        chunk2.usage.total_tokens = 15

        provider.client.chat.completions.create.return_value = [chunk1, chunk2]

        events = list(provider.chat_stream(
            [{"role": "user", "content": "Hi"}]
        ))

        assert len(events) > 0
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["content"] == "Hello"
        assert token_events[1]["content"] == " world"

    def test_chat_stream_with_tools(self, provider):
        """chat_stream yields tool_call events when tool calls are streamed."""
        # Chunk with tool call start
        tc_start = MagicMock()
        tc_start.choices = [MagicMock()]
        tc_start.choices[0].delta = MagicMock()
        tc_start.choices[0].delta.content = None
        tc_func = MagicMock()
        tc_func.name = "read_file"
        tc_func.arguments = ""
        tc = MagicMock()
        tc.index = 0
        tc.id = "call_123"
        tc.function = tc_func
        tc_start.choices[0].delta.tool_calls = [tc]
        tc_start.choices[0].finish_reason = None
        tc_start.usage = None

        # Chunk with tool call args
        tc_arg = MagicMock()
        tc_arg.choices = [MagicMock()]
        tc_arg.choices[0].delta = MagicMock()
        tc_arg.choices[0].delta.content = None
        tc_func2 = MagicMock()
        tc_func2.name = None
        tc_func2.arguments = '{"path": "/tmp"}'
        tc2 = MagicMock()
        tc2.index = 0
        tc2.id = None
        tc2.function = tc_func2
        tc_arg.choices[0].delta.tool_calls = [tc2]
        tc_arg.choices[0].finish_reason = "tool_calls"
        tc_arg.usage = MagicMock()
        tc_arg.usage.prompt_tokens = 5
        tc_arg.usage.completion_tokens = 10
        tc_arg.usage.total_tokens = 15

        provider.client.chat.completions.create.return_value = [tc_start, tc_arg]

        events = list(provider.chat_stream(
            [{"role": "user", "content": "Read a file"}]
        ))

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) >= 1
        assert tool_call_events[0]["name"] == "read_file"

        # Final done event should have tool_calls
        done = events[-1]
        assert done["type"] == "done"
        assert done["tool_calls"] is not None
        assert done["tool_calls"][0]["name"] == "read_file"

    def test_chat_stream_empty_delta(self, provider):
        """chat_stream skips chunks with no delta."""
        # Chunk with no choices
        chunk = MagicMock()
        chunk.choices = []
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = 1
        chunk.usage.completion_tokens = 1
        chunk.usage.total_tokens = 2

        provider.client.chat.completions.create.return_value = [chunk]

        events = list(provider.chat_stream([{"role": "user", "content": "x"}]))
        # Should only have done event (no tokens)
        assert len(events) == 1
        assert events[0]["type"] == "done"


class TestStatefulAgentStreaming:
    """StatefulAgent.chat_stream() integration."""

    @pytest.fixture
    def agent(self):
        """Create a StatefulAgent with mocked provider."""
        from kairos.core.stateful_agent import StatefulAgent
        config = ModelConfig(api_key="test-key")
        agent = StatefulAgent(model=config)

        # Replace model with mock
        agent.model = MagicMock()
        return agent

    def test_chat_stream_yields_events(self, agent):
        """chat_stream yields events from the provider."""
        # Mock stream: single content chunk then done
        def mock_stream(messages, tools=None, **kwargs):
            yield {"type": "token", "content": "Hello"}
            yield {
                "type": "done",
                "content": "Hello",
                "tool_calls": None,
                "usage": {},
            }
        agent.model.chat_stream = mock_stream

        events = list(agent.chat_stream("Hi"))

        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) >= 1
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

    def test_chat_stream_interrupted(self, agent):
        """chat_stream yields error when interrupted."""
        agent._interrupted = True
        events = list(agent.chat_stream("Hi"))
        assert events[0]["type"] == "error"

    def test_chat_stream_tool_loop(self, agent):
        """chat_stream handles tool calls and loops for more content."""
        # Simulate: first stream = tool call, second stream = final content
        call_count = [0]

        def mock_stream(messages, tools=None, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                yield {"type": "tool_call", "index": 0, "name": "read_file"}
                yield {"type": "tool_delta", "index": 0, "arguments": '{"path": "/tmp"}'}
                yield {
                    "type": "done",
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "read_file", "arguments": '{"path": "/tmp"}'}],
                    "usage": {},
                }
            else:
                yield {"type": "token", "content": "Result"}
                yield {"type": "done", "content": "Result", "tool_calls": None, "usage": {}}

        agent.model.chat_stream = mock_stream

        # execute_tool is imported inside _execute_loop_stream from kairos.tools.registry
        with patch("kairos.tools.registry.execute_tool", return_value={"output": "done"}):
            events = list(agent.chat_stream("Read file"))

        # Should have token + done from final iteration
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) >= 1
