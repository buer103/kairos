"""Tests for Kairos provider layer: CredentialPool, RetryConfig, ModelConfig,
Anthropic/Gemini adapters, ProviderFactory, and ModelHealth.

Covers: 5 modules (base, credential, anthropic_adapter, gemini_adapter, loop.ProviderFactory+ModelHealth)
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kairos.providers.base import ModelConfig, ModelProvider
from kairos.providers.credential import Credential, CredentialPool, RetryConfig


# ============================================================================
# Credential
# ============================================================================

class TestCredential:
    """Unit tests for the Credential dataclass."""

    def test_default_state(self):
        c = Credential(key="sk-test")
        assert c.key == "sk-test"
        assert c.active is True
        assert c.cooldown_until == 0.0
        assert c.consecutive_failures == 0
        assert c.total_calls == 0
        assert c.rate_limit_hits == 0
        assert c.available is True

    def test_record_success_resets_failures(self):
        c = Credential(key="sk-test", consecutive_failures=2)
        c.record_success()
        assert c.consecutive_failures == 0
        assert c.total_calls == 1

    def test_record_failure_increments(self):
        c = Credential(key="sk-test")
        c.record_failure()
        assert c.consecutive_failures == 1
        # Not yet at 3, still active
        assert c.active is True

    def test_three_consecutive_failures_disables(self):
        c = Credential(key="sk-test")
        for _ in range(3):
            c.record_failure()
        assert c.consecutive_failures == 3
        assert c.active is False

    def test_record_rate_limit_sets_cooldown(self):
        c = Credential(key="sk-test")
        c.record_rate_limit(retry_after=60)
        assert c.rate_limit_hits == 1
        assert c.cooldown_until > time.time() + 55  # roughly 60s
        assert c.available is False  # in cooldown

    def test_available_respects_active_and_cooldown(self):
        c = Credential(key="sk-test")
        assert c.available is True
        c.record_rate_limit(retry_after=0.1)
        assert c.available is False
        time.sleep(0.15)
        assert c.available is True  # cooldown expired

    def test_available_false_when_inactive(self):
        c = Credential(key="sk-test", active=False)
        assert c.available is False

    def test_label_and_provider_fields(self):
        c = Credential(key="sk-abc", provider="openai", label="production")
        assert c.provider == "openai"
        assert c.label == "production"


# ============================================================================
# CredentialPool
# ============================================================================

class TestCredentialPool:
    """Tests for the CredentialPool multi-key manager."""

    def test_add_returns_credential(self):
        pool = CredentialPool()
        cred = pool.add("sk-abc", provider="openai", label="personal")
        assert isinstance(cred, Credential)
        assert cred.key == "sk-abc"
        assert cred.provider == "openai"

    def test_add_batch(self):
        pool = CredentialPool()
        creds = pool.add_batch(["sk-a", "sk-b", "sk-c"], provider="openai")
        assert len(creds) == 3
        assert all(isinstance(c, Credential) for c in creds)

    def test_acquire_returns_best_credential(self):
        pool = CredentialPool()
        # Add two keys, one with failures
        good = pool.add("sk-good", provider="openai")
        bad = pool.add("sk-bad", provider="openai")
        bad.record_failure()
        bad.record_failure()  # 2 failures

        acquired = pool.acquire("openai")
        assert acquired is not None
        assert acquired.key == "sk-good"  # prefers fewer failures

    def test_acquire_returns_none_when_empty(self):
        pool = CredentialPool()
        assert pool.acquire("openai") is None
        assert pool.acquire("nonexistent") is None

    def test_acquire_returns_none_when_all_in_cooldown(self):
        pool = CredentialPool()
        pool.add("sk-a", provider="openai")
        pool.add("sk-b", provider="openai")
        # Put both in cooldown (acquire then mark each one)
        for _ in range(2):
            cred = pool.acquire("openai")
            pool.mark_rate_limited(cred, retry_after=60)

        assert pool.acquire("openai") is None

    def test_acquire_returns_none_when_all_disabled(self):
        pool = CredentialPool()
        pool.add("sk-a", provider="openai")
        pool.add("sk-b", provider="openai")
        # Disable each credential (acquire + disable, then the other)
        for _ in range(2):
            cred = pool.acquire("openai")
            pool.mark_disabled(cred)

        assert pool.acquire("openai") is None

    def test_release_success_updates_stats(self):
        pool = CredentialPool()
        cred = pool.add("sk-a", provider="openai")
        pool.release(cred, success=True)
        assert cred.total_calls == 1
        assert cred.consecutive_failures == 0

    def test_release_failure_updates_stats(self):
        pool = CredentialPool()
        cred = pool.add("sk-a", provider="openai")
        pool.release(cred, success=False)
        assert cred.consecutive_failures == 1

    def test_mark_rate_limited(self):
        pool = CredentialPool()
        cred = pool.add("sk-a", provider="openai")
        pool.mark_rate_limited(cred, retry_after=30)
        assert cred.rate_limit_hits == 1
        assert cred.available is False

    def test_mark_disabled(self):
        pool = CredentialPool()
        cred = pool.add("sk-a", provider="openai")
        pool.mark_disabled(cred)
        assert cred.active is False

    def test_stats_per_provider(self):
        pool = CredentialPool()
        pool.add("sk-a", provider="openai", label="key1")
        pool.add("sk-b", provider="openai", label="key2")
        pool.add("sk-c", provider="anthropic", label="key3")

        stats = pool.stats("openai")
        assert "openai" in stats
        assert stats["openai"]["total_keys"] == 2
        assert stats["openai"]["active_keys"] == 2
        assert stats["openai"]["available_keys"] == 2
        assert len(stats["openai"]["keys"]) == 2

    def test_stats_all_providers(self):
        pool = CredentialPool()
        pool.add("sk-a", provider="openai")
        pool.add("sk-b", provider="anthropic")

        stats = pool.stats()
        assert "openai" in stats
        assert "anthropic" in stats
        assert stats["openai"]["total_keys"] == 1

    def test_reset_clears_cooldowns_and_reactivates(self):
        pool = CredentialPool()
        c = pool.add("sk-a", provider="openai")
        c.active = False
        c.cooldown_until = time.time() + 3600

        pool.reset("openai")
        assert c.active is True
        assert c.cooldown_until == 0.0
        assert c.consecutive_failures == 0

    def test_reset_all_providers(self):
        pool = CredentialPool()
        c1 = pool.add("sk-a", provider="openai")
        c2 = pool.add("sk-b", provider="anthropic")
        c1.active = False
        c2.cooldown_until = time.time() + 3600

        pool.reset()  # reset all
        assert c1.active is True
        assert c2.cooldown_until == 0.0

    def test_thread_safety_concurrent_acquire(self):
        """Multiple threads acquiring should not corrupt state."""
        pool = CredentialPool()
        for i in range(5):
            pool.add(f"sk-{i}", provider="openai")

        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    cred = pool.acquire("openai")
                    if cred:
                        results.append(cred.key)
                        pool.release(cred, success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 40  # 4 threads * 10 acquires

    def test_acquire_prefers_active_over_cooldown(self):
        pool = CredentialPool()
        c1 = pool.add("sk-1", provider="openai")
        c2 = pool.add("sk-2", provider="openai")
        pool.mark_rate_limited(c1, retry_after=60)

        acquired = pool.acquire("openai")
        assert acquired is not None
        assert acquired.key == "sk-2"

    def test_acquire_sorts_by_rate_limit_hits_when_failures_equal(self):
        pool = CredentialPool()
        a = pool.add("sk-a", provider="openai")
        b = pool.add("sk-b", provider="openai")
        pool.mark_rate_limited(a, retry_after=0.01)
        time.sleep(0.02)  # a available again but has rate_limit_hits=1

        acquired = pool.acquire("openai")
        assert acquired is not None
        assert acquired.key == "sk-b"  # b has 0 rate_limit_hits


# ============================================================================
# RetryConfig
# ============================================================================

class TestRetryConfig:
    """Tests for RetryConfig delay calculation."""

    def test_delay_increases_exponentially(self):
        config = RetryConfig(base_delay=1.0, backoff_factor=2.0, jitter=False)
        d0 = config.delay_for_attempt(0)  # base * 2^0 = 1
        d1 = config.delay_for_attempt(1)  # base * 2^1 = 2
        d2 = config.delay_for_attempt(2)  # base * 2^2 = 4
        assert d0 == 1.0
        assert d1 == 2.0
        assert d2 == 4.0

    def test_delay_capped_at_max(self):
        config = RetryConfig(base_delay=1.0, max_delay=5.0, backoff_factor=3.0, jitter=False)
        d4 = config.delay_for_attempt(4)  # 1 * 3^4 = 81, capped at 5
        assert d4 == 5.0

    def test_jitter_in_range(self):
        config = RetryConfig(base_delay=1.0, max_delay=60.0, jitter=True)
        # With jitter, delay is base * (0.5 + random) * factor
        # So delay is between 0.5 * base and 1.5 * base
        delays = [config.delay_for_attempt(0) for _ in range(100)]
        for d in delays:
            assert 0.5 <= d <= 1.5, f"Delay {d} outside jitter range"

    def test_default_retryable_statuses(self):
        config = RetryConfig()
        assert 429 in config.retry_on_status
        assert 500 in config.retry_on_status
        assert 502 in config.retry_on_status
        assert 503 in config.retry_on_status
        assert 504 in config.retry_on_status


# ============================================================================
# ModelConfig
# ============================================================================

class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_defaults(self):
        config = ModelConfig(api_key="sk-test")
        assert config.api_key == "sk-test"
        assert config.base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-chat"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0
        assert config.extra_headers == {}

    def test_custom_config(self):
        config = ModelConfig(
            api_key="sk-custom",
            base_url="https://api.openai.com",
            model="gpt-4",
            max_tokens=8000,
            temperature=0.7,
            extra_headers={"X-Custom": "value"},
        )
        assert config.model == "gpt-4"
        assert config.max_tokens == 8000
        assert config.temperature == 0.7
        assert config.extra_headers == {"X-Custom": "value"}


# ============================================================================
# AnthropicProvider
# ============================================================================

class TestAnthropicProvider:
    """Tests for Anthropic native API adapter (conversion + provider)."""

    # ── Tool conversion ──────────────────────────────────────

    def test_convert_tools_openai_to_anthropic(self):
        from kairos.providers.anthropic_adapter import _convert_openai_tools_to_anthropic

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        result = _convert_openai_tools_to_anthropic(openai_tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["input_schema"] == openai_tools[0]["function"]["parameters"]

    def test_convert_tools_none(self):
        from kairos.providers.anthropic_adapter import _convert_openai_tools_to_anthropic
        assert _convert_openai_tools_to_anthropic(None) is None
        assert _convert_openai_tools_to_anthropic([]) is None

    # ── Message conversion ──────────────────────────────────

    def test_convert_messages_extracts_system(self):
        from kairos.providers.anthropic_adapter import _convert_openai_messages_to_anthropic

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, anth_msgs = _convert_openai_messages_to_anthropic(messages)
        assert system == "You are helpful."
        assert len(anth_msgs) == 1
        assert anth_msgs[0]["role"] == "user"
        assert anth_msgs[0]["content"] == "Hello"

    def test_convert_messages_assistant_with_tool_calls(self):
        from kairos.providers.anthropic_adapter import _convert_openai_messages_to_anthropic

        messages = [
            {"role": "user", "content": "Read /tmp/x.txt"},
            {
                "role": "assistant",
                "content": "Let me read that.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "/tmp/x.txt"}'},
                    }
                ],
            },
        ]
        system, anth_msgs = _convert_openai_messages_to_anthropic(messages)
        assert system is None
        assert len(anth_msgs) == 2

        # Assistant message should have content blocks
        assistant_msg = anth_msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        blocks = assistant_msg["content"]
        # First block is text, second is tool_use
        text_blocks = [b for b in blocks if b["type"] == "text"]
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Let me read that."
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "read_file"

    def test_convert_messages_tool_result(self):
        from kairos.providers.anthropic_adapter import _convert_openai_messages_to_anthropic

        messages = [
            {"role": "user", "content": "Read file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "read_file", "arguments": '{"path": "/f"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents here"},
        ]
        system, anth_msgs = _convert_openai_messages_to_anthropic(messages)
        # Third message should be user with tool_result block
        tool_msg = anth_msgs[2]
        assert tool_msg["role"] == "user"
        assert isinstance(tool_msg["content"], list)
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_1"
        assert tool_msg["content"][0]["content"] == "file contents here"

    def test_convert_messages_multiple_system(self):
        from kairos.providers.anthropic_adapter import _convert_openai_messages_to_anthropic

        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Ok"},
        ]
        system, _ = _convert_openai_messages_to_anthropic(messages)
        assert "Rule 1" in system
        assert "Rule 2" in system

    # ── Response extraction ─────────────────────────────────

    def test_extract_content_from_response(self):
        """Extract text from a mock Anthropic response with TextBlock."""
        from anthropic.types import TextBlock
        import kairos.providers.anthropic_adapter as adapter
        adapter.TextBlock = TextBlock  # inject for lazy module

        from kairos.providers.anthropic_adapter import _extract_content_and_tools_from_anthropic_response

        mock_response = MagicMock()
        text_block = TextBlock(text="Hello world", type="text")
        mock_response.content = [text_block]

        text, tool_calls, reasoning = _extract_content_and_tools_from_anthropic_response(mock_response)
        assert text == "Hello world"
        assert tool_calls == []
        assert reasoning is None

    def test_extract_tool_use_from_response(self):
        """Extract tool calls from a mock Anthropic response."""
        from anthropic.types import ToolUseBlock, TextBlock
        import kairos.providers.anthropic_adapter as adapter
        adapter.TextBlock = TextBlock
        adapter.ToolUseBlock = ToolUseBlock

        from kairos.providers.anthropic_adapter import _extract_content_and_tools_from_anthropic_response

        mock_response = MagicMock()
        tool_block = ToolUseBlock(
            id="toolu_01",
            name="read_file",
            input={"path": "/tmp/x.txt"},
            type="tool_use",
        )
        mock_response.content = [tool_block]

        text, tool_calls, reasoning = _extract_content_and_tools_from_anthropic_response(mock_response)
        assert text == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "toolu_01"
        assert tool_calls[0]["function"]["name"] == "read_file"
        parsed_args = json.loads(tool_calls[0]["function"]["arguments"])
        assert parsed_args == {"path": "/tmp/x.txt"}

    # ── Provider initialization (requires SDK) ──────────────

    def test_anthropic_provider_init(self):
        """AnthropicProvider can be instantiated with a valid API key."""
        import anthropic
        import kairos.providers.anthropic_adapter as adapter
        adapter._anthropic = anthropic  # pre-load lazy import

        from kairos.providers.anthropic_adapter import AnthropicProvider

        config = ModelConfig(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        provider = AnthropicProvider(config)
        assert provider.config.api_key == "sk-ant-test"
        assert provider.config.model == "claude-sonnet-4-20250514"

    def test_retry_after_extraction(self):
        from kairos.providers.anthropic_adapter import _retry_after_from_error

        # Error without response
        error = Exception("generic")
        assert _retry_after_from_error(error) == 30.0

        # Error with retry-after header
        mock_error = MagicMock()
        mock_error.response = MagicMock()
        mock_error.response.headers = {"retry-after": "15"}

        assert _retry_after_from_error(mock_error) == 15.0


# ============================================================================
# GeminiProvider
# ============================================================================

class TestGeminiProvider:
    """Tests for Gemini adapter (google-genai not installed — test import handling)."""

    def test_lazy_import_raises_without_sdk(self):
        """_get_genai raises ImportError when google-genai is not installed."""
        from kairos.providers.gemini_adapter import _get_genai

        # Clear any cached import
        import kairos.providers.gemini_adapter as gm
        gm._genai = None
        gm._genai_types = None

        # google-genai is not installed in this env
        with pytest.raises(ImportError, match="google-genai"):
            _get_genai()

    def test_convert_tools_openai_to_gemini_mocked(self):
        """Tool conversion works with mocked genai types."""
        # Mock _get_genai to avoid SDK requirement
        with patch("kairos.providers.gemini_adapter._get_genai") as mock_get:
            mock_types = MagicMock()
            mock_types.FunctionDeclaration = lambda name, description, parameters: {
                "name": name, "description": description, "parameters": parameters
            }
            mock_types.Tool = lambda function_declarations: function_declarations
            mock_get.return_value = (MagicMock(), mock_types)

            from kairos.providers.gemini_adapter import _convert_tools

            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
            result = _convert_tools(openai_tools)
            assert result is not None
            # result is a list of FunctionDeclaration dicts wrapped by Tool
            assert len(result) == 1

    def test_convert_tools_none(self):
        with patch("kairos.providers.gemini_adapter._get_genai") as mock_get:
            mock_get.return_value = (MagicMock(), MagicMock())
            from kairos.providers.gemini_adapter import _convert_tools
            assert _convert_tools(None) is None
            assert _convert_tools([]) is None

    def test_gemini_response_wrapper(self):
        """GeminiResponse wraps content/tool_calls/reasoning correctly."""
        from kairos.providers.gemini_adapter import GeminiResponse

        resp = GeminiResponse(content="Hello", reasoning="Let me think...",
                              usage={"prompt_tokens": 10})
        assert resp.choices[0].message.content == "Hello"
        assert resp.choices[0].message.reasoning_content == "Let me think..."
        assert resp.usage == {"prompt_tokens": 10}

    def test_gemini_response_with_tool_calls(self):
        from kairos.providers.gemini_adapter import GeminiResponse

        tc = [{"id": "c1", "function": {"name": "read_file", "arguments": '{"path":"/f"}'}}]
        resp = GeminiResponse(content="", tool_calls=tc)
        assert resp.choices[0].message.tool_calls == tc


# ============================================================================
# ProviderFactory
# ============================================================================

class TestProviderFactory:
    """Tests for ProviderFactory.create() detection logic."""

    def test_creates_anthropic_by_provider_name(self):
        import anthropic
        import kairos.providers.anthropic_adapter as adapter
        adapter._anthropic = anthropic  # pre-load lazy import

        from kairos.core.loop import ProviderFactory
        from kairos.providers.anthropic_adapter import AnthropicProvider

        config = ModelConfig(api_key="sk-ant-test", base_url="https://api.anthropic.com")
        config = MagicMock(wraps=config)
        type(config).provider = PropertyMock(return_value="anthropic")
        type(config).base_url = PropertyMock(return_value="https://api.anthropic.com")

        provider = ProviderFactory.create(config)
        assert isinstance(provider, AnthropicProvider)

    def test_creates_anthropic_by_base_url(self):
        import anthropic
        import kairos.providers.anthropic_adapter as adapter
        adapter._anthropic = anthropic  # pre-load lazy import

        from kairos.core.loop import ProviderFactory
        from kairos.providers.anthropic_adapter import AnthropicProvider

        config = ModelConfig(api_key="sk-ant-test", base_url="https://api.anthropic.com/v1")
        provider = ProviderFactory.create(config)
        assert isinstance(provider, AnthropicProvider)

    def test_creates_default_modelprovider(self):
        from kairos.core.loop import ProviderFactory

        config = ModelConfig(api_key="sk-test", base_url="https://api.deepseek.com")
        provider = ProviderFactory.create(config)
        assert isinstance(provider, ModelProvider)

    def test_creates_openai_compatible(self):
        from kairos.core.loop import ProviderFactory

        config = ModelConfig(api_key="sk-test", base_url="https://api.openai.com/v1")
        provider = ProviderFactory.create(config)
        # Should still be ModelProvider (OpenAI-compatible)
        assert isinstance(provider, ModelProvider)


# ============================================================================
# ModelHealth
# ============================================================================

class TestModelHealth:
    """Tests for model health tracking and fallback decisions."""

    def test_initial_state_healthy(self):
        from kairos.core.loop import ModelHealth

        h = ModelHealth()
        assert h.is_healthy is True
        assert h.failure_rate == 0.0
        assert h.consecutive_failures == 0

    def test_three_failures_makes_unhealthy(self):
        from kairos.core.loop import ModelHealth, ErrorKind

        h = ModelHealth()
        for _ in range(3):
            h.record_failure(ErrorKind.UNKNOWN)
        assert h.is_healthy is False

    def test_success_resets_consecutive_failures(self):
        from kairos.core.loop import ModelHealth, ErrorKind

        h = ModelHealth()
        h.record_failure(ErrorKind.NETWORK)
        h.record_failure(ErrorKind.NETWORK)
        assert h.consecutive_failures == 2
        h.record_success()
        assert h.consecutive_failures == 0
        assert h.is_healthy is True

    def test_rate_limit_sets_cooldown(self):
        from kairos.core.loop import ModelHealth, ErrorKind

        h = ModelHealth()
        h.record_failure(ErrorKind.RATE_LIMIT)
        assert h.is_healthy is False  # in cooldown

    def test_cooldown_expires(self):
        from kairos.core.loop import ModelHealth, ErrorKind

        h = ModelHealth()
        # Manually set short cooldown
        h.cooldown_until = time.time() + 0.05
        assert h.is_healthy is False
        time.sleep(0.06)
        assert h.is_healthy is True

    def test_failure_rate(self):
        from kairos.core.loop import ModelHealth, ErrorKind

        h = ModelHealth()
        h.record_success()
        h.record_success()
        h.record_failure(ErrorKind.NETWORK)
        assert h.failure_rate == 1.0 / 3.0
