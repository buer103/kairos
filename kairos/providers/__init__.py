"""Providers package — multi-provider LLM abstraction with 17 built-in profiles.

Provides:
- ``ModelConfig`` — unified model configuration (api_key, base_url, model)
- ``ModelProvider`` — OpenAI-compatible provider (covers 15 profiles)
- ``AnthropicProvider`` — native Anthropic Messages API adapter
- ``GeminiProvider`` — native Google Generative AI adapter
- ``CredentialPool`` — multi-key rotation with 429 rate-limit handling
- ``RetryConfig`` — exponential backoff configuration
- ``ProviderRegistry`` — 17 built-in profiles (deepseek, openai, anthropic, etc.)

Usage:
    from kairos.providers import ModelConfig, ModelProvider
    provider = ModelProvider(ModelConfig(api_key="...", model="gpt-4o"))
    response = provider.chat([{"role": "user", "content": "Hello"}])
"""

from kairos.providers.base import ModelConfig, ModelProvider
from kairos.providers.anthropic_adapter import AnthropicProvider
from kairos.providers.gemini_adapter import GeminiProvider
from kairos.providers.credential import CredentialPool, RetryConfig

__all__ = [
    "ModelConfig",
    "ModelProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "CredentialPool",
    "RetryConfig",
]
