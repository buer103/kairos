"""Provider registry — pre-configured profiles for popular LLM providers.

Each profile is a ModelConfig template. The user supplies their own API key;
the profile fills in base_url, default model, and provider-specific headers.

Supported profiles (all OpenAI-compatible unless noted):
  - deepseek       DeepSeek (deepseek-chat, deepseek-reasoner)
  - openrouter     OpenRouter (model routing, 200+ models)
  - groq           Groq (fast inference, llama/mixtral)
  - qwen           通义千问 / Alibaba Cloud DashScope
  - anthropic      Anthropic Claude (native SDK, not OpenAI-compat)
  - gemini         Google Gemini (native SDK, not OpenAI-compat)
  - openai         OpenAI (any OpenAI-compatible endpoint)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from kairos.providers.base import ModelConfig


@dataclass
class ProviderProfile:
    """A named provider with default configuration."""

    name: str                    # e.g. "deepseek", "openrouter"
    display_name: str            # e.g. "DeepSeek", "OpenRouter"
    base_url: str
    default_model: str
    default_max_tokens: int = 4096
    extra_headers: dict[str, str] = field(default_factory=dict)
    env_api_key: str = ""        # env var for API key (e.g. "DEEPSEEK_API_KEY")
    description: str = ""
    requires_native_sdk: bool = False  # True for Anthropic/Gemini (non-OpenAI-compat)

    def make_config(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> ModelConfig:
        """Build a ModelConfig from this profile, optionally overriding fields."""
        return ModelConfig(
            api_key=api_key or os.getenv(self.env_api_key, ""),
            base_url=self.base_url,
            model=model or self.default_model,
            max_tokens=max_tokens or self.default_max_tokens,
            extra_headers=self.extra_headers,
        )


# ============================================================================
# Built-in provider profiles
# ============================================================================

BUILTIN_PROFILES: dict[str, ProviderProfile] = {
    # ---- OpenAI-compatible providers ----

    "deepseek": ProviderProfile(
        name="deepseek",
        display_name="DeepSeek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        default_max_tokens=8192,
        env_api_key="DEEPSEEK_API_KEY",
        description="DeepSeek V3/R1 — cost-effective Chinese+English models with strong reasoning",
    ),

    "openrouter": ProviderProfile(
        name="openrouter",
        display_name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o",
        default_max_tokens=4096,
        env_api_key="OPENROUTER_API_KEY",
        extra_headers={
            "HTTP-Referer": "https://github.com/buer103/kairos",
            "X-Title": "Kairos Agent",
        },
        description="OpenRouter — unified API for 200+ models across providers",
    ),

    "groq": ProviderProfile(
        name="groq",
        display_name="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        default_max_tokens=4096,
        env_api_key="GROQ_API_KEY",
        description="Groq — ultra-fast LPU inference for Llama/Mixtral models",
    ),

    "qwen": ProviderProfile(
        name="qwen",
        display_name="通义千问 (Qwen)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        default_max_tokens=4096,
        env_api_key="DASHSCOPE_API_KEY",
        description="通义千问 — Alibaba Cloud DashScope, Qwen series models",
    ),

    "openai": ProviderProfile(
        name="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        default_max_tokens=4096,
        env_api_key="OPENAI_API_KEY",
        description="OpenAI — GPT-4o, GPT-4.1, o3, o4-mini",
    ),

    # ---- Native SDK providers (non-OpenAI-compatible) ----

    "anthropic": ProviderProfile(
        name="anthropic",
        display_name="Anthropic Claude",
        base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-20250514",
        default_max_tokens=4096,
        env_api_key="ANTHROPIC_API_KEY",
        description="Anthropic Claude — Sonnet 4, Opus 4 (native SDK, not OpenAI-compat)",
        requires_native_sdk=True,
    ),

    "gemini": ProviderProfile(
        name="gemini",
        display_name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com",
        default_model="gemini-2.5-flash",
        default_max_tokens=4096,
        env_api_key="GEMINI_API_KEY",
        description="Google Gemini — Flash/Pro models (native SDK, not OpenAI-compat)",
        requires_native_sdk=True,
    ),
}


# ============================================================================
# Provider Registry
# ============================================================================


class ProviderRegistry:
    """Central registry for LLM provider profiles.

    Usage::

        reg = ProviderRegistry()
        config = reg.get("deepseek").make_config(api_key="sk-...")
        provider = ModelProvider(config)
        # Or for native SDKs:
        provider = reg.create_provider("anthropic", api_key="sk-ant-...")
    """

    def __init__(self, profiles: dict[str, ProviderProfile] | None = None):
        self._profiles: dict[str, ProviderProfile] = dict(
            profiles or BUILTIN_PROFILES
        )

    def register(self, profile: ProviderProfile) -> None:
        """Register a new provider profile."""
        self._profiles[profile.name] = profile

    def unregister(self, name: str) -> bool:
        """Remove a provider profile. Returns False if not found."""
        return self._profiles.pop(name, None) is not None

    def get(self, name: str) -> ProviderProfile | None:
        """Get a provider profile by name."""
        return self._profiles.get(name)

    def list(self) -> list[ProviderProfile]:
        """List all registered provider profiles."""
        return sorted(self._profiles.values(), key=lambda p: p.display_name)

    def list_names(self) -> list[str]:
        """List all registered provider names."""
        return sorted(self._profiles.keys())

    def create_provider(
        self,
        name: str,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Create a ModelProvider (or native adapter) from a profile.

        For OpenAI-compatible providers, returns a ModelProvider.
        For Anthropic/Gemini, returns the native adapter.

        Args:
            name: Provider name (e.g. "deepseek", "anthropic").
            api_key: Override API key. Falls back to env var.
            model: Override model name.
            max_tokens: Override max_tokens.

        Returns:
            ModelProvider | AnthropicProvider | GeminiProvider

        Raises:
            ValueError: If the provider name is not registered.
        """
        profile = self.get(name)
        if profile is None:
            raise ValueError(
                f"Unknown provider: {name!r}. "
                f"Available: {self.list_names()}"
            )

        config = profile.make_config(api_key=api_key, model=model, max_tokens=max_tokens)

        if profile.requires_native_sdk:
            return self._create_native(profile, config)

        return self._create_openai_compat(config)

    def _create_openai_compat(self, config: ModelConfig) -> Any:
        """Create a ModelProvider for OpenAI-compatible APIs."""
        from kairos.providers.base import ModelProvider
        return ModelProvider(config)

    def _create_native(
        self, profile: ProviderProfile, config: ModelConfig
    ) -> Any:
        """Create a native SDK provider (Anthropic or Gemini)."""
        if profile.name == "anthropic":
            from kairos.providers.anthropic_adapter import AnthropicProvider
            return AnthropicProvider(config)
        elif profile.name == "gemini":
            from kairos.providers.gemini_adapter import GeminiProvider
            return GeminiProvider(config)
        else:
            raise ValueError(
                f"Native SDK not implemented for {profile.name!r}"
            )

    def make_config(
        self,
        name: str,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> ModelConfig:
        """Build a ModelConfig from a provider profile.

        Shorthand for get(name).make_config(...).
        """
        profile = self.get(name)
        if profile is None:
            raise ValueError(f"Unknown provider: {name!r}")
        return profile.make_config(api_key=api_key, model=model, max_tokens=max_tokens)

    def __len__(self) -> int:
        return len(self._profiles)

    def __contains__(self, name: str) -> bool:
        return name in self._profiles

    def __iter__(self):
        return iter(self._profiles.values())


# ============================================================================
# Convenience helpers
# ============================================================================


def list_providers() -> list[dict[str, Any]]:
    """List all built-in providers with metadata.

    Returns a list of dicts suitable for CLI display or config generation.
    """
    profiles = BUILTIN_PROFILES
    result = []
    for name, p in sorted(profiles.items()):
        result.append({
            "name": p.name,
            "display_name": p.display_name,
            "default_model": p.default_model,
            "env_api_key": p.env_api_key,
            "description": p.description,
            "requires_native_sdk": p.requires_native_sdk,
        })
    return result


def get_provider(name: str) -> ProviderProfile | None:
    """Get a built-in provider profile by name."""
    return BUILTIN_PROFILES.get(name)
