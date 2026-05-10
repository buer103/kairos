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
  - mistral        Mistral AI (Large, Small, Codestral)
  - together       Together AI (200+ open-source models)
  - perplexity     Perplexity (search-grounded LLMs)
  - cohere         Cohere (Command R/R+)
  - xai            xAI Grok (Grok-3)
  - replicate      Replicate (run OS models in cloud)
  - azure          Azure OpenAI Service
  - cloudflare     Cloudflare Workers AI
  - fireworks      Fireworks AI
  - huggingface    HuggingFace TGI
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from kairos.providers.base import ModelConfig


@dataclass
class ProviderProfile:
    """A named provider with default configuration."""

    name: str
    display_name: str
    base_url: str
    default_model: str
    default_max_tokens: int = 4096
    extra_headers: dict[str, str] = field(default_factory=dict)
    env_api_key: str = ""
    description: str = ""
    requires_native_sdk: bool = False

    def make_config(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> ModelConfig:
        return ModelConfig(
            api_key=api_key or os.getenv(self.env_api_key, ""),
            base_url=self.base_url,
            model=model or self.default_model,
            max_tokens=max_tokens or self.default_max_tokens,
            extra_headers=self.extra_headers,
        )


BUILTIN_PROFILES: dict[str, ProviderProfile] = {
    "deepseek": ProviderProfile(
        name="deepseek", display_name="DeepSeek",
        base_url="https://api.deepseek.com", default_model="deepseek-chat",
        default_max_tokens=8192, env_api_key="DEEPSEEK_API_KEY",
        description="DeepSeek V3/R1 — cost-effective Chinese+English models with strong reasoning",
    ),
    "openrouter": ProviderProfile(
        name="openrouter", display_name="OpenRouter",
        base_url="https://openrouter.ai/api/v1", default_model="openai/gpt-4o",
        env_api_key="OPENROUTER_API_KEY",
        extra_headers={"HTTP-Referer": "https://github.com/buer103/kairos", "X-Title": "Kairos Agent"},
        description="OpenRouter — unified API for 200+ models across providers",
    ),
    "groq": ProviderProfile(
        name="groq", display_name="Groq",
        base_url="https://api.groq.com/openai/v1", default_model="llama-3.3-70b-versatile",
        env_api_key="GROQ_API_KEY",
        description="Groq — ultra-fast LPU inference for Llama/Mixtral models",
    ),
    "qwen": ProviderProfile(
        name="qwen", display_name="通义千问 (Qwen)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", default_model="qwen-plus",
        env_api_key="DASHSCOPE_API_KEY",
        description="通义千问 — Alibaba Cloud DashScope, Qwen series models",
    ),
    "openai": ProviderProfile(
        name="openai", display_name="OpenAI",
        base_url="https://api.openai.com/v1", default_model="gpt-4o",
        env_api_key="OPENAI_API_KEY",
        description="OpenAI — GPT-4o, GPT-4.1, o3, o4-mini",
    ),
    "anthropic": ProviderProfile(
        name="anthropic", display_name="Anthropic Claude",
        base_url="https://api.anthropic.com", default_model="claude-sonnet-4-20250514",
        env_api_key="ANTHROPIC_API_KEY",
        description="Anthropic Claude — Sonnet 4, Opus 4 (native SDK)",
        requires_native_sdk=True,
    ),
    "gemini": ProviderProfile(
        name="gemini", display_name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com", default_model="gemini-2.5-flash",
        env_api_key="GEMINI_API_KEY",
        description="Google Gemini — Flash/Pro models (native SDK)",
        requires_native_sdk=True,
    ),
    # ── Extended providers ──────────────────────────────────────
    "mistral": ProviderProfile(
        name="mistral", display_name="Mistral AI",
        base_url="https://api.mistral.ai/v1", default_model="mistral-large-latest",
        env_api_key="MISTRAL_API_KEY",
        description="Mistral AI — Mistral Large, Small, Codestral",
    ),
    "together": ProviderProfile(
        name="together", display_name="Together AI",
        base_url="https://api.together.xyz/v1", default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        env_api_key="TOGETHER_API_KEY",
        description="Together AI — fast open-source model inference, 200+ models",
    ),
    "perplexity": ProviderProfile(
        name="perplexity", display_name="Perplexity",
        base_url="https://api.perplexity.ai", default_model="sonar-pro",
        env_api_key="PERPLEXITY_API_KEY",
        description="Perplexity — search-grounded LLMs (sonar, llama-sonar)",
    ),
    "cohere": ProviderProfile(
        name="cohere", display_name="Cohere",
        base_url="https://api.cohere.ai/v1", default_model="command-r-plus",
        env_api_key="COHERE_API_KEY",
        description="Cohere — Command R/R+, enterprise RAG models",
    ),
    "xai": ProviderProfile(
        name="xai", display_name="xAI (Grok)",
        base_url="https://api.x.ai/v1", default_model="grok-3-beta",
        env_api_key="XAI_API_KEY",
        description="xAI Grok — Grok-3, real-time knowledge, strong reasoning",
    ),
    "replicate": ProviderProfile(
        name="replicate", display_name="Replicate",
        base_url="https://api.replicate.com/v1", default_model="meta/meta-llama-3-70b-instruct",
        env_api_key="REPLICATE_API_KEY",
        description="Replicate — run open-source models in the cloud",
    ),
    "azure": ProviderProfile(
        name="azure", display_name="Azure OpenAI",
        base_url="https://{resource}.openai.azure.com/openai/deployments/{deployment}",
        default_model="gpt-4o", env_api_key="AZURE_OPENAI_API_KEY",
        description="Azure OpenAI Service — enterprise-grade GPT models",
    ),
    "cloudflare": ProviderProfile(
        name="cloudflare", display_name="Cloudflare Workers AI",
        base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        default_model="@cf/meta/llama-3.1-8b-instruct", env_api_key="CLOUDFLARE_API_TOKEN",
        description="Cloudflare Workers AI — serverless inference at the edge",
    ),
    "fireworks": ProviderProfile(
        name="fireworks", display_name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        env_api_key="FIREWORKS_API_KEY",
        description="Fireworks AI — fast compound AI inference, 100+ models",
    ),
    "huggingface": ProviderProfile(
        name="huggingface", display_name="HuggingFace TGI",
        base_url="https://api-inference.huggingface.co/v1",
        default_model="meta-llama/Meta-Llama-3-8B-Instruct", env_api_key="HF_API_KEY",
        description="HuggingFace Inference API — serverless open model hosting",
    ),
}


# ============================================================================
# Provider Registry
# ============================================================================


class ProviderRegistry:
    """Central registry for LLM provider profiles."""

    def __init__(self, profiles: dict[str, ProviderProfile] | None = None):
        self._profiles: dict[str, ProviderProfile] = dict(profiles or BUILTIN_PROFILES)

    def register(self, profile: ProviderProfile) -> None:
        self._profiles[profile.name] = profile

    def unregister(self, name: str) -> bool:
        return self._profiles.pop(name, None) is not None

    def get(self, name: str) -> ProviderProfile | None:
        return self._profiles.get(name)

    def list(self) -> list[ProviderProfile]:
        return sorted(self._profiles.values(), key=lambda p: p.display_name)

    def list_names(self) -> list[str]:
        return sorted(self._profiles.keys())

    def create_provider(self, name: str, api_key: str | None = None,
                        model: str | None = None, max_tokens: int | None = None) -> Any:
        profile = self.get(name)
        if profile is None:
            raise ValueError(f"Unknown provider: {name!r}. Available: {self.list_names()}")
        config = profile.make_config(api_key=api_key, model=model, max_tokens=max_tokens)
        if profile.requires_native_sdk:
            return self._create_native(profile, config)
        return self._create_openai_compat(config)

    def _create_openai_compat(self, config: ModelConfig) -> Any:
        from kairos.providers.base import ModelProvider
        return ModelProvider(config)

    def _create_native(self, profile: ProviderProfile, config: ModelConfig) -> Any:
        if profile.name == "anthropic":
            from kairos.providers.anthropic_adapter import AnthropicProvider
            return AnthropicProvider(config)
        elif profile.name == "gemini":
            from kairos.providers.gemini_adapter import GeminiProvider
            return GeminiProvider(config)
        raise ValueError(f"Native SDK not implemented for {profile.name!r}")

    def make_config(self, name: str, api_key: str | None = None,
                    model: str | None = None, max_tokens: int | None = None) -> ModelConfig:
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


def list_providers() -> list[dict[str, Any]]:
    profiles = BUILTIN_PROFILES
    result = []
    for name, p in sorted(profiles.items()):
        result.append({
            "name": p.name, "display_name": p.display_name,
            "default_model": p.default_model, "env_api_key": p.env_api_key,
            "description": p.description, "requires_native_sdk": p.requires_native_sdk,
        })
    return result


def get_provider(name: str) -> ProviderProfile | None:
    return BUILTIN_PROFILES.get(name)
