"""Kairos Plugin — Model Provider Plugin.

Provides multi-provider model support with lazy SDK loading.
Registers as a 'provider' type plugin.

Supported providers:
  - OpenAI (native, always available)
  - Anthropic (lazy-loads anthropic SDK)
  - Google Gemini (lazy-loads google-genai SDK)
  - DeepSeek (OpenAI-compatible, auto-detected)
"""

from __future__ import annotations

PLUGIN_NAME = "kairos-provider-adapters"
PLUGIN_VERSION = "1.0.0"
PLUGIN_TYPE = "provider"
PLUGIN_DESCRIPTION = "Multi-provider model adapters with lazy SDK loading"


def register(manager) -> None:
    """Register provider adapter plugins.

    Makes AnthropicAdapter and GeminiAdapter discoverable through the
    PluginManager. OpenAI is built-in (not a plugin).

    Args:
        manager: The PluginManager instance
    """
    # Register Anthropic adapter
    manager._registry["provider"]["anthropic"] = {
        "description": "Anthropic Claude models (lazy-loads anthropic SDK)",
        "version": PLUGIN_VERSION,
        "models": [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ],
        "adapter_module": "kairos.providers.anthropic_adapter",
        "adapter_class": "AnthropicProvider",
    }

    # Register Gemini adapter
    manager._registry["provider"]["gemini"] = {
        "description": "Google Gemini models (lazy-loads google-genai SDK)",
        "version": PLUGIN_VERSION,
        "models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ],
        "adapter_module": "kairos.providers.gemini_adapter",
        "adapter_class": "GeminiProvider",
    }

    # Register DeepSeek (no extra SDK needed — OpenAI-compatible)
    manager._registry["provider"]["deepseek"] = {
        "description": "DeepSeek models (OpenAI-compatible, no extra SDK)",
        "version": PLUGIN_VERSION,
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "base_url": "https://api.deepseek.com/v1",
        "openai_compatible": True,
    }


def get_available_providers() -> dict[str, dict]:
    """Get all registered provider info."""
    from kairos.providers.registry import BUILTIN_PROFILES
    result = {}
    for name, profile in BUILTIN_PROFILES.items():
        entry = {
            "display_name": profile.display_name,
            "description": profile.description,
            "base_url": profile.base_url,
            "default_model": profile.default_model,
            "env_api_key": profile.env_api_key,
            "requires_native_sdk": profile.requires_native_sdk,
        }
        result[name] = entry
    return result
