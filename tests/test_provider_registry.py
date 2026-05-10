"""Tests for kairos.providers.registry — provider profiles and registry."""

import os

import pytest

from kairos.providers.base import ModelConfig
from kairos.providers.registry import (
    BUILTIN_PROFILES,
    ProviderProfile,
    ProviderRegistry,
    get_provider,
    list_providers,
)


class TestProviderProfile:
    """Tests for ProviderProfile dataclass and make_config."""

    def test_make_config_defaults(self):
        profile = ProviderProfile(
            name="test",
            display_name="Test",
            base_url="https://test.api.com/v1",
            default_model="test-model",
        )
        config = profile.make_config(api_key="sk-test")
        assert config.base_url == "https://test.api.com/v1"
        assert config.model == "test-model"
        assert config.max_tokens == 4096
        assert config.api_key == "sk-test"

    def test_make_config_overrides(self):
        profile = ProviderProfile(
            name="test",
            display_name="Test",
            base_url="https://test.api.com/v1",
            default_model="test-model",
        )
        config = profile.make_config(
            api_key="sk-override",
            model="custom-model",
            max_tokens=100,
        )
        assert config.api_key == "sk-override"
        assert config.model == "custom-model"
        assert config.max_tokens == 100

    def test_make_config_env_key(self):
        os.environ["TEST_PROVIDER_KEY"] = "sk-from-env"
        try:
            profile = ProviderProfile(
                name="test",
                display_name="Test",
                base_url="https://test.api.com/v1",
                default_model="test-model",
                env_api_key="TEST_PROVIDER_KEY",
            )
            config = profile.make_config()
            assert config.api_key == "sk-from-env"
        finally:
            del os.environ["TEST_PROVIDER_KEY"]

    def test_make_config_explicit_overrides_env(self):
        os.environ["TEST_PROVIDER_KEY"] = "sk-from-env"
        try:
            profile = ProviderProfile(
                name="test",
                display_name="Test",
                base_url="https://test.api.com/v1",
                default_model="test-model",
                env_api_key="TEST_PROVIDER_KEY",
            )
            config = profile.make_config(api_key="sk-explicit")
            assert config.api_key == "sk-explicit"
        finally:
            del os.environ["TEST_PROVIDER_KEY"]

    def test_extra_headers_propagated(self):
        profile = ProviderProfile(
            name="test",
            display_name="Test",
            base_url="https://test.api.com/v1",
            default_model="test-model",
            extra_headers={"X-Custom": "value"},
        )
        config = profile.make_config(api_key="sk-test")
        assert config.extra_headers == {"X-Custom": "value"}


class TestBuiltinProfiles:
    """Verify all built-in profiles have correct structure."""

    def test_all_builtin_profiles_present(self):
        expected = {"deepseek", "openrouter", "groq", "qwen", "openai", "anthropic", "gemini"}
        assert set(BUILTIN_PROFILES.keys()) == expected

    def test_deepseek_profile(self):
        p = BUILTIN_PROFILES["deepseek"]
        assert p.base_url == "https://api.deepseek.com"
        assert p.default_model == "deepseek-chat"
        assert p.env_api_key == "DEEPSEEK_API_KEY"
        assert not p.requires_native_sdk

    def test_openrouter_profile(self):
        p = BUILTIN_PROFILES["openrouter"]
        assert p.base_url == "https://openrouter.ai/api/v1"
        assert p.env_api_key == "OPENROUTER_API_KEY"
        assert "HTTP-Referer" in p.extra_headers
        assert not p.requires_native_sdk

    def test_groq_profile(self):
        p = BUILTIN_PROFILES["groq"]
        assert p.base_url == "https://api.groq.com/openai/v1"
        assert "llama" in p.default_model
        assert p.env_api_key == "GROQ_API_KEY"
        assert not p.requires_native_sdk

    def test_qwen_profile(self):
        p = BUILTIN_PROFILES["qwen"]
        assert "dashscope" in p.base_url
        assert p.default_model.startswith("qwen")
        assert p.env_api_key == "DASHSCOPE_API_KEY"
        assert not p.requires_native_sdk

    def test_openai_profile(self):
        p = BUILTIN_PROFILES["openai"]
        assert p.base_url == "https://api.openai.com/v1"
        assert p.default_model == "gpt-4o"
        assert not p.requires_native_sdk

    def test_anthropic_profile(self):
        p = BUILTIN_PROFILES["anthropic"]
        assert p.env_api_key == "ANTHROPIC_API_KEY"
        assert p.requires_native_sdk

    def test_gemini_profile(self):
        p = BUILTIN_PROFILES["gemini"]
        assert p.env_api_key == "GEMINI_API_KEY"
        assert p.requires_native_sdk

    def test_all_profiles_have_display_name(self):
        for profile in BUILTIN_PROFILES.values():
            assert profile.display_name, f"{profile.name} missing display_name"
            assert profile.description, f"{profile.name} missing description"


class TestProviderRegistry:
    """Tests for ProviderRegistry — creation, queries, and provider creation."""

    @pytest.fixture
    def registry(self):
        return ProviderRegistry()

    def test_get_known_provider(self, registry):
        profile = registry.get("deepseek")
        assert profile is not None
        assert profile.name == "deepseek"

    def test_get_unknown_provider(self, registry):
        assert registry.get("nonexistent") is None

    def test_list_names(self, registry):
        names = registry.list_names()
        assert "deepseek" in names
        assert "openrouter" in names
        assert "groq" in names
        assert "qwen" in names

    def test_list_profiles(self, registry):
        profiles = registry.list()
        assert len(profiles) >= 7
        # Sorted by display_name
        display_names = [p.display_name for p in profiles]
        assert display_names == sorted(display_names)

    def test_register_custom_profile(self, registry):
        custom = ProviderProfile(
            name="custom-provider",
            display_name="Custom",
            base_url="https://custom.api.com/v1",
            default_model="custom-model",
        )
        registry.register(custom)
        assert "custom-provider" in registry
        assert registry.get("custom-provider").base_url == "https://custom.api.com/v1"

    def test_unregister(self, registry):
        custom = ProviderProfile(
            name="temp",
            display_name="Temp",
            base_url="https://temp.api.com/v1",
            default_model="temp-model",
        )
        registry.register(custom)
        assert registry.unregister("temp") is True
        assert registry.get("temp") is None
        assert registry.unregister("nonexistent") is False

    def test_contains(self, registry):
        assert "deepseek" in registry
        assert "nonexistent" not in registry

    def test_len(self, registry):
        assert len(registry) == len(BUILTIN_PROFILES)

    def test_iter(self, registry):
        names = {p.name for p in registry}
        assert "deepseek" in names
        assert "openrouter" in names

    def test_make_config(self, registry):
        config = registry.make_config("deepseek", api_key="sk-test")
        assert isinstance(config, ModelConfig)
        assert config.base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-chat"

    def test_make_config_unknown_raises(self, registry):
        with pytest.raises(ValueError, match="Unknown provider"):
            registry.make_config("nonexistent")

    def test_create_provider_openai_compat(self, registry):
        provider = registry.create_provider("deepseek", api_key="sk-test")
        from kairos.providers.base import ModelProvider
        assert isinstance(provider, ModelProvider)
        assert provider.config.base_url == "https://api.deepseek.com"

    def test_create_provider_unknown_raises(self, registry):
        with pytest.raises(ValueError, match="Unknown provider"):
            registry.create_provider("nonexistent")

    def test_create_provider_native_sdk(self, registry):
        """Anthropic and Gemini should return native adapters."""
        # Anthropic
        provider = registry.create_provider("anthropic", api_key="sk-ant-test")
        from kairos.providers.anthropic_adapter import AnthropicProvider
        assert isinstance(provider, AnthropicProvider)

        # Gemini
        provider = registry.create_provider("gemini", api_key="sk-gem-test")
        from kairos.providers.gemini_adapter import GeminiProvider
        assert isinstance(provider, GeminiProvider)


class TestListProviders:
    """Tests for the list_providers() convenience function."""

    def test_list_providers_returns_all(self):
        providers = list_providers()
        names = {p["name"] for p in providers}
        assert "deepseek" in names
        assert "openrouter" in names
        assert "groq" in names
        assert "qwen" in names

    def test_list_providers_has_keys(self):
        providers = list_providers()
        for p in providers:
            assert "name" in p
            assert "display_name" in p
            assert "default_model" in p
            assert "env_api_key" in p
            assert "description" in p

    def test_get_provider(self):
        p = get_provider("deepseek")
        assert p is not None
        assert p.display_name == "DeepSeek"

        assert get_provider("nonexistent") is None


class TestProviderRegistryConfigOverride:
    """Tests for model/max_tokens overrides in create_provider."""

    @pytest.fixture
    def registry(self):
        return ProviderRegistry()

    def test_model_override(self, registry):
        config = registry.make_config("openai", api_key="sk-test", model="gpt-4.1")
        assert config.model == "gpt-4.1"

    def test_max_tokens_override(self, registry):
        config = registry.make_config("deepseek", api_key="sk-test", max_tokens=100)
        assert config.max_tokens == 100


class TestProviderProfilesEnvIntegration:
    """Tests for env var fallback behavior."""

    def test_env_fallback_used_when_no_explicit_key(self):
        os.environ["DEEPSEEK_API_KEY"] = "sk-env-deepseek"
        try:
            registry = ProviderRegistry()
            config = registry.make_config("deepseek")
            assert config.api_key == "sk-env-deepseek"
        finally:
            del os.environ["DEEPSEEK_API_KEY"]

    def test_explicit_key_wins_over_env(self):
        os.environ["DEEPSEEK_API_KEY"] = "sk-env-deepseek"
        try:
            registry = ProviderRegistry()
            config = registry.make_config("deepseek", api_key="sk-explicit")
            assert config.api_key == "sk-explicit"
        finally:
            del os.environ["DEEPSEEK_API_KEY"]
