"""Tests for Kairos config schema validation (config_schema.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kairos.config_schema import (
    KairosConfigSchema, validate_config, validate_config_or_report,
    ModelConfigSchema, AgentConfigSchema, GatewayConfigSchema,
    LoggingConfigSchema, SandboxConfigSchema, ProviderConfigSchema,
)


class TestModelConfigSchema:
    def test_defaults(self):
        c = ModelConfigSchema()
        assert c.provider == "deepseek"
        assert c.temperature == 0.7

    def test_temperature_range(self):
        ModelConfigSchema(temperature=0.0)
        ModelConfigSchema(temperature=2.0)
        with pytest.raises(ValidationError):
            ModelConfigSchema(temperature=3.0)
        with pytest.raises(ValidationError):
            ModelConfigSchema(temperature=-0.1)

    def test_max_tokens_range(self):
        ModelConfigSchema(max_tokens=1)
        ModelConfigSchema(max_tokens=200000)
        with pytest.raises(ValidationError):
            ModelConfigSchema(max_tokens=0)
        with pytest.raises(ValidationError):
            ModelConfigSchema(max_tokens=200001)


class TestGatewayConfigSchema:
    def test_port_range(self):
        GatewayConfigSchema(port=1)
        GatewayConfigSchema(port=65535)
        with pytest.raises(ValidationError):
            GatewayConfigSchema(port=0)
        with pytest.raises(ValidationError):
            GatewayConfigSchema(port=65536)


class TestSandboxConfigSchema:
    def test_provider_valid(self):
        SandboxConfigSchema(provider="local")
        SandboxConfigSchema(provider="docker")
        SandboxConfigSchema(provider="ssh")
        with pytest.raises(ValidationError):
            SandboxConfigSchema(provider="invalid")


class TestLoggingConfigSchema:
    def test_level_valid(self):
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            LoggingConfigSchema(level=level)
        with pytest.raises(ValidationError):
            LoggingConfigSchema(level="TRACE")


class TestAgentConfigSchema:
    def test_max_iterations_range(self):
        AgentConfigSchema(max_iterations=1)
        AgentConfigSchema(max_iterations=200)
        with pytest.raises(ValidationError):
            AgentConfigSchema(max_iterations=0)


class TestProviderConfigSchema:
    def test_base_url_must_be_http(self):
        ProviderConfigSchema(base_url="https://api.test.com")
        with pytest.raises(ValidationError):
            ProviderConfigSchema(base_url="ftp://bad.url")


class TestKairosFullConfig:
    """End-to-end config validation."""

    def test_minimal_valid_config(self):
        config = validate_config({})
        assert config.model.name == "deepseek-chat"

    def test_full_valid_config(self):
        raw = {
            "model": {"provider": "openai", "name": "gpt-4", "temperature": 0.5},
            "providers": {"openai": {"api_key": "sk-test", "base_url": "https://api.openai.com"}},
            "agent": {"name": "MyAgent", "max_iterations": 50},
            "gateway": {"port": 9000},
            "sandbox": {"provider": "docker", "timeout": 600},
        }
        config = validate_config(raw)
        assert config.model.provider == "openai"
        assert config.gateway.port == 9000

    def test_invalid_temperature(self):
        _, errors = validate_config_or_report({"model": {"temperature": 5.0}})
        assert len(errors) > 0
        assert any("temperature" in e for e in errors)

    def test_invalid_port(self):
        _, errors = validate_config_or_report({"gateway": {"port": 99999}})
        assert len(errors) > 0

    def test_extra_fields_allowed(self):
        """Unknown fields should be accepted for forward compatibility."""
        config = validate_config({"custom_feature": {"enabled": True}})
        assert config is not None

    def test_env_var_placeholders_ok(self):
        """API keys with ${VAR} placeholders are valid strings."""
        config = validate_config({
            "providers": {"deepseek": {"api_key": "${DEEPSEEK_API_KEY}"}}
        })
        assert config is not None
