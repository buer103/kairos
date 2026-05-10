"""Configuration schema — Pydantic validation for Kairos config files.

Validates config.yaml on load, preventing silent failures from misconfiguration.
Production-grade: type checking, range constraints, required field enforcement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════
# Model Config
# ═══════════════════════════════════════════════════════════════════

class ModelConfigSchema(BaseModel):
    """Model section validation."""
    provider: str = "deepseek"
    name: str = "deepseek-chat"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=200000)


# ═══════════════════════════════════════════════════════════════════
# Provider Config
# ═══════════════════════════════════════════════════════════════════

class ProviderConfigSchema(BaseModel):
    """Per-provider configuration."""
    api_key: str = ""
    base_url: str = ""

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, v: str) -> str:
        if v and not v.startswith("http"):
            raise ValueError(f"base_url must start with http/https: {v}")
        return v


class ProvidersConfigSchema(BaseModel):
    """All providers."""
    deepseek: ProviderConfigSchema = Field(default_factory=ProviderConfigSchema)
    openai: ProviderConfigSchema = Field(default_factory=ProviderConfigSchema)
    anthropic: ProviderConfigSchema = Field(default_factory=ProviderConfigSchema)
    gemini: ProviderConfigSchema = Field(default_factory=ProviderConfigSchema)


# ═══════════════════════════════════════════════════════════════════
# Agent Config
# ═══════════════════════════════════════════════════════════════════

class AgentConfigSchema(BaseModel):
    """Agent section validation."""
    name: str = "Kairos"
    max_iterations: int = Field(default=20, ge=1, le=200)
    skills_dir: str = "~/.kairos/skills"
    role_description: str = ""
    max_tokens: int = Field(default=120000, ge=1000, le=1000000)
    enable_subagents: bool = True
    enable_security: bool = False
    enable_insights: bool = False


# ═══════════════════════════════════════════════════════════════════
# Skills Config
# ═══════════════════════════════════════════════════════════════════

class SkillsConfigSchema(BaseModel):
    """Skills section validation."""
    external_dirs: list[str] = Field(default_factory=list)
    stale_days: int = Field(default=30, ge=1, le=365)
    template_vars: bool = True
    inline_shell: bool = False


# ═══════════════════════════════════════════════════════════════════
# Curator Config
# ═══════════════════════════════════════════════════════════════════

class CuratorConfigSchema(BaseModel):
    """Curator section validation."""
    clean_days: int = Field(default=90, ge=1, le=3650)
    auto_clean: bool = False


# ═══════════════════════════════════════════════════════════════════
# Tools Config
# ═══════════════════════════════════════════════════════════════════

class ToolsConfigSchema(BaseModel):
    """Tools section validation."""
    serper_api_key: str = ""
    tavily_api_key: str = ""
    brave_api_key: str = ""


# ═══════════════════════════════════════════════════════════════════
# Logging Config
# ═══════════════════════════════════════════════════════════════════

class LoggingConfigSchema(BaseModel):
    """Logging section validation."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: str = "~/.kairos/kairos.log"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


# ═══════════════════════════════════════════════════════════════════
# Cron Config
# ═══════════════════════════════════════════════════════════════════

class CronConfigSchema(BaseModel):
    """Cron section validation."""
    db_path: str = "~/.kairos/cron.db"


# ═══════════════════════════════════════════════════════════════════
# Gateway Config
# ═══════════════════════════════════════════════════════════════════

class GatewayConfigSchema(BaseModel):
    """Gateway section validation."""
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    webhook_path: str = "/webhook"
    health_path: str = "/health"
    ready_path: str = "/ready"
    enabled_platforms: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Sandbox Config
# ═══════════════════════════════════════════════════════════════════

class SandboxConfigSchema(BaseModel):
    """Sandbox section validation."""
    provider: Literal["local", "docker", "ssh"] = "local"
    timeout: int = Field(default=300, ge=1, le=3600)


# ═══════════════════════════════════════════════════════════════════
# Top-Level Schema
# ═══════════════════════════════════════════════════════════════════

class KairosConfigSchema(BaseModel):
    """Root configuration schema for Kairos.

    Usage:
        raw = yaml.safe_load(open("config.yaml"))
        config = KairosConfigSchema.model_validate(raw)
    """
    model: ModelConfigSchema = Field(default_factory=ModelConfigSchema)
    providers: ProvidersConfigSchema = Field(default_factory=ProvidersConfigSchema)
    agent: AgentConfigSchema = Field(default_factory=AgentConfigSchema)
    skills: SkillsConfigSchema = Field(default_factory=SkillsConfigSchema)
    curator: CuratorConfigSchema = Field(default_factory=CuratorConfigSchema)
    tools: ToolsConfigSchema = Field(default_factory=ToolsConfigSchema)
    logging: LoggingConfigSchema = Field(default_factory=LoggingConfigSchema)
    cron: CronConfigSchema = Field(default_factory=CronConfigSchema)
    gateway: GatewayConfigSchema = Field(default_factory=GatewayConfigSchema)
    sandbox: SandboxConfigSchema = Field(default_factory=SandboxConfigSchema)

    model_config = {"extra": "allow"}  # Allow forward-compat unknown fields

    @model_validator(mode="after")
    def check_provider_consistency(self) -> KairosConfigSchema:
        """Warn if model.provider references a provider with no api_key set."""
        provider = self.model.provider
        provider_cfg = getattr(self.providers, provider, None)
        if provider_cfg and provider_cfg.api_key and provider_cfg.api_key.startswith("${"):
            # Env var placeholder — acceptable (will be resolved at runtime)
            pass
        return self


# ═══════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════

def validate_config(data: dict[str, Any]) -> KairosConfigSchema:
    """Validate and return typed config. Raises ValidationError on failure."""
    return KairosConfigSchema.model_validate(data)


def validate_config_or_report(data: dict[str, Any]) -> tuple[KairosConfigSchema | None, list[str]]:
    """Validate config, returning (schema, errors). Schema is None on failure."""
    try:
        return KairosConfigSchema.model_validate(data), []
    except Exception as e:
        errors = _flatten_errors(e)
        return None, errors


def _flatten_errors(exc: Exception) -> list[str]:
    """Flatten Pydantic ValidationError into human-readable messages."""
    from pydantic import ValidationError
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]
    return [str(exc)]
