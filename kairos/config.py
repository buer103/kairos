"""Configuration system — YAML config file + env var fallback.

Priority: env var > config.yaml > default

Config file locations (searched in order):
    1. KAIROS_CONFIG env var
    2. ./kairos.yaml (project-local)
    3. ~/.config/kairos/config.yaml (user-global)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Lazy import — only load yaml when needed
_yaml = None


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, using json as fallback if yaml not available."""
    global _yaml
    if _yaml is None:
        try:
            import yaml as _y
            _yaml = _y
        except ImportError:
            _yaml = False

    if _yaml:
        try:
            with open(path) as f:
                return _yaml.safe_load(f) or {}
        except Exception:
            return {}
    else:
        # JSON fallback
        import json
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}


def _find_config() -> Path | None:
    """Find the config file in standard locations."""
    # 1. Env var override
    env_path = os.environ.get("KAIROS_CONFIG")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    # 2. Project-local
    local = Path("kairos.yaml")
    if local.exists():
        return local

    # 3. User-global
    global_config = Path.home() / ".config" / "kairos" / "config.yaml"
    if global_config.exists():
        return global_config

    return None


class Config:
    """Layered configuration reader with dot-notation access.

    Usage::

        config = Config()
        api_key = config.get("providers.deepseek.api_key")
        model = config.get("model.name", default="deepseek-chat")
    """

    def __init__(self, path: str | Path | None = None, validate: bool = True):
        self._data: dict[str, Any] = {}
        self._path: Path | None = None
        self._validated: bool = False
        self._validation_errors: list[str] = []

        if path:
            p = Path(path).expanduser()
            if p.exists():
                self._path = p
                self._data = _load_yaml(p)
        else:
            found = _find_config()
            if found:
                self._path = found
                self._data = _load_yaml(found)

        # Merge env vars
        self._merge_env()

        # Validate schema
        if validate:
            self._run_validation()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by dot-separated key. Falls back to env var, then default.

        Examples:
            config.get("providers.deepseek.api_key")
            config.get("model.temperature", 0.7)
        """
        # Try config data
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = None
                break
        if value is not None:
            return value

        # Fall back to env var (uppercase, dots → underscores)
        env_key = key.upper().replace(".", "_")
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val

        return default

    def all(self) -> dict[str, Any]:
        """Return the full config dict (after env merge)."""
        return dict(self._data)

    def _merge_env(self) -> None:
        """Merge environment variables into the config tree.

        KAIROS_PROVIDERS_DEEPSEEK_API_KEY → config["providers"]["deepseek"]["api_key"]
        """
        # Map known env vars to config paths
        env_map = {
            "DEEPSEEK_API_KEY": "providers.deepseek.api_key",
            "OPENAI_API_KEY": "providers.openai.api_key",
            "ANTHROPIC_API_KEY": "providers.anthropic.api_key",
            "SERPER_API_KEY": "tools.serper_api_key",
            "TAVILY_API_KEY": "tools.tavily_api_key",
            "BRAVE_API_KEY": "tools.brave_api_key",
            "KAIROS_LOG_LEVEL": "logging.level",
            "KAIROS_LOG_FILE": "logging.file",
        }

        for env_var, config_path in env_map.items():
            val = os.environ.get(env_var)
            if val:
                self._set_nested(self._data, config_path.split("."), val)

    @staticmethod
    def _set_nested(d: dict, keys: list[str], value: Any) -> None:
        """Set a value in a nested dict, creating intermediate dicts."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def validated(self) -> bool:
        """Whether the loaded config passed schema validation."""
        return self._validated

    @property
    def validation_errors(self) -> list[str]:
        """Schema validation errors (empty if valid)."""
        return list(self._validation_errors)

    def validate(self) -> bool:
        """Explicitly re-validate the config. Returns True if valid."""
        self._run_validation()
        return self._validated

    def _run_validation(self) -> None:
        """Run schema validation and store results."""
        try:
            from kairos.config_schema import validate_config
            validate_config(self._data)
            self._validated = True
            self._validation_errors = []
        except Exception as e:
            self._validated = False
            from kairos.config_schema import _flatten_errors
            self._validation_errors = _flatten_errors(e)
            # Log but don't crash — backward compatible
            import logging
            logger = logging.getLogger("kairos.config")
            logger.warning(
                "Config validation failed (%d errors): %s",
                len(self._validation_errors),
                "; ".join(self._validation_errors[:3]),
            )

    def __repr__(self) -> str:
        return f"Config(path={self._path})" if self._path else "Config(empty)"


# Global config instance (lazy)
_config: Config | None = None


def get_config() -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def write_default_config(path: str | Path) -> None:
    """Write a default config.yaml file.

    Usage: kairos config init
    """
    import json

    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    default = {
        "model": {
            "provider": "deepseek",
            "name": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        "providers": {
            "deepseek": {
                "api_key": "${DEEPSEEK_API_KEY}",
                "base_url": "https://api.deepseek.com/v1",
            },
            "openai": {
                "api_key": "${OPENAI_API_KEY}",
                "base_url": "https://api.openai.com/v1",
            },
        },
        "agent": {
            "name": "Kairos",
            "max_iterations": 20,
            "skills_dir": "~/.kairos/skills",
        },
        "skills": {
            "external_dirs": [],
            "stale_days": 30,
            "template_vars": True,
            "inline_shell": False,
        },
        "curator": {
            "clean_days": 90,
            "auto_clean": False,
        },
        "tools": {
            "serper_api_key": "${SERPER_API_KEY}",
            "tavily_api_key": "${TAVILY_API_KEY}",
        },
        "logging": {
            "level": "INFO",
            "file": "~/.kairos/kairos.log",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
        "cron": {
            "db_path": "~/.kairos/cron.db",
        },
        "gateway": {
            "host": "127.0.0.1",
            "port": 8080,
        },
        "sandbox": {
            "provider": "local",
            "timeout": 300,
        },
    }

    with open(p, "w") as f:
        json.dump(default, f, indent=2)
        f.write("\n")

    print(f"✅ Default config written to {p}")
    print(f"   You can also use YAML format (rename to .yaml).")
