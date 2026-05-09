"""Setup Wizard — guided first-time configuration for Kairos.

Interactive prompts walk the user through:
  - API key configuration (OpenAI / DeepSeek / Anthropic / Gemini)
  - Provider selection and model preference
  - Gateway configuration (enable platforms, set credentials)
  - Skin/theme selection
  - Plugin discovery

Saves configuration to ~/.config/kairos/config.yaml
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any


class SetupWizard:
    """Guided first-time setup with Rich TUI prompts."""

    def __init__(self):
        self._config: dict[str, Any] = {}
        self._steps_completed: list[str] = []

    # ── Main entry ────────────────────────────────────────────────────────

    def run(self, force: bool = False) -> dict[str, Any]:
        """Run the full setup wizard. Returns the generated config dict.

        If a config already exists and force=False, prompts to overwrite.
        """
        config_path = self._default_config_path()

        if config_path.exists() and not force:
            self._print_header("Configuration Found")
            self._print(f"Existing config at: {config_path}")
            self._print()
            choice = self._ask("Overwrite existing config?", ["y", "n"], default="n")
            if choice != "y":
                self._print("Setup cancelled. Use --force to overwrite.")
                return {}

        self._print_header("Kairos Setup Wizard")
        self._print("Let's configure your Kairos agent.")
        self._print()

        # Step 1: API keys
        self._config["providers"] = self._setup_api_keys()

        # Step 2: Model preferences
        self._config["model"] = self._setup_model()

        # Step 3: Skin
        self._config["cli"] = self._setup_skin()

        # Step 4: Gateway
        self._config["gateway"] = self._setup_gateway()

        # Step 5: Plugins
        self._config["plugins"] = self._setup_plugins()

        # Save
        self._save_config(config_path)

        self._print_header("Setup Complete!")
        self._print(f"Configuration saved to: {config_path}")
        self._print()
        self._print("Next steps:")
        self._print("  • Run 'kairos chat' to start chatting")
        self._print("  • Run 'kairos skill install <url>' to add skills")
        self._print("  • Run 'kairos gateway start' to enable multi-platform")
        self._print()

        return self._config

    # ── Step 1: API keys ──────────────────────────────────────────────────

    def _setup_api_keys(self) -> dict[str, Any]:
        self._print_header("1/5 — API Keys")
        self._print("Kairos needs at least one LLM provider to function.")
        self._print("Keys are stored in ~/.config/kairos/config.yaml")
        self._print("(You can also use environment variables: OPENAI_API_KEY, etc.)")
        self._print()

        providers: dict[str, Any] = {}

        # OpenAI
        self._print("OpenAI (chat.openai.com/api-keys)")
        key = self._ask("OpenAI API Key", default="", password=True)
        if key:
            providers["openai"] = {"api_key": key, "base_url": "https://api.openai.com/v1"}
            self._steps_completed.append("openai")

        # DeepSeek
        self._print()
        self._print("DeepSeek (platform.deepseek.com/api_keys)")
        key = self._ask("DeepSeek API Key", default="", password=True)
        if key:
            providers["deepseek"] = {"api_key": key, "base_url": "https://api.deepseek.com/v1"}
            self._steps_completed.append("deepseek")

        # Anthropic
        self._print()
        self._print("Anthropic (console.anthropic.com/settings/keys)")
        key = self._ask("Anthropic API Key", default="", password=True)
        if key:
            providers["anthropic"] = {"api_key": key}
            self._steps_completed.append("anthropic")

        # Google Gemini
        self._print()
        self._print("Google Gemini (aistudio.google.com/app/apikey)")
        key = self._ask("Gemini API Key", default="", password=True)
        if key:
            providers["gemini"] = {"api_key": key}
            self._steps_completed.append("gemini")

        if not providers:
            self._print()
            self._print("⚠️  No API keys configured. You can add them later:")
            self._print("   Run 'kairos config init' or edit ~/.config/kairos/config.yaml")

        return providers

    # ── Step 2: Model ────────────────────────────────────────────────────

    def _setup_model(self) -> dict[str, Any]:
        self._print_header("2/5 — Model Preferences")

        # Default provider
        providers = list(self._config.get("providers", {}).keys())
        if not providers:
            self._print("No providers configured, skipping model setup.")
            return {}

        if len(providers) == 1:
            default_provider = providers[0]
        else:
            self._print("Available providers: " + ", ".join(providers))
            default_provider = self._ask("Default provider", choices=providers, default=providers[0])

        # Common models per provider
        model_map = {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "deepseek": ["deepseek-chat", "deepseek-reasoner"],
            "anthropic": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
            "gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
        }

        models = model_map.get(default_provider, [default_provider])
        default_model = self._ask("Default model", choices=models, default=models[0] if models else "")

        temperature = self._ask("Temperature (0.0-2.0)", default="0.7")
        max_tokens = self._ask("Max tokens per response", default="4096")

        return {
            "provider": default_provider,
            "model": default_model,
            "temperature": float(temperature) if temperature else 0.7,
            "max_tokens": int(max_tokens) if max_tokens and max_tokens.isdigit() else 4096,
        }

    # ── Step 3: Skin ──────────────────────────────────────────────────────

    def _setup_skin(self) -> dict[str, Any]:
        self._print_header("3/5 — Terminal Skin")

        skins = [
            ("default", "Clean, readable default"),
            ("hacker", "Matrix-inspired green-on-black"),
            ("retro", "80s terminal aesthetic"),
            ("minimal", "Bare essentials, low noise"),
            ("ocean", "Deep blue ocean colors"),
            ("sunset", "Warm orange/purple palette"),
            ("forest", "Natural greens and browns"),
            ("midnight", "Dark mode with subtle accents"),
            ("neon", "Cyberpunk neon aesthetic"),
            ("mono", "Black and white, high contrast"),
        ]

        choices = [name for name, _ in skins]
        self._print("Available skins:")
        for name, desc in skins:
            self._print(f"  {name:<12} — {desc}")
        self._print()

        skin = self._ask("Choose skin", choices=choices, default="default")

        stream = self._ask("Enable streaming output? (recommended)", ["y", "n"], default="y")

        return {
            "skin": skin,
            "streaming": stream == "y",
        }

    # ── Step 4: Gateway ──────────────────────────────────────────────────

    def _setup_gateway(self) -> dict[str, Any]:
        self._print_header("4/5 — Multi-Platform Gateway")
        self._print("Kairos can connect to messaging platforms.")
        self._print("You need platform-specific tokens/credentials.")
        self._print()

        enable = self._ask("Enable multi-platform gateway?", ["y", "n"], default="n")
        if enable != "y":
            return {"enabled": False}

        gateway: dict[str, Any] = {"enabled": True, "platforms": {}}

        platforms = {
            "telegram": ("Telegram", "Bot Token (from @BotFather)"),
            "discord": ("Discord", "Bot Token"),
            "slack": ("Slack", "Bot Token + Signing Secret"),
            "wechat": ("WeChat", "App ID + App Secret"),
            "whatsapp": ("WhatsApp", "Phone Number ID + Access Token"),
        }

        for key, (name, desc) in platforms.items():
            self._print(f"--- {name} ---")
            self._print(f"Requires: {desc}")
            enable_plat = self._ask(f"Enable {name}?", ["y", "n"], default="n")
            if enable_plat == "y":
                creds: dict[str, str] = {}
                if key == "telegram":
                    creds["bot_token"] = self._ask("Bot Token", password=True)
                elif key == "discord":
                    creds["bot_token"] = self._ask("Bot Token", password=True)
                elif key == "slack":
                    creds["bot_token"] = self._ask("Bot Token", password=True)
                    creds["signing_secret"] = self._ask("Signing Secret", password=True)
                elif key == "wechat":
                    creds["app_id"] = self._ask("App ID")
                    creds["app_secret"] = self._ask("App Secret", password=True)
                elif key == "whatsapp":
                    creds["phone_number_id"] = self._ask("Phone Number ID")
                    creds["access_token"] = self._ask("Access Token", password=True)

                if creds:
                    gateway["platforms"][key] = creds

            self._print()

        gateway["webhook_port"] = int(self._ask("Webhook port", default="8080") or "8080")
        gateway["webhook_host"] = self._ask("Webhook host", default="0.0.0.0") or "0.0.0.0"

        return gateway

    # ── Step 5: Plugins ──────────────────────────────────────────────────

    def _setup_plugins(self) -> dict[str, Any]:
        self._print_header("5/5 — Plugins")
        self._print("Plugins extend Kairos with custom tools, middleware, and providers.")
        self._print()

        enable = self._ask("Enable plugin auto-discovery?", ["y", "n"], default="n")
        if enable != "y":
            return {"enabled": False, "auto_discover": False}

        plugin_dir = self._ask("Plugin directory", default="~/.kairos/plugins")

        return {
            "enabled": True,
            "auto_discover": True,
            "plugin_dir": plugin_dir,
        }

    # ── Save ──────────────────────────────────────────────────────────────

    def _save_config(self, path: Path) -> None:
        """Write the generated config to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("# Kairos Agent Configuration")
        lines.append(f"# Generated by setup wizard: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # YAML-like output (simple key-value for each section)
        lines.append("# --- Providers ---")
        for provider, cfg in self._config.get("providers", {}).items():
            lines.append(f"{provider}:")
            for k, v in cfg.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        lines.append("# --- Model ---")
        for k, v in self._config.get("model", {}).items():
            lines.append(f"{k}: {v}")
        lines.append("")

        lines.append("# --- CLI ---")
        for k, v in self._config.get("cli", {}).items():
            lines.append(f"{k}: {v}")
        lines.append("")

        lines.append("# --- Gateway ---")
        lines.append("gateway:")
        gw = self._config.get("gateway", {})
        lines.append(f"  enabled: {gw.get('enabled', False)}")
        if gw.get("enabled"):
            lines.append(f"  webhook_port: {gw.get('webhook_port', 8080)}")
            lines.append(f"  webhook_host: {gw.get('webhook_host', '0.0.0.0')}")
            for plat, creds in gw.get("platforms", {}).items():
                lines.append(f"  {plat}:")
                for k, v in creds.items():
                    lines.append(f"    {k}: {v}")
        lines.append("")

        lines.append("# --- Plugins ---")
        for k, v in self._config.get("plugins", {}).items():
            lines.append(f"{k}: {v}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    # ── UI helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _default_config_path() -> Path:
        return Path.home() / ".config" / "kairos" / "config.yaml"

    @staticmethod
    def _print_header(text: str) -> None:
        print()
        print("=" * 60)
        print(f"  {text}")
        print("=" * 60)
        print()

    @staticmethod
    def _print(text: str = "") -> None:
        if text:
            print(f"  {text}")
        else:
            print()

    @staticmethod
    def _ask(
        prompt: str,
        choices: list[str] | None = None,
        default: str = "",
        password: bool = False,
    ) -> str:
        """Ask a question and return the user's answer.

        Handles: open-ended, multiple choice, password masking.
        """
        if password:
            import getpass
            while True:
                value = getpass.getpass(f"  {prompt}: ").strip()
                if value:
                    return value
                if default:
                    return default
                print("  (required)")

        if choices:
            choices_str = "/".join(f"[{c}]" if c == default else c for c in choices)
            prompt_text = f"  {prompt} [{choices_str}]: "
        else:
            prompt_text = f"  {prompt}: "

        while True:
            value = input(prompt_text).strip()

            if not value and default:
                return default

            if choices and value not in choices:
                print(f"  Please choose: {', '.join(choices)}")
                continue

            if not value and not choices and not default:
                print("  (required)")
                continue

            return value


# ── Quick start ──────────────────────────────────────────────────────────


def quick_setup(api_key: str = "") -> dict[str, Any]:
    """Minimal setup with just an API key. Returns config dict.

    Args:
        api_key: OpenAI-compatible API key. If empty, reads from env.
    """
    wizard = SetupWizard()
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""

    if not key:
        key = wizard._ask("API Key", password=True)

    config = {
        "providers": {
            "openai": {"api_key": key, "base_url": "https://api.openai.com/v1"},
        },
        "model": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096},
        "cli": {"skin": "default", "streaming": True},
        "gateway": {"enabled": False},
        "plugins": {"enabled": False},
    }

    wizard._config = config
    config_path = wizard._default_config_path()
    wizard._save_config(config_path)

    print(f"✅ Quick setup complete! Config saved to {config_path}")
    return config
