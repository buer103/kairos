"""Provider failover — multi-backend automatic degradation.

When the primary provider fails (429, 5xx, timeout), Kairos automatically
switches to fallback providers in a pre-configured chain.  Failed providers
enter a cooldown period and are retried after recovery.

Supports:
  - Ordered fallback chain (primary → fb1 → fb2 → ...)
  - Per-provider health tracking (consecutive failures, cooldown)
  - Auto-recovery after cooldown expires
  - Circuit breaker: 3 consecutive failures → disable until reset
  - Stats: failover_count, provider_health for observability
  - Thread-safe

Usage:
    primary = ModelConfig(api_key="sk-...", model="deepseek-chat")
    fb1 = ModelConfig(api_key="sk-...", model="claude-sonnet-4")
    fb2 = ModelConfig(api_key="sk-...", model="gpt-4o")

    failover = ProviderFailoverManager(
        primary=primary,
        fallbacks=[fb1, fb2],
        cooldown_seconds=300,
    )

    config = failover.current  # Use this for API calls
    failover.mark_success()    # Call after successful API call
    failover.mark_failure()    # Call on error → triggers failover

    if failover.is_failover_active:
        logger.warning("Running on fallback: %s", config.model)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from kairos.providers.base import ModelConfig


@dataclass
class ProviderHealth:
    """Health state for a single provider."""

    provider_key: str  # e.g. "deepseek-chat", "claude-sonnet-4"
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    disabled: bool = False
    total_calls: int = 0
    failover_count: int = 0  # Times this provider was used as fallback

    @property
    def available(self) -> bool:
        """Provider is healthy: not disabled, cooldown expired."""
        # Auto-recover: reset when cooldown expires (only if cooldown was set)
        if self.disabled and self.cooldown_until > 0 and time.time() >= self.cooldown_until:
            self.disabled = False
            self.cooldown_until = 0.0
            self.consecutive_failures = 0
        if self.disabled:
            return False
        if self.cooldown_until > 0 and time.time() < self.cooldown_until:
            return False
        return True

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_calls += 1

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.total_calls += 1

    def record_failover_use(self) -> None:
        self.failover_count += 1
        self.total_calls += 1


class ProviderFailoverManager:
    """Multi-backend failover with cooldown and auto-recovery.

    Maintains an ordered chain: primary → fallback[0] → fallback[1] → ...
    On failure, advances to the next healthy provider in the chain.
    Failed providers enter cooldown; recovered automatically after cooldown
    expires.
    """

    def __init__(
        self,
        primary: ModelConfig,
        fallbacks: list[ModelConfig] | None = None,
        cooldown_seconds: float = 300.0,
        max_consecutive_failures: int = 3,
    ):
        self._lock = threading.Lock()

        self._primary = primary
        self._fallbacks = fallbacks or []
        self._cooldown_seconds = cooldown_seconds
        self._max_failures = max_consecutive_failures

        # Build the full provider chain
        self._chain: list[ModelConfig] = [primary] + self._fallbacks
        self._provider_keys: list[str] = [
            self._make_key(cfg) for cfg in self._chain
        ]

        # Health tracking per provider
        self._health: dict[str, ProviderHealth] = {
            key: ProviderHealth(provider_key=key)
            for key in self._provider_keys
        }

        # Current position in the chain
        self._active_index: int = 0
        self._failover_active: bool = False
        self._failover_count: int = 0

    # ── Properties ───────────────────────────────────────────────

    @property
    def current(self) -> ModelConfig:
        """The currently active provider config."""
        with self._lock:
            return self._chain[self._active_index]

    @property
    def current_key(self) -> str:
        return self._provider_keys[self._active_index]

    @property
    def primary_key(self) -> str:
        return self._provider_keys[0]

    @property
    def is_failover_active(self) -> bool:
        return self._failover_active

    @property
    def failover_count(self) -> int:
        return self._failover_count

    # ── Public API ───────────────────────────────────────────────

    def acquire(self) -> ModelConfig:
        """Get the current provider config, recovering primary if possible."""
        with self._lock:
            self._try_recover_primary()
            return self._chain[self._active_index]

    def mark_success(self) -> None:
        """Record a successful API call on the current provider."""
        with self._lock:
            health = self._health[self.current_key]
            health.record_success()

    def mark_failure(self, error: str | None = None) -> bool:
        """Record a failed API call. Returns True if failover was triggered."""
        with self._lock:
            key = self.current_key
            health = self._health[key]
            health.record_failure()

            # Circuit breaker: consecutive failures > threshold → disable
            if health.consecutive_failures >= self._max_failures:
                health.disabled = True
                health.cooldown_until = time.time() + self._cooldown_seconds

                # Try failover
                if self._try_advance():
                    self._failover_active = True
                    self._failover_count += 1
                    new_health = self._health[self.current_key]
                    new_health.record_failover_use()
                    return True

            return False

    def mark_rate_limited(self, retry_after: float = 30.0) -> bool:
        """Mark current provider as rate-limited. Triggers failover."""
        with self._lock:
            key = self.current_key
            health = self._health[key]
            health.record_failure()
            health.cooldown_until = time.time() + retry_after

            if self._try_advance():
                self._failover_active = True
                self._failover_count += 1
                return True
            return False

    def reset(self) -> None:
        """Reset all providers to healthy state, return to primary."""
        with self._lock:
            for health in self._health.values():
                health.consecutive_failures = 0
                health.cooldown_until = 0.0
                health.disabled = False
            self._active_index = 0
            self._failover_active = False

    def stats(self) -> dict[str, Any]:
        """Return failover statistics."""
        with self._lock:
            providers = {}
            for key, health in self._health.items():
                providers[key] = {
                    "consecutive_failures": health.consecutive_failures,
                    "disabled": health.disabled,
                    "cooldown_left": max(0, health.cooldown_until - time.time()),
                    "total_calls": health.total_calls,
                    "failover_count": health.failover_count,
                    "is_current": key == self.current_key,
                    "is_primary": key == self.primary_key,
                }
            return {
                "active_provider": self.current_key,
                "is_failover": self._failover_active,
                "failover_count": self._failover_count,
                "chain_length": len(self._chain),
                "providers": providers,
            }

    # ── Internal ─────────────────────────────────────────────────

    def _try_advance(self) -> bool:
        """Advance to the next healthy provider. Returns True if advanced."""
        for offset in range(1, len(self._chain)):
            idx = (self._active_index + offset) % len(self._chain)
            if idx == self._active_index:
                break
            key = self._provider_keys[idx]
            if self._health[key].available:
                self._active_index = idx
                return True
        return False

    def _try_recover_primary(self) -> None:
        """If primary has recovered from cooldown, switch back."""
        if self._active_index == 0:
            return
        primary_health = self._health[self.primary_key]
        if primary_health.available:
            self._active_index = 0
            self._failover_active = False

    @staticmethod
    def _make_key(cfg: ModelConfig) -> str:
        """Generate a stable provider key from config."""
        parts = []
        if cfg.model:
            parts.append(cfg.model)
        if cfg.base_url:
            from urllib.parse import urlparse
            try:
                host = urlparse(cfg.base_url).hostname or ""
                if host:
                    parts.append(host)
            except Exception:
                pass
        return "-".join(parts) if parts else "unknown"


def build_failover_from_env() -> ProviderFailoverManager:
    """Build a failover chain from environment variables.

    Primary: DEEPSEEK_API_KEY → deepseek
    Fallbacks:
      - OPENAI_API_KEY → gpt-4o
      - ANTHROPIC_API_KEY → claude-sonnet-4

    Use KAIROS_FAILOVER_COOLDOWN to set cooldown seconds (default 300).
    """
    import os

    configs: list[ModelConfig] = []
    cooldown = float(os.getenv("KAIROS_FAILOVER_COOLDOWN", "300"))

    # DeepSeek
    if os.getenv("DEEPSEEK_API_KEY"):
        configs.append(ModelConfig(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        ))

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        configs.append(ModelConfig(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        ))

    # Anthropic (native SDK, requires anthropic package)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from kairos.providers.anthropic_adapter import AnthropicAdapter
            configs.append(ModelConfig(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            ))
        except ImportError:
            pass

    # Any generic OpenAI-compatible backup
    if os.getenv("KAIROS_FALLBACK_BASE_URL") and os.getenv("KAIROS_FALLBACK_API_KEY"):
        configs.append(ModelConfig(
            api_key=os.environ["KAIROS_FALLBACK_API_KEY"],
            base_url=os.environ["KAIROS_FALLBACK_BASE_URL"],
            model=os.getenv("KAIROS_FALLBACK_MODEL", "default"),
        ))

    if not configs:
        return None

    primary = configs[0]
    fallbacks = configs[1:] if len(configs) > 1 else []

    return ProviderFailoverManager(
        primary=primary,
        fallbacks=fallbacks,
        cooldown_seconds=cooldown,
    )
