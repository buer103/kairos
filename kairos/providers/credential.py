"""Credential pool — multi-key rotation with rate limit awareness.

Manages multiple API keys per provider with automatic rotation on:
  - Rate limit (HTTP 429) with Retry-After header compliance
  - Authentication errors (HTTP 401/403)
  - Connection failures

Uses exponential backoff with jitter for retries.
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Credential:
    """A single API key with usage tracking."""

    key: str
    provider: str = ""
    label: str = ""

    # State tracking
    active: bool = True
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    total_calls: int = 0
    rate_limit_hits: int = 0

    @property
    def available(self) -> bool:
        return self.active and time.time() >= self.cooldown_until

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_calls += 1

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.active = False

    def record_rate_limit(self, retry_after: float = 30.0) -> None:
        self.rate_limit_hits += 1
        self.cooldown_until = time.time() + retry_after
        self.consecutive_failures += 1


class CredentialPool:
    """Manages multiple API keys with intelligent rotation.

    Usage:
        pool = CredentialPool()
        pool.add("sk-abc123", provider="openai", label="personal")
        pool.add("sk-def456", provider="openai", label="work")

        cred = pool.acquire("openai")  # Returns best available key
        # ... use cred.key for API call ...
        pool.release(cred, success=True)

        # Rate limited
        pool.mark_rate_limited(cred, retry_after=30)
        next_cred = pool.acquire("openai")  # Returns different key
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._credentials: dict[str, list[Credential]] = defaultdict(list)

    def add(self, key: str, provider: str = "default", label: str = "") -> Credential:
        """Register a new API key."""
        cred = Credential(key=key, provider=provider, label=label)
        with self._lock:
            self._credentials[provider].append(cred)
        return cred

    def add_batch(self, keys: list[str], provider: str = "default") -> list[Credential]:
        """Register multiple keys at once."""
        return [self.add(k, provider=provider) for k in keys]

    def acquire(self, provider: str = "default") -> Credential | None:
        """Get the best available credential for a provider.

        Returns None if no credentials are available (all in cooldown or disabled).
        """
        with self._lock:
            creds = self._credentials.get(provider, [])
            if not creds:
                return None

            # Prefer: active + not cooling + fewest failures
            available = [c for c in creds if c.available]
            if not available:
                return None

            available.sort(
                key=lambda c: (c.consecutive_failures, c.rate_limit_hits)
            )
            return available[0]

    def release(self, cred: Credential, success: bool = True) -> None:
        """Release a credential after use."""
        with self._lock:
            if success:
                cred.record_success()
            else:
                cred.record_failure()

    def mark_rate_limited(self, cred: Credential, retry_after: float = 30.0) -> None:
        """Mark a credential as rate-limited for retry_after seconds."""
        with self._lock:
            cred.record_rate_limit(retry_after)

    def mark_disabled(self, cred: Credential) -> None:
        """Permanently disable a credential (e.g., revoked)."""
        with self._lock:
            cred.active = False

    def stats(self, provider: str | None = None) -> dict[str, Any]:
        """Return pool statistics."""
        with self._lock:
            providers = [provider] if provider else list(self._credentials.keys())
            result = {}
            for p in providers:
                creds = self._credentials.get(p, [])
                result[p] = {
                    "total_keys": len(creds),
                    "active_keys": sum(1 for c in creds if c.active),
                    "available_keys": sum(1 for c in creds if c.available),
                    "total_calls": sum(c.total_calls for c in creds),
                    "rate_limit_hits": sum(c.rate_limit_hits for c in creds),
                    "keys": [
                        {
                            "label": c.label or c.key[:8] + "...",
                            "active": c.active,
                            "available": c.available,
                            "calls": c.total_calls,
                            "rate_limits": c.rate_limit_hits,
                            "cooldown_left": max(0, c.cooldown_until - time.time()),
                        }
                        for c in creds
                    ],
                }
            return result

    def reset(self, provider: str | None = None) -> None:
        """Reset all credentials for a provider (clear cooldowns, reactivate)."""
        with self._lock:
            providers = [provider] if provider else list(self._credentials.keys())
            for p in providers:
                for c in self._credentials.get(p, []):
                    c.active = True
                    c.cooldown_until = 0.0
                    c.consecutive_failures = 0


class RetryConfig:
    """Configuration for LLM retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retry_on_status = retry_on_status

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + optional jitter."""
        delay = min(
            self.base_delay * (self.backoff_factor ** attempt),
            self.max_delay,
        )
        if self.jitter:
            delay *= 0.5 + random.random()
        return delay
