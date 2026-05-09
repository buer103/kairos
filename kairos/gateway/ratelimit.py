"""Rate limiter — per-platform and per-IP rate limiting with sliding window.

Configurable limits per platform per time window. Supports:
  - Sliding window counting (configurable window size)
  - Burst allowance (max burst above steady rate)
  - Auto-ban on repeated violations
  - Per-IP tracking
  - Thread-safe
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class RateLimiter:
    """Sliding-window rate limiter with per-key tracking.

    Usage:
        limiter = RateLimiter(default_rps=30)
        if limiter.is_allowed("telegram"):
            process()
        else:
            return 429

    Keys can be platform names, IPs, or custom identifiers.
    """

    def __init__(
        self,
        default_rps: int = 50,
        window_seconds: float = 1.0,
        max_burst: int | None = None,
        ban_threshold: int = 10,
        ban_duration: float = 60,
    ):
        self._default_rps = default_rps
        self._window = window_seconds
        self._max_burst = max_burst or (default_rps * 2)
        self._ban_threshold = ban_threshold
        self._ban_duration = ban_duration

        self._limits: dict[str, int] = {}
        self._timestamps: dict[str, list[float]] = defaultdict(list)
        self._violations: dict[str, int] = defaultdict(int)
        self._banned: dict[str, float] = {}  # key → ban_until timestamp
        self._lock = threading.Lock()

    # ── Configuration ──────────────────────────────────────────────────────

    def set_limit(self, key: str, rps: int) -> None:
        """Set rate limit for a specific key."""
        with self._lock:
            self._limits[key] = rps

    def get_limit(self, key: str) -> int:
        """Get rate limit for a key (default if not set)."""
        return self._limits.get(key, self._default_rps)

    # ── Rate checking ─────────────────────────────────────────────────────

    def is_allowed(self, key: str, now: float | None = None) -> bool:
        """Check if a request for `key` is allowed. Returns True if within limit."""
        if now is None:
            now = time.time()

        with self._lock:
            # Check ban
            ban_until = self._banned.get(key, 0)
            if now < ban_until:
                return False

            # Clean old timestamps
            limit = self.get_limit(key)
            window = self._window
            timestamps = self._timestamps[key]
            cutoff = now - window

            # Remove entries outside window
            self._timestamps[key] = [t for t in timestamps if t >= cutoff]
            current_count = len(self._timestamps[key])

            # Check limit
            if current_count >= limit:
                # Check burst allowance
                burst_window = now - (window * 2)
                burst_count = sum(1 for t in self._timestamps[key] if t >= burst_window)
                if burst_count >= self._max_burst * 2:
                    self._violations[key] += 1
                    if self._violations[key] >= self._ban_threshold:
                        self._banned[key] = now + self._ban_duration
                    return False

            # Allow and record
            self._timestamps[key].append(now)
            return True

    def allow_or_wait(self, key: str, max_wait: float = 5.0) -> bool:
        """Block until allowed, or timeout. Returns True if allowed."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if self.is_allowed(key):
                return True
            time.sleep(0.05)
        return False

    # ── Violations & bans ─────────────────────────────────────────────────

    def get_violations(self, key: str) -> int:
        """Get violation count for a key."""
        return self._violations.get(key, 0)

    def is_banned(self, key: str) -> bool:
        """Check if a key is currently banned."""
        ban_until = self._banned.get(key, 0)
        return time.time() < ban_until

    def unban(self, key: str) -> None:
        """Remove a ban on a key."""
        with self._lock:
            self._banned.pop(key, None)
            self._violations[key] = 0

    def reset(self, key: str) -> None:
        """Reset all tracking for a key."""
        with self._lock:
            self._timestamps.pop(key, None)
            self._violations.pop(key, None)
            self._banned.pop(key, None)

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self, key: str | None = None) -> dict[str, Any]:
        """Get rate limit stats. If key is None, returns summary for all keys."""
        with self._lock:
            now = time.time()
            if key:
                timestamps = self._timestamps.get(key, [])
                recent = [t for t in timestamps if t >= now - self._window]
                return {
                    "key": key,
                    "limit": self.get_limit(key),
                    "current_rps": len(recent),
                    "violations": self._violations.get(key, 0),
                    "banned": self.is_banned(key),
                    "ban_remaining": max(0, self._banned.get(key, 0) - now),
                }

            all_stats = {}
            for k in self._timestamps:
                recent = [t for t in self._timestamps[k] if t >= now - self._window]
                all_stats[k] = {
                    "limit": self.get_limit(k),
                    "current_rps": len(recent),
                    "violations": self._violations.get(k, 0),
                    "banned": self.is_banned(k),
                }
            return {
                "total_keys": len(self._timestamps),
                "banned_keys": sum(1 for _ in self._banned if self.is_banned(_)),
                "stats": all_stats,
            }

    def __repr__(self) -> str:
        return (
            f"RateLimiter(default_rps={self._default_rps}, "
            f"window={self._window}s, max_burst={self._max_burst})"
        )


# ── Multi-tier rate limiter ──────────────────────────────────────────────


class MultiTierLimiter:
    """Three-tier rate limiter: per-IP → per-platform → global.

    Checks are hierarchical: if any tier rejects, the request is denied.
    """

    def __init__(
        self,
        per_ip_rps: int = 10,
        per_platform_rps: int = 50,
        global_rps: int = 200,
    ):
        self.ip_limiter = RateLimiter(default_rps=per_ip_rps)
        self.platform_limiter = RateLimiter(default_rps=per_platform_rps)
        self.global_limiter = RateLimiter(default_rps=global_rps)

    def is_allowed(self, platform: str, ip: str = "") -> bool:
        """Check all three tiers. Returns True if all pass."""
        now = time.time()

        # Global tier
        if not self.global_limiter.is_allowed("global", now):
            return False

        # Platform tier
        if not self.platform_limiter.is_allowed(platform, now):
            return False

        # IP tier (optional)
        if ip and not self.ip_limiter.is_allowed(ip, now):
            return False

        return True

    def stats(self) -> dict[str, Any]:
        """Get combined stats from all tiers."""
        return {
            "ip": self.ip_limiter.stats() if hasattr(self.ip_limiter, 'stats') else {},
            "platform": self.platform_limiter.stats() if self.platform_limiter._timestamps else {},
            "global": self.global_limiter.stats(),
        }
