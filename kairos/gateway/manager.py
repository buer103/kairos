"""Gateway Manager — centralized adapter lifecycle, health checks, and routing.

Manages all platform adapters, their connection states, and routes
incoming messages to the appropriate adapter for translation and
outgoing responses back through the correct adapter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ConnectionState,
)
from kairos.gateway.adapters.base import PlatformAdapter
from kairos.gateway.auth import GatewayAuth

logger = logging.getLogger("kairos.gateway.manager")

# ── Reconnect config ──────────────────────────────────────────
RECONNECT_BASE_DELAY = 10.0      # seconds — first retry
RECONNECT_MAX_DELAY = 300.0      # seconds — max backoff (5 min)
RECONNECT_BACKOFF_FACTOR = 2.0   # exponential factor
RECONNECT_JITTER = 0.2           # ±20% jitter
RECONNECT_CHECK_INTERVAL = 5.0   # seconds between watcher cycles


@dataclass
class AdapterHealth:
    """Health status for a single adapter."""
    name: str
    state: ConnectionState
    connected_at: float = 0
    last_activity: float = 0
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    avg_latency_ms: float = 0

    @property
    def uptime_seconds(self) -> float:
        if self.connected_at > 0:
            return time.time() - self.connected_at
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "connected_at": self.connected_at,
            "uptime_s": round(self.uptime_seconds, 1),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "errors": self.errors,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "error_rate": round(
                self.errors / max(self.messages_received + self.messages_sent, 1), 3
            ),
        }


@dataclass
class RouteResult:
    """Result of routing a message through the gateway."""
    adapter_name: str
    response: UnifiedResponse | None
    error: str | None = None
    latency_ms: float = 0


@dataclass
class _ReconnectState:
    """Tracks exponential backoff state for a reconnecting adapter."""
    name: str
    failures: int = 0
    delay: float = 0.0
    next_attempt: float = 0.0


class GatewayManager:
    """Central manager for all platform adapters.

    Responsibilities:
      - Adapter lifecycle (connect all, disconnect all, health checks)
      - Message routing (incoming → translate → agent → translate → outgoing)
      - Health monitoring (per-adapter stats, error rates, latency)
      - Auto-reconnect for disconnected adapters
    """

    def __init__(self, auth: GatewayAuth | None = None):
        self._adapters: dict[str, PlatformAdapter] = {}
        self._health: dict[str, AdapterHealth] = {}
        self._started_at = time.time()
        self._lock = asyncio.Lock()
        self.auth = auth or GatewayAuth()

        # Reconnect watcher state
        self._reconnect_task: asyncio.Task | None = None
        self._reconnect_tracker: dict[str, _ReconnectState] = {}
        self._watcher_running = False

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, adapter: PlatformAdapter) -> None:
        """Register a platform adapter."""
        name = adapter.platform_name
        self._adapters[name] = adapter
        self._health[name] = AdapterHealth(
            name=name,
            state=adapter.state,
        )

    def unregister(self, platform_name: str) -> bool:
        """Remove a platform adapter."""
        if platform_name in self._adapters:
            del self._adapters[platform_name]
            self._health.pop(platform_name, None)
            return True
        return False

    def get(self, platform_name: str) -> PlatformAdapter | None:
        """Get an adapter by platform name."""
        return self._adapters.get(platform_name)

    @property
    def platforms(self) -> list[str]:
        """List all registered platform names."""
        return list(self._adapters.keys())

    @property
    def adapter_count(self) -> int:
        return len(self._adapters)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect_all(self) -> dict[str, bool]:
        """Connect all registered adapters. Starts background reconnect watcher."""
        results: dict[str, bool] = {}
        tasks = []

        for name, adapter in self._adapters.items():
            tasks.append((name, adapter.connect()))

        for name, coro in tasks:
            try:
                success = await asyncio.wait_for(coro, timeout=10)
                results[name] = success
                if self._health.get(name):
                    self._health[name].state = adapter.state
                    if success:
                        self._health[name].connected_at = time.time()
                        self._reconnect_tracker.pop(name, None)  # Clear retry state
                else:
                    self._record_reconnect_failure(name)
            except asyncio.TimeoutError:
                results[name] = False
                logger.warning("Adapter %s connection timed out", name)
                self._record_reconnect_failure(name)
            except Exception as e:
                results[name] = False
                logger.error("Adapter %s connection error: %s", name, e)
                self._record_reconnect_failure(name)

        connected = sum(1 for v in results.values() if v)
        logger.info(
            "Gateway connected %d/%d adapters", connected, len(results)
        )

        # Start background reconnect watcher
        self._start_watcher()
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all registered adapters. Stops reconnect watcher."""
        self._stop_watcher()
        for name, adapter in self._adapters.items():
            try:
                await asyncio.wait_for(adapter.disconnect(), timeout=5)
            except Exception as e:
                logger.warning("Adapter %s disconnect error: %s", name, e)

    async def shutdown(self) -> None:
        """Graceful shutdown: stop watcher, disconnect all adapters."""
        self._stop_watcher()
        await self.disconnect_all()

    # ── Message routing ───────────────────────────────────────────────────

    async def route(
        self,
        platform: str,
        chat_id: str,
        raw_message: dict[str, Any] | None = None,
        user_id: str = "",
    ) -> RouteResult:
        """Route a message through the pipeline.

        1. Auth check (whitelist/pairing/allow-all)
        2. Get the adapter for the platform
        3. Translate incoming → UnifiedMessage
        4. (Agent processing happens externally)

        Args:
            platform: Platform name (e.g., "telegram", "discord")
            chat_id: Platform-specific chat/channel ID
            raw_message: Raw platform payload (dict for translate_incoming)
            user_id: Optional user identifier for auth

        Returns:
            RouteResult with the adapter name and translated UnifiedMessage
        """
        t0 = time.time()

        # ── Auth check ──────────────────────────────────
        auth_result = self.auth.check(platform, chat_id=chat_id, user_id=user_id)
        if not auth_result.allowed:
            reason = auth_result.reason
            pairing_code = auth_result.pairing_code
            if pairing_code:
                reason = f"pairing_required (code: {pairing_code})"
            return RouteResult(
                adapter_name=platform,
                response=None,
                error=f"Auth denied: {reason}",
                latency_ms=(time.time() - t0) * 1000,
            )

        adapter = self._adapters.get(platform)

        if not adapter:
            return RouteResult(
                adapter_name=platform,
                response=None,
                error=f"No adapter registered for platform: {platform}",
                latency_ms=(time.time() - t0) * 1000,
            )

        if adapter.state != ConnectionState.CONNECTED:
            logger.warning("Adapter %s is %s, attempting reconnect", platform, adapter.state.value)
            try:
                await asyncio.wait_for(adapter.connect(), timeout=5)
            except Exception:
                pass

        if adapter.state != ConnectionState.CONNECTED:
            return RouteResult(
                adapter_name=platform,
                response=None,
                error=f"Adapter {platform} is {adapter.state.value}",
                latency_ms=(time.time() - t0) * 1000,
            )

        # Track health
        if platform in self._health:
            self._health[platform].messages_received += 1
            self._health[platform].last_activity = time.time()

        msg = adapter.translate_incoming(raw_message or {})
        latency = (time.time() - t0) * 1000
        if platform in self._health:
            self._health[platform].avg_latency_ms = (
                self._health[platform].avg_latency_ms * 0.9 + latency * 0.1
            )

        return RouteResult(
            adapter_name=platform,
            response=None,  # Caller provides agent response
            latency_ms=latency,
        )

    async def deliver(
        self,
        platform: str,
        chat_id: str,
        response: UnifiedResponse,
    ) -> RouteResult:
        """Deliver a UnifiedResponse through the appropriate adapter.

        Args:
            platform: Platform name to deliver through
            chat_id: Platform-specific chat/channel ID
            response: The UnifiedResponse to deliver

        Returns:
            RouteResult with success/failure status
        """
        t0 = time.time()
        adapter = self._adapters.get(platform)

        if not adapter:
            return RouteResult(
                adapter_name=platform,
                response=response,
                error=f"No adapter for platform: {platform}",
                latency_ms=(time.time() - t0) * 1000,
            )

        try:
            success = await asyncio.wait_for(
                adapter.send(chat_id, response), timeout=30
            )
            latency = (time.time() - t0) * 1000

            if platform in self._health:
                h = self._health[platform]
                h.messages_sent += 1
                h.last_activity = time.time()
                h.avg_latency_ms = h.avg_latency_ms * 0.9 + latency * 0.1
                if not success:
                    h.errors += 1

            return RouteResult(
                adapter_name=platform,
                response=response if success else None,
                error=None if success else f"Send failed for {platform}/{chat_id}",
                latency_ms=latency,
            )
        except asyncio.TimeoutError:
            if platform in self._health:
                self._health[platform].errors += 1
            return RouteResult(
                adapter_name=platform,
                response=None,
                error=f"Send timed out for {platform}/{chat_id}",
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            if platform in self._health:
                self._health[platform].errors += 1
            return RouteResult(
                adapter_name=platform,
                response=None,
                error=str(e),
                latency_ms=(time.time() - t0) * 1000,
            )

    # ── Health & monitoring ───────────────────────────────────────────────

    def health_status(self) -> dict[str, Any]:
        """Get health status for all adapters and the gateway itself."""
        adapter_healths = {}
        for name, h in self._health.items():
            adapter = self._adapters.get(name)
            if adapter:
                h.state = adapter.state
            adapter_healths[name] = h.to_dict()

        total_connected = sum(
            1 for h in adapter_healths.values()
            if h["state"] == "connected"
        )

        return {
            "gateway_uptime_s": round(time.time() - self._started_at, 1),
            "adapters_total": len(self._adapters),
            "adapters_connected": total_connected,
            "adapters": adapter_healths,
        }

    # ── Reconnect watcher ──────────────────────────────────────────────────

    def _record_reconnect_failure(self, name: str) -> None:
        """Initialize or escalate reconnect backoff for a failed adapter."""
        import random
        if name not in self._reconnect_tracker:
            tracker = _ReconnectState(
                name=name,
                failures=0,
                delay=RECONNECT_BASE_DELAY,
                next_attempt=0,
            )
            self._reconnect_tracker[name] = tracker
        tracker = self._reconnect_tracker[name]
        tracker.failures += 1
        tracker.delay = min(
            RECONNECT_BASE_DELAY * (RECONNECT_BACKOFF_FACTOR ** (tracker.failures - 1)),
            RECONNECT_MAX_DELAY,
        )
        jitter = tracker.delay * RECONNECT_JITTER
        tracker.delay = tracker.delay + random.uniform(-jitter, jitter)
        tracker.next_attempt = time.time() + tracker.delay

    def _start_watcher(self) -> None:
        """Start the background reconnect watcher if not already running."""
        if self._watcher_running:
            return
        self._watcher_running = True
        self._reconnect_task = asyncio.ensure_future(self._reconnect_watcher())
        logger.debug("Reconnect watcher started")

    def _stop_watcher(self) -> None:
        """Stop the background reconnect watcher."""
        self._watcher_running = False
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        self._reconnect_task = None
        logger.debug("Reconnect watcher stopped")

    async def _reconnect_watcher(self) -> None:
        """Background coroutine: attempts to reconnect disconnected adapters
        with exponential backoff."""
        while self._watcher_running:
            await asyncio.sleep(RECONNECT_CHECK_INTERVAL)
            if not self._watcher_running:
                break

            now = time.time()
            for name, tracker in list(self._reconnect_tracker.items()):
                if now < tracker.next_attempt:
                    continue  # Not time yet
                adapter = self._adapters.get(name)
                if not adapter:
                    self._reconnect_tracker.pop(name, None)
                    continue
                if adapter.state == ConnectionState.CONNECTED:
                    self._reconnect_tracker.pop(name, None)
                    continue

                logger.info(
                    "Reconnecting %s (attempt %d, delay=%.1fs)",
                    name, tracker.failures, tracker.delay,
                )
                try:
                    success = await asyncio.wait_for(adapter.connect(), timeout=10)
                    if success:
                        logger.info("Adapter %s reconnected", name)
                        self._reconnect_tracker.pop(name, None)
                        if self._health.get(name):
                            self._health[name].state = ConnectionState.CONNECTED
                            self._health[name].connected_at = time.time()
                    else:
                        self._record_reconnect_failure(name)
                except asyncio.TimeoutError:
                    self._record_reconnect_failure(name)
                    logger.warning("Adapter %s reconnect timed out", name)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._record_reconnect_failure(name)
                    logger.error("Adapter %s reconnect error: %s", name, e)

    def check_reconnects(self) -> dict[str, bool]:
        """Check reconnect tracker status. Returns per-adapter pending info."""
        return {
            name: time.time() >= t.next_attempt
            for name, t in self._reconnect_tracker.items()
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    @property
    def reconnect_pending(self) -> list[str]:
        """List of adapters pending reconnection."""
        return list(self._reconnect_tracker.keys())

    @property
    def reconnect_state(self) -> dict[str, dict]:
        """Get reconnect state for all tracked adapters."""
        return {
            name: {
                "failures": t.failures,
                "delay": round(t.delay, 1),
                "next_attempt": t.next_attempt,
                "seconds_until": max(0, round(t.next_attempt - time.time(), 1)),
            }
            for name, t in self._reconnect_tracker.items()
        }

    def __repr__(self) -> str:
        return (
            f"GatewayManager(adapters={self.adapter_count}, "
            f"platforms={self.platforms})"
        )
