"""
Token usage and cost tracking with time-bucketed storage.

Tracks every LLM API call (provider, model, tokens, duration, success) and
provides aggregated metrics — total cost, calls-per-minute, average latency,
success rate, and daily stats.  Pricing is configurable per model.

Storage strategy
----------------
* **Granular bucket** — keeps individual call records for the last hour
  (default *granular_seconds* = 3600).
* **Aggregate bucket** — older calls are rolled into hourly summary buckets
  for long-term daily statistics.

Design
------
Zero external dependencies (stdlib only).  All time-series data lives in
plain Python lists/dicts.  Use :class:`UsageTracker` standalone or composed
inside :class:`AgentInsights`.

Typical usage::

    tracker = UsageTracker()

    tracker.track_call(
        provider="deepseek",
        model="deepseek-v4-pro",
        prompt_tokens=1200,
        completion_tokens=500,
        duration_ms=2340.0,
        success=True,
    )

    print(f"Total cost: ${tracker.total_cost:.4f}")
    print(f"Success rate: {tracker.success_rate:.1%}")
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("kairos.observability.usage_tracker")


# ============================================================================
# Pricing table  (input / output  per 1 000 000 tokens)
# ============================================================================

#: Default pricing in USD per 1M tokens: ``(input_price, output_price)``.
#:
#: Provider prefixes are used as a fallback when the exact model key is not
#: found.  The lookup order is: exact model key → provider prefix → ``"default"``.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-v3": (0.27, 1.10),
    "deepseek-v4": (0.27, 1.10),
    "deepseek-v4-pro": (0.27, 1.10),
    "deepseek": (0.27, 1.10),  # provider fallback
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "openai": (2.50, 10.00),  # provider fallback
    # Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-sonnet": (3.00, 15.00),
    "claude": (3.00, 15.00),  # provider fallback
    "anthropic": (3.00, 15.00),  # provider fallback (alt)
    # Google
    "gemini-2.5-pro": (2.50, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-pro": (1.25, 5.00),
    "gemini-flash": (0.075, 0.30),
    "gemini": (1.25, 5.00),  # provider fallback
    "google": (1.25, 5.00),  # provider fallback (alt)
    # Fallback
    "default": (1.00, 4.00),
}


def _resolve_price(
    model: str,
    provider: str | None = None,
    pricing: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Look up ``(input_price, output_price)`` per 1M tokens.

    Resolution order:
    1. Exact match on *model* key.
    2. Match on *provider* key.
    3. ``"default"`` key.
    """
    table = pricing if pricing is not None else DEFAULT_PRICING

    model_key = model.lower().strip()
    provider_key = (provider or "").lower().strip()

    # 1. Exact model
    if model_key in table:
        return table[model_key]

    # 2. Provider fallback
    if provider_key and provider_key in table:
        return table[provider_key]

    # 3. Try substring match on model name against known families
    for key, price in table.items():
        if key in ("default",):
            continue
        if key in model_key or model_key.startswith(key):
            return price

    return table.get("default", (1.00, 4.00))


# ============================================================================
# Data classes
# ============================================================================


@dataclass(slots=True)
class CallRecord:
    """A single tracked API call.

    Attributes
    ----------
    timestamp : float
        Unix epoch when the call completed.
    provider : str
        Provider name (e.g. ``"deepseek"``, ``"openai"``).
    model : str
        Model identifier (e.g. ``"deepseek-v4-pro"``).
    prompt_tokens : int
        Tokens consumed by the input/prompt.
    completion_tokens : int
        Tokens generated in the output.
    duration_ms : float
        Wall-clock duration of the call in milliseconds.
    success : bool
        ``True`` if the call completed without error.
    """

    timestamp: float
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    success: bool


@dataclass
class HourlyBucket:
    """Aggregated stats for calls in a one-hour window.

    Used for older (archived) data to keep memory bounded.
    """

    hour_start: float  # Unix timestamp snapped to the hour
    call_count: int = 0
    success_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_duration_ms: float = 0.0
    total_cost: float = 0.0

    # Per-model sub-breakdown within this hour
    model_calls: Counter[str] = field(default_factory=lambda: Counter())  # type: ignore[valid-type]


# ============================================================================
# UsageTracker
# ============================================================================


class UsageTracker:
    """Track LLM API usage with time-bucketed storage and cost calculation.

    Parameters
    ----------
    granular_seconds : float
        How long to keep per-call records (default 3600 = 1 hour).
        Older calls are rolled into :class:`HourlyBucket` aggregates.
    max_granular_entries : int
        Maximum number of per-call records in the granular window
        (default 100 000).
    pricing : dict[str, tuple[float, float]] | None
        Custom pricing table.  If ``None``, uses :data:`DEFAULT_PRICING`.
        Each entry is ``(input_price, output_price)`` per 1M tokens.
    """

    def __init__(
        self,
        granular_seconds: float = 3600.0,
        max_granular_entries: int = 100_000,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._granular_seconds = float(granular_seconds)
        self._max_granular = int(max_granular_entries)
        self._pricing = pricing or DEFAULT_PRICING

        # Granular (per-call) window
        self._calls: deque[CallRecord] = deque()

        # Hourly aggregates for older data
        self._hourly: list[HourlyBucket] = []

        # Totals (never expire)
        self._total_calls: int = 0
        self._total_success: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_duration_ms: float = 0.0
        self._total_cost: float = 0.0
        self._first_call_at: float | None = None

        self._last_prune: float = time.time()
        self._prune_interval: float = 60.0  # seconds between prunes

    # ========================================================================
    # Recording
    # ========================================================================

    def track_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
        success: bool,
    ) -> CallRecord:
        """Record an API call and update all aggregates.

        Parameters
        ----------
        provider : str
            Provider name (e.g. ``"deepseek"``).
        model : str
            Model name (e.g. ``"deepseek-v4-pro"``).
        prompt_tokens : int
            Tokens in the prompt.
        completion_tokens : int
            Tokens in the completion.
        duration_ms : float
            Call duration in milliseconds.
        success : bool
            Whether the call succeeded.

        Returns
        -------
        CallRecord
            The stored record.
        """
        record = CallRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            duration_ms=float(duration_ms),
            success=bool(success),
        )

        # Append to granular window
        self._calls.append(record)
        while len(self._calls) > self._max_granular:
            self._archive_oldest()

        # Update totals
        self._total_calls += 1
        if success:
            self._total_success += 1
        self._total_prompt_tokens += record.prompt_tokens
        self._total_completion_tokens += record.completion_tokens
        self._total_duration_ms += record.duration_ms

        # Cost
        cost = self._compute_cost(record)
        self._total_cost += cost

        if self._first_call_at is None:
            self._first_call_at = record.timestamp

        # Periodic maintenance
        if time.time() - self._last_prune > self._prune_interval:
            self._maintenance()

        return record

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion) across all calls."""
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def total_cost(self) -> float:
        """Total estimated cost in USD across all calls."""
        return self._total_cost

    @property
    def calls_per_minute(self) -> float:
        """Calls per minute over the granular window."""
        self._maintenance()
        window_calls = len(self._calls)
        if window_calls < 2:
            if window_calls == 1:
                return 1.0 / max(self._granular_seconds / 60.0, 1.0)
            return 0.0
        span = self._calls[-1].timestamp - self._calls[0].timestamp
        span = max(span, 1.0)
        return (window_calls / span) * 60.0

    @property
    def average_latency_ms(self) -> float:
        """Average call duration in milliseconds across all calls."""
        if self._total_calls == 0:
            return 0.0
        return self._total_duration_ms / self._total_calls

    @property
    def success_rate(self) -> float:
        """Fraction of successful calls (0.0–1.0)."""
        if self._total_calls == 0:
            return 1.0
        return self._total_success / self._total_calls

    # ========================================================================
    # Daily stats
    # ========================================================================

    def get_daily_stats(self) -> dict[str, Any]:
        """Return aggregated statistics for the current UTC day.

        Combines granular window data + hourly bucket data, filtering to
        calls made since midnight UTC.

        Returns
        -------
        dict
            Keys: ``date``, ``total_calls``, ``success_calls``, ``failed_calls``,
            ``success_rate``, ``total_tokens``, ``prompt_tokens``,
            ``completion_tokens``, ``total_cost``, ``avg_latency_ms``,
            ``models_used``.
        """
        self._maintenance()

        midnight = _utc_midnight()

        calls_today = 0
        success_today = 0
        prompt_today = 0
        completion_today = 0
        duration_today = 0.0
        cost_today = 0.0
        models_used: Counter[str] = Counter()

        # Granular window (only calls from today)
        for r in self._calls:
            if r.timestamp >= midnight:
                calls_today += 1
                if r.success:
                    success_today += 1
                prompt_today += r.prompt_tokens
                completion_today += r.completion_tokens
                duration_today += r.duration_ms
                cost_today += self._compute_cost(r)
                models_used[r.model] += 1

        # Hourly buckets
        for bucket in self._hourly:
            if bucket.hour_start >= midnight - 3600:
                # Bucket may span midnight; approximate by including full bucket
                # that starts after midnight or straddles it
                pass
            if bucket.hour_start >= midnight:
                calls_today += bucket.call_count
                success_today += bucket.success_count
                prompt_today += bucket.total_prompt_tokens
                completion_today += bucket.total_completion_tokens
                duration_today += bucket.total_duration_ms
                cost_today += bucket.total_cost
                for model, count in bucket.model_calls.items():
                    models_used[model] += count

        total_tokens_today = prompt_today + completion_today
        avg_latency = (duration_today / calls_today) if calls_today > 0 else 0.0
        sr = (success_today / calls_today) if calls_today > 0 else 1.0

        from datetime import datetime, timezone

        return {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_calls": calls_today,
            "success_calls": success_today,
            "failed_calls": calls_today - success_today,
            "success_rate": round(sr, 4),
            "total_tokens": total_tokens_today,
            "prompt_tokens": prompt_today,
            "completion_tokens": completion_today,
            "total_cost": round(cost_today, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "models_used": dict(models_used.most_common()),
        }

    # ========================================================================
    # Snapshot / reset
    # ========================================================================

    def get_snapshot(self) -> dict[str, Any]:
        """Return a comprehensive snapshot of current state.

        Returns
        -------
        dict
            All tracked metrics.
        """
        self._maintenance()
        return {
            "total_calls": self._total_calls,
            "total_success": self._total_success,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_cost": round(self._total_cost, 6),
            "calls_per_minute": round(self.calls_per_minute, 4),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "granular_window_calls": len(self._calls),
            "hourly_buckets": len(self._hourly),
        }

    def reset(self) -> None:
        """Clear all history."""
        self._calls.clear()
        self._hourly.clear()
        self._total_calls = 0
        self._total_success = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_duration_ms = 0.0
        self._total_cost = 0.0
        self._first_call_at = None
        logger.info("UsageTracker reset — all history cleared.")

    # ========================================================================
    # Pricing helpers
    # ========================================================================

    def get_price(self, model: str, provider: str | None = None) -> tuple[float, float]:
        """Look up ``(input_price, output_price)`` per 1M tokens for a model.

        Parameters
        ----------
        model : str
            Model name.
        provider : str | None
            Provider name for fallback lookup.

        Returns
        -------
        tuple[float, float]
            ``(input_price_per_1M, output_price_per_1M)`` in USD.
        """
        return _resolve_price(model, provider, self._pricing)

    @classmethod
    def with_custom_pricing(
        cls,
        pricing: dict[str, tuple[float, float]],
        **kwargs: Any,
    ) -> UsageTracker:
        """Factory that creates a tracker with merged pricing.

        The *pricing* dict is layered on top of :data:`DEFAULT_PRICING` so
        you only need to specify overrides.

        Parameters
        ----------
        pricing : dict[str, tuple[float, float]]
            Pricing overrides/additions.
        **kwargs
            Forwarded to :meth:`__init__`.
        """
        merged = {**DEFAULT_PRICING, **pricing}
        return cls(pricing=merged, **kwargs)

    # ========================================================================
    # Internals
    # ========================================================================

    def _compute_cost(self, record: CallRecord) -> float:
        """Compute cost in USD for a single :class:`CallRecord`."""
        in_price, out_price = _resolve_price(record.model, record.provider, self._pricing)
        input_cost = (record.prompt_tokens / 1_000_000) * in_price
        output_cost = (record.completion_tokens / 1_000_000) * out_price
        return input_cost + output_cost

    def _maintenance(self) -> None:
        """Run periodic upkeep: prune old granular records, archive to hourly
        buckets.
        """
        self._last_prune = time.time()
        cutoff = time.time() - self._granular_seconds

        while self._calls and self._calls[0].timestamp < cutoff:
            self._archive_oldest()

    def _archive_oldest(self) -> None:
        """Move the oldest granular call into an hourly bucket."""
        if not self._calls:
            return
        record = self._calls.popleft()
        hour_start = _snap_to_hour(record.timestamp)

        # Find or create bucket
        bucket: HourlyBucket | None = None
        for b in self._hourly:
            if b.hour_start == hour_start:
                bucket = b
                break
        if bucket is None:
            bucket = HourlyBucket(hour_start=hour_start)
            self._hourly.append(bucket)

        # Merge
        bucket.call_count += 1
        if record.success:
            bucket.success_count += 1
        bucket.total_prompt_tokens += record.prompt_tokens
        bucket.total_completion_tokens += record.completion_tokens
        bucket.total_duration_ms += record.duration_ms
        bucket.total_cost += self._compute_cost(record)
        bucket.model_calls[record.model] += 1

        # Limit hourly buckets to prevent unbounded growth (keep ~90 days)
        MAX_HOURLY_BUCKETS = 90 * 24  # 90 days * 24 hours
        while len(self._hourly) > MAX_HOURLY_BUCKETS:
            self._hourly.pop(0)


# ============================================================================
# Helpers
# ============================================================================

from collections import Counter  # noqa: E402 (keep imports grouped)


def _snap_to_hour(ts: float) -> float:
    """Snap a Unix timestamp down to the start of its UTC hour."""
    return ts - (ts % 3600)


def _utc_midnight() -> float:
    """Return the Unix timestamp for midnight UTC of the current day."""
    now = time.time()
    return now - (now % 86400)
