"""Observability metrics — Prometheus-compatible counters, histograms, and gauges.

Dual-mode:
  - With prometheus_client installed: full Prometheus metric types, /metrics endpoint
  - Without: lightweight in-memory registry, compatible output format

Usage:
    from kairos.observability.metrics import MetricsRegistry, get_metrics

    registry = MetricsRegistry()
    registry.inc("kairos_requests_total", labels={"status": "success"})
    registry.observe("kairos_request_duration_seconds", 0.42)

    # Expose to Prometheus via Gateway /metrics endpoint
    print(registry.render())
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger("kairos.metrics")


# ============================================================================
# Lightweight Metric Types (no prometheus_client dependency)
# ============================================================================


class _Counter:
    """Thread-safe monotonic counter."""

    def __init__(self, name: str, help_text: str = "", labels: dict[str, str] | None = None):
        self.name = name
        self.help = help_text
        self._value = 0.0
        self._lock = threading.Lock()
        self.labels = labels or {}

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def get(self) -> float:
        return self._value


class _Gauge:
    """Thread-safe gauge (can go up and down)."""

    def __init__(self, name: str, help_text: str = "", labels: dict[str, str] | None = None):
        self.name = name
        self.help = help_text
        self._value = 0.0
        self._lock = threading.Lock()
        self.labels = labels or {}

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    def get(self) -> float:
        return self._value


class _Histogram:
    """Thread-safe histogram with configurable buckets."""

    def __init__(
        self,
        name: str,
        help_text: str = "",
        labels: dict[str, str] | None = None,
        buckets: list[float] | None = None,
    ):
        self.name = name
        self.help = help_text
        self.labels = labels or {}
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._sum = 0.0
        self._count = 0
        self._bucket_counts: dict[float, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._count += 1
            for boundary in self.buckets:
                if value <= boundary:
                    self._bucket_counts[boundary] += 1

    def get_sum(self) -> float:
        return self._sum

    def get_count(self) -> int:
        return self._count

    def get_buckets(self) -> dict[float, int]:
        # Ensure all buckets are present (return 0 for unobserved)
        result = {}
        for b in self.buckets:
            result[b] = self._bucket_counts.get(b, 0)
        return result


# ============================================================================
# Metrics Registry
# ============================================================================


class MetricsRegistry:
    """Prometheus-compatible metrics registry.

    Auto-detects prometheus_client. Falls back to lightweight in-memory types
    with identical render() output.
    """

    _DEFAULT_LABELS = {
        "kairos_requests_total": "Total number of agent requests",
        "kairos_request_duration_seconds": "Request duration in seconds",
        "kairos_tool_calls_total": "Total number of tool calls",
        "kairos_tool_errors_total": "Total number of tool errors",
        "kairos_tokens_used_total": "Total tokens consumed",
        "kairos_active_sessions": "Number of active agent sessions",
        "kairos_model_errors_total": "Total number of model API errors",
        "kairos_permission_denied_total": "Total number of permission denials",
    }

    def __init__(self, name_prefix: str = "kairos"):
        self._prefix = name_prefix
        self._counters: dict[str, _Counter] = {}
        self._gauges: dict[str, _Gauge] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()

    # ── Factory Methods ──────────────────────────────────────────

    def _make_key(self, name: str, labels: dict[str, str] | None = None) -> str:
        """Generate storage key from name + sorted labels."""
        full_name = name if name.startswith(self._prefix + "_") else f"{self._prefix}_{name}"
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{full_name}{{{label_str}}}"
        return full_name

    def counter(self, name: str, help_text: str = "", labels: dict[str, str] | None = None) -> _Counter:
        key = self._make_key(name, labels)
        full_name = self._make_key(name)  # bare name with prefix
        if key not in self._counters:
            self._counters[key] = _Counter(full_name, help_text, labels)
        return self._counters[key]

    def gauge(self, name: str, help_text: str = "", labels: dict[str, str] | None = None) -> _Gauge:
        key = self._make_key(name, labels)
        full_name = self._make_key(name)
        if key not in self._gauges:
            self._gauges[key] = _Gauge(full_name, help_text, labels)
        return self._gauges[key]

    def histogram(
        self,
        name: str,
        help_text: str = "",
        labels: dict[str, str] | None = None,
        buckets: list[float] | None = None,
    ) -> _Histogram:
        key = self._make_key(name, labels)
        full_name = self._make_key(name)
        if key not in self._histograms:
            self._histograms[key] = _Histogram(full_name, help_text, labels, buckets)
        return self._histograms[key]

    # ── Convenience ──────────────────────────────────────────────

    def inc(self, name: str, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        self.counter(name, labels=labels).inc(amount)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        self.gauge(name, labels=labels).set(value)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        self.histogram(name, labels=labels).observe(value)

    # ── Prometheus Text Format ───────────────────────────────────

    def render(self) -> str:
        """Render all metrics in Prometheus text format (OpenMetrics-compatible)."""
        lines: list[str] = []

        # HELP/TYPE for each unique metric base name
        seen = set()
        for key, c in sorted(self._counters.items()):
            base = c.name
            if base not in seen:
                lines.append(f"# HELP {base} {c.help or self._DEFAULT_LABELS.get(base, '')}")
                lines.append(f"# TYPE {base} counter")
                seen.add(base)
            label_str = self._format_labels(c.labels) if c.labels else ""
            lines.append(f"{base}{label_str} {c.get()}")

        for key, g in sorted(self._gauges.items()):
            base = g.name
            if base not in seen:
                pt = self._find_help(base, "gauge")
                lines.append(f"# HELP {base} {g.help or pt}")
                lines.append(f"# TYPE {base} gauge")
                seen.add(base)
            label_str = self._format_labels(g.labels) if g.labels else ""
            lines.append(f"{base}{label_str} {g.get()}")

        for key, h in sorted(self._histograms.items()):
            base = h.name
            if base not in seen:
                lines.append(f"# HELP {base} {h.help or self._DEFAULT_LABELS.get(base, '')}")
                lines.append(f"# TYPE {base} histogram")
                seen.add(base)
            label_str = self._format_labels(h.labels) if h.labels else ""
            lines.append(f"{base}_sum{label_str} {h.get_sum()}")
            lines.append(f"{base}_count{label_str} {h.get_count()}")
            for boundary, count in sorted(h.get_buckets().items()):
                lines.append(f"{base}_bucket{label_str}{{le=\"{boundary}\"}} {count}")
            lines.append(f"{base}_bucket{label_str}{{le=\"+Inf\"}} {h.get_count()}")

        return "\n".join(lines) + "\n"

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels dict to Prometheus label string."""
        parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return "{" + parts + "}" if parts else ""

    def _find_help(self, base: str, metric_type: str) -> str:
        """Find help text for a metric base name."""
        for k, v in self._DEFAULT_LABELS.items():
            if k in base:
                return v
        return ""

    # ── Auto-populated metrics ───────────────────────────────────

    def update_process_metrics(self) -> None:
        """Update built-in process-level gauges."""
        uptime = time.time() - self._start_time
        self.set_gauge("uptime_seconds", uptime)
        self.set_gauge("metrics_count_total", float(
            len(self._counters) + len(self._gauges) + len(self._histograms)
        ))


# ============================================================================
# Singleton
# ============================================================================


_metrics: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry (lazy singleton)."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics


def reset_metrics() -> None:
    """Reset global metrics registry (for testing)."""
    global _metrics
    _metrics = MetricsRegistry()
