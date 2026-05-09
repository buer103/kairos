"""
Agent health insights — combines :class:`ErrorClassifier` and :class:`UsageTracker`
to produce comprehensive health reports, efficiency scoring, anomaly detection,
and session summaries.

Designed to be wired into :meth:`kairos.core.loop.Agent.health_status` for
rich observability with minimal overhead.

Typical usage::

    from kairos.observability import AgentInsights

    insights = AgentInsights()

    # After each LLM call:
    insights.usage.track_call(...)

    # After each error:
    insights.errors.record_error(exc, context={...})

    # Periodic health check:
    report = insights.get_health_report()
    if insights.detect_anomalies():
        logger.warning("Anomalies detected: %s", insights.detect_anomalies())

    # End-of-session summary:
    summary = insights.get_session_summary()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from kairos.observability.error_classifier import ErrorClassifier
from kairos.observability.usage_tracker import UsageTracker

logger = logging.getLogger("kairos.observability.insights")


# ============================================================================
# Anomaly detection thresholds
# ============================================================================

@dataclass
class AnomalyConfig:
    """Thresholds that trigger anomaly alerts.

    Attributes
    ----------
    error_spike_threshold : float
        Errors-per-minute above this value triggers an error-spike anomaly.
    latency_spike_multiplier : float
        If average latency exceeds *latency_spike_multiplier* × the running
        mean, a latency anomaly is triggered.
    cost_spike_threshold : float
        Total cost per minute above this value triggers a cost anomaly.
    success_rate_drop : float
        If success rate drops below this fraction (0–1), a reliability
        anomaly is triggered.
    """

    error_spike_threshold: float = 10.0  # errors/minute
    latency_spike_multiplier: float = 3.0  # × baseline
    cost_spike_threshold: float = 0.10  # USD/minute
    success_rate_drop: float = 0.80  # 80%


# ============================================================================
# AgentInsights
# ============================================================================


class AgentInsights:
    """Combined observability: error classification + usage tracking +
    anomaly detection.

    Parameters
    ----------
    error_classifier : ErrorClassifier | None
        Pre-configured classifier.  Created with defaults if ``None``.
    usage_tracker : UsageTracker | None
        Pre-configured tracker.  Created with defaults if ``None``.
    anomaly_config : AnomalyConfig | None
        Threshold overrides for anomaly detection.
    session_id : str | None
        Identifier for the current session (used in session summaries).
    """

    def __init__(
        self,
        error_classifier: ErrorClassifier | None = None,
        usage_tracker: UsageTracker | None = None,
        anomaly_config: AnomalyConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self.errors = error_classifier or ErrorClassifier()
        self.usage = usage_tracker or UsageTracker()
        self.anomaly_config = anomaly_config or AnomalyConfig()
        self._session_id = session_id or f"session_{int(time.time())}"
        self._session_start = time.time()

        # Running baseline for anomaly detection
        self._baseline_latency: float = 0.0
        self._baseline_samples: int = 0

    # ========================================================================
    # Health report
    # ========================================================================

    def get_health_report(self) -> dict[str, Any]:
        """Return a comprehensive health status dictionary.

        Suitable for wiring into :meth:`Agent.health_status` or for
        exposing via a monitoring endpoint.

        Returns
        -------
        dict
            Keys: ``status``, ``errors``, ``usage``, ``anomalies``,
            ``session``, ``timestamp``.
        """
        anomalies = self.detect_anomalies()

        error_breakdown = self.errors.get_error_breakdown()
        alert_status = self.errors.get_alert_status()

        snapshot = self.usage.get_snapshot()

        overall_status = "degraded" if anomalies else "healthy"

        return {
            "status": overall_status,
            "errors": {
                "rate": round(self.errors.get_error_rate(), 4),
                "window_count": alert_status["window_count"],
                "alert_triggered": alert_status["triggered"],
                "breakdown": {k.value: v for k, v in error_breakdown.items()},
                "total_errors": self.errors._total_errors,
            },
            "usage": snapshot,
            "anomalies": anomalies,
            "session": {
                "id": self._session_id,
                "uptime_seconds": round(time.time() - self._session_start, 1),
            },
            "timestamp": time.time(),
        }

    # ========================================================================
    # Efficiency score
    # ========================================================================

    def get_efficiency_score(self) -> float:
        """Compute a 0–1 efficiency score based on success rate, token
        efficiency, and latency.

        The score is a weighted combination:

        * **Success rate** (40%) — higher is better.
        * **Token efficiency** (35%) — tokens per call relative to a
          reference (fewer tokens = more efficient).
        * **Latency factor** (25%) — scaled relative to a 2000 ms baseline
          (lower latency = better).

        Returns
        -------
        float
            Score from 0.0 (worst) to 1.0 (best).
        """
        sr = self.usage.success_rate
        total_calls = self.usage._total_calls

        # Token efficiency: tokens-per-call vs reference of 4000 tokens
        if total_calls > 0:
            tokens_per_call = self.usage.total_tokens / total_calls
        else:
            tokens_per_call = 0.0
        token_ref = 4000.0  # reference tokens per call
        token_score = max(0.0, 1.0 - (tokens_per_call / token_ref))
        token_score = min(token_score, 1.0)

        # Latency: scale against a 2000 ms reference
        avg_lat = self.usage.average_latency_ms
        latency_ref = 2000.0
        latency_score = max(0.0, 1.0 - (avg_lat / latency_ref))
        latency_score = min(latency_score, 1.0)

        score = (0.40 * sr) + (0.35 * token_score) + (0.25 * latency_score)
        return round(max(0.0, min(score, 1.0)), 4)

    # ========================================================================
    # Anomaly detection
    # ========================================================================

    def detect_anomalies(self) -> list[str]:
        """Scan current metrics for anomalies and return human-readable
        descriptions.

        Detects:
        * Sudden error rate spikes
        * Latency increases relative to baseline
        * Cost anomalies
        * Success rate degradation

        Returns
        -------
        list[str]
            Descriptions of detected anomalies (empty if none).
        """
        cfg = self.anomaly_config
        alerts: list[str] = []

        # --- Error spike ---
        error_rate = self.errors.get_error_rate()
        if error_rate > cfg.error_spike_threshold:
            alerts.append(
                f"Error spike: {error_rate:.1f} errors/min "
                f"(threshold {cfg.error_spike_threshold:.1f}/min)"
            )

        # --- Latency spike ---
        avg_lat = self.usage.average_latency_ms
        # Update baseline
        if self._baseline_samples == 0 and avg_lat > 0:
            self._baseline_latency = avg_lat
            self._baseline_samples = 1
        elif avg_lat > 0:
            # Exponential moving average baseline
            alpha = 0.1
            self._baseline_latency = (
                alpha * avg_lat + (1 - alpha) * self._baseline_latency
            )
            self._baseline_samples += 1

        if (
            self._baseline_latency > 0
            and self._baseline_samples > 5
            and avg_lat > self._baseline_latency * cfg.latency_spike_multiplier
        ):
            alerts.append(
                f"Latency spike: {avg_lat:.0f}ms vs baseline "
                f"{self._baseline_latency:.0f}ms "
                f"({avg_lat / self._baseline_latency:.1f}×)"
            )

        # --- Cost anomaly ---
        cpm = self.usage.calls_per_minute
        if cpm > 0:
            avg_cost_per_call = (
                self.usage.total_cost / self.usage._total_calls
                if self.usage._total_calls > 0
                else 0.0
            )
            cost_per_minute = avg_cost_per_call * cpm
            if cost_per_minute > cfg.cost_spike_threshold:
                alerts.append(
                    f"Cost anomaly: ${cost_per_minute:.4f}/min "
                    f"(threshold ${cfg.cost_spike_threshold:.4f}/min)"
                )

        # --- Success rate drop ---
        sr = self.usage.success_rate
        if self.usage._total_calls >= 5 and sr < cfg.success_rate_drop:
            alerts.append(
                f"Reliability drop: success rate {sr:.1%} "
                f"(threshold {cfg.success_rate_drop:.1%})"
            )

        return alerts

    # ========================================================================
    # Session summary
    # ========================================================================

    def get_session_summary(self) -> dict[str, Any]:
        """Return a summary of the entire session suitable for persistence
        or display.

        Returns
        -------
        dict
            Keys: ``session_id``, ``duration_seconds``, ``total_errors``,
            ``error_breakdown``, ``total_calls``, ``success_rate``,
            ``total_tokens``, ``total_cost``, ``efficiency_score``,
            ``anomalies_detected``, ``root_cause``.
        """
        error_breakdown = {
            k.value: v for k, v in self.errors.get_total_breakdown().items()
        }
        return {
            "session_id": self._session_id,
            "duration_seconds": round(time.time() - self._session_start, 1),
            "total_errors": self.errors._total_errors,
            "error_breakdown": error_breakdown,
            "total_calls": self.usage._total_calls,
            "success_rate": round(self.usage.success_rate, 4),
            "total_tokens": self.usage.total_tokens,
            "total_cost": round(self.usage.total_cost, 6),
            "efficiency_score": self.get_efficiency_score(),
            "anomalies_detected": self.detect_anomalies(),
            "root_cause": self.errors.get_root_cause(),
        }

    # ========================================================================
    # Convenience
    # ========================================================================

    def record_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
        success: bool,
        error: Exception | dict | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed LLM call with optional error in one method.

        Convenience wrapper that calls both :meth:`UsageTracker.track_call`
        and (if *error* is given) :meth:`ErrorClassifier.record_error`.

        Parameters
        ----------
        provider : str
            Provider name.
        model : str
            Model name.
        prompt_tokens : int
            Prompt token count.
        completion_tokens : int
            Completion token count.
        duration_ms : float
            Call duration in milliseconds.
        success : bool
            Whether the call succeeded.
        error : Exception | dict | None
            The error, if the call failed.
        error_context : dict | None
            Additional context for error classification.
        """
        self.usage.track_call(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            success=success,
        )
        if error is not None:
            ctx: dict[str, Any] = {
                "provider": provider,
                "model": model,
            }
            if error_context:
                ctx.update(error_context)
            self.errors.record_error(error, context=ctx)

    def reset(self) -> None:
        """Reset both error and usage history."""
        self.errors.reset()
        self.usage.reset()
        self._baseline_latency = 0.0
        self._baseline_samples = 0
        self._session_start = time.time()
        logger.info("AgentInsights reset — new session started.")
