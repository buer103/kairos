"""
Production-grade error classification and aggregation.

Extends :class:`kairos.core.loop.ErrorKind` and :func:`kairos.core.loop.classify_error`
with sliding-window aggregation, confidence scoring, root-cause analysis, and
threshold-based alerting.  Designed for zero external dependencies (stdlib only).

Typical usage::

    classifier = ErrorClassifier(window_seconds=300, alert_threshold=5)

    try:
        some_llm_call()
    except Exception as exc:
        classifier.record_error(exc, context={
            "provider": "deepseek", "model": "deepseek-v4", "tool": "search"
        })

    if classifier.should_alert():
        print(classifier.get_root_cause())
        print(classifier.get_error_breakdown())
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from kairos.core.loop import AgentError, ErrorKind, classify_error

logger = logging.getLogger("kairos.observability.error_classifier")


# ============================================================================
# Confidence scoring helpers
# ============================================================================

#: Keywords and substrings that boost confidence for each :class:`ErrorKind`.
_PATTERN_CONFIDENCE_BOOST: dict[ErrorKind, list[tuple[str, float]]] = {
    ErrorKind.RATE_LIMIT: [
        ("rate limit exceeded", 1.0),
        ("too many requests", 0.95),
        ("429", 0.9),
        ("quota exceeded", 0.85),
        ("try again later", 0.6),
    ],
    ErrorKind.AUTH: [
        ("invalid api key", 1.0),
        ("unauthorized", 0.95),
        ("authentication failed", 0.95),
        ("403", 0.85),
        ("401", 0.85),
        ("permission denied", 0.7),
    ],
    ErrorKind.NETWORK: [
        ("connection refused", 1.0),
        ("connection reset by peer", 1.0),
        ("timeout", 0.9),
        ("name resolution", 0.9),
        ("tls", 0.8),
        ("socket", 0.7),
        ("eof", 0.6),
    ],
    ErrorKind.CONTEXT_OVERFLOW: [
        ("context length exceeded", 1.0),
        ("maximum context", 0.95),
        ("token limit exceeded", 0.95),
        ("reduce the length", 0.85),
        ("too many tokens", 0.9),
    ],
    ErrorKind.TOOL_ERROR: [
        ("tool execution failed", 1.0),
        ("tool not found", 0.95),
        ("invalid tool arguments", 0.9),
        ("tool timeout", 0.9),
    ],
    ErrorKind.UNKNOWN: [],
}


def _compute_confidence(kind: ErrorKind, message: str) -> float:
    """Return a confidence score 0-1 for how likely *kind* is correct.

    Parameters
    ----------
    kind : ErrorKind
        The classified error kind.
    message : str
        The original error message (lowercased during processing).

    Returns
    -------
    float
        Confidence score from 0.0 (guess) to 1.0 (certain).
    """
    if kind == ErrorKind.UNKNOWN:
        # Unknown kind is inherently low-confidence
        return 0.2

    msg_lower = message.lower()
    best = 0.0
    for pattern, boost in _PATTERN_CONFIDENCE_BOOST.get(kind, []):
        if pattern in msg_lower:
            best = max(best, boost)
    # Base confidence for any classified error that didn't hit a pattern
    return max(best, 0.5)


# ============================================================================
# Data classes
# ============================================================================


@dataclass(slots=True)
class ErrorRecord:
    """A single classified error event stored in the sliding window.

    Attributes
    ----------
    timestamp : float
        Unix timestamp when the error was recorded.
    kind : ErrorKind
        Classified error category.
    message : str
        Error message string (truncated to 500 chars).
    confidence : float
        Classification confidence (0-1).
    provider : str | None
        LLM provider that triggered the error (e.g. ``"deepseek"``).
    model : str | None
        Model name that triggered the error (e.g. ``"deepseek-v4"``).
    tool : str | None
        Tool name if the error happened during a tool call.
    raw : Exception | dict | None
        The original error object (for debugging).
    """

    timestamp: float
    kind: ErrorKind
    message: str
    confidence: float
    provider: str | None = None
    model: str | None = None
    tool: str | None = None
    raw: Exception | dict | None = field(default=None, compare=False)


# ============================================================================
# ErrorClassifier
# ============================================================================


class ErrorClassifier:
    """Sliding-window error classifier with aggregation, root-cause analysis,
    and threshold-based alerting.

    Parameters
    ----------
    window_seconds : float
        Size of the sliding window in seconds (default 300 = 5 min).
    alert_threshold : int
        Number of errors inside the window that triggers an alert
        (default 5).
    max_window_entries : int
        Maximum number of error records to keep in the sliding window
        (default 10 000).
    """

    # ---- Public API --------------------------------------------------------

    def __init__(
        self,
        window_seconds: float = 300.0,
        alert_threshold: int = 5,
        max_window_entries: int = 10_000,
    ) -> None:
        self._window_seconds = float(window_seconds)
        self._alert_threshold = int(alert_threshold)
        self._max_entries = int(max_window_entries)

        #: Sliding window of error records (newest at right).
        self._window: deque[ErrorRecord] = deque()

        #: Total errors *ever* recorded (not pruned by the window).
        self._total_errors: int = 0

        #: First error timestamp (epoch) for computing overall error rate.
        self._first_error_at: float | None = None

        #: Accumulated breakdown counters (never pruned).
        self._breakdown: Counter[ErrorKind] = Counter()

        # Alert cooldown to prevent alert storms.
        self._last_alert_at: float = 0.0
        self._alert_cooldown_seconds: float = 60.0  # 1 min between alerts

    # ========================================================================
    # Recording
    # ========================================================================

    def record_error(
        self,
        error: Exception | dict,
        context: dict[str, Any] | None = None,
    ) -> ErrorRecord:
        """Classify and store an error with optional context.

        Parameters
        ----------
        error : Exception | dict
            The exception object or an error response dict.
        context : dict | None
            Optional metadata such as ``provider``, ``model``, ``tool``,
            ``iteration``, ``session_id``, etc.

        Returns
        -------
        ErrorRecord
            The classified and stored record.
        """
        ctx = context or {}

        # Classify using the core loop's function
        agent_error: AgentError = classify_error(error)

        # Compute confidence
        confidence = _compute_confidence(agent_error.kind, agent_error.message)

        # Truncate message
        message = agent_error.message[:500]

        record = ErrorRecord(
            timestamp=time.time(),
            kind=agent_error.kind,
            message=message,
            confidence=confidence,
            provider=ctx.get("provider"),
            model=ctx.get("model"),
            tool=ctx.get("tool"),
            raw=error,
        )

        self._add_to_window(record)
        logger.debug(
            "ErrorClassifier recorded [%s] conf=%.2f provider=%s model=%s tool=%s",
            record.kind.value,
            record.confidence,
            record.provider,
            record.model,
            record.tool,
        )

        return record

    # ========================================================================
    # Querying
    # ========================================================================

    def get_error_rate(self) -> float:
        """Compute the current error rate as errors per minute.

        Returns
        -------
        float
            Errors per minute over the sliding window.  Returns 0.0 if
            no window data is available.
        """
        self._prune_window()
        window_errors = len(self._window)
        if window_errors == 0:
            return 0.0
        # Use actual time span in the window (capped at window_seconds)
        if len(self._window) >= 2:
            span = self._window[-1].timestamp - self._window[0].timestamp
            span = max(span, 1.0)  # avoid division by zero
        else:
            span = 60.0  # single error: assume 1 min
        return (window_errors / span) * 60.0

    def get_error_breakdown(self) -> dict[ErrorKind, int]:
        """Return the number of active errors per :class:`ErrorKind` inside
        the current window.

        Returns
        -------
        dict[ErrorKind, int]
            Mapping from error kind to the count inside the sliding window.
        """
        self._prune_window()
        counter: Counter[ErrorKind] = Counter()
        for record in self._window:
            counter[record.kind] += 1
        return dict(counter)

    def get_total_breakdown(self) -> dict[ErrorKind, int]:
        """Return the total error breakdown since the classifier was created
        (or last :meth:`reset`).

        Returns
        -------
        dict[ErrorKind, int]
        """
        return dict(self._breakdown)

    def get_recent_errors(self, limit: int = 20) -> list[dict]:
        """Return the most recent errors (newest first) as a list of dicts.

        Parameters
        ----------
        limit : int
            Maximum number of records to return (default 20).

        Returns
        -------
        list[dict]
            Each dict contains ``timestamp``, ``kind``, ``message``,
            ``confidence``, ``provider``, ``model``, ``tool``.
        """
        self._prune_window()
        recent = list(self._window)[-limit:]
        recent.reverse()
        return [
            {
                "timestamp": r.timestamp,
                "kind": r.kind.value,
                "message": r.message,
                "confidence": r.confidence,
                "provider": r.provider,
                "model": r.model,
                "tool": r.tool,
            }
            for r in recent
        ]

    # ========================================================================
    # Alerting
    # ========================================================================

    def should_alert(self) -> bool:
        """Check whether the alert threshold has been exceeded inside the
        sliding window and the alert cooldown has elapsed.

        Returns
        -------
        bool
            ``True`` if an alert should be fired.
        """
        self._prune_window()
        window_count = len(self._window)
        if window_count < self._alert_threshold:
            return False

        # Cooldown check
        now = time.time()
        if now - self._last_alert_at < self._alert_cooldown_seconds:
            return False

        self._last_alert_at = now
        return True

    def get_alert_status(self) -> dict:
        """Return a summary of the current alert state.

        Returns
        -------
        dict
            Contains ``triggered``, ``window_count``, ``threshold``,
            ``window_seconds``, ``error_rate``, ``on_cooldown``.
        """
        self._prune_window()
        window_count = len(self._window)
        triggered = window_count >= self._alert_threshold
        on_cooldown = (
            time.time() - self._last_alert_at < self._alert_cooldown_seconds
        )
        return {
            "triggered": triggered,
            "window_count": window_count,
            "threshold": self._alert_threshold,
            "window_seconds": self._window_seconds,
            "error_rate": self.get_error_rate(),
            "on_cooldown": on_cooldown,
        }

    # ========================================================================
    # Root cause analysis
    # ========================================================================

    def get_root_cause(self) -> str:
        """Analyze the error window to identify the most probable root cause.

        Correlates errors by kind, provider, model, and tool to produce a
        human-readable root cause string.

        Returns
        -------
        str
            A single-line summary (or multi-line if patterns suggest
            multiple causes).
        """
        self._prune_window()
        if not self._window:
            return "No errors recorded."

        # 1. Dominant error kind
        kind_counter: Counter[ErrorKind] = Counter()
        for r in self._window:
            kind_counter[r.kind] += 1

        total = len(self._window)
        dominant_kind, dominant_count = kind_counter.most_common(1)[0]
        kind_pct = (dominant_count / total) * 100

        # 2. Dominant provider/model correlation (only for classified kinds)
        provider_counter: Counter[str] = Counter()
        model_counter: Counter[str] = Counter()
        tool_counter: Counter[str] = Counter()
        for r in self._window:
            if r.kind == dominant_kind:
                if r.provider:
                    provider_counter[r.provider] += 1
                if r.model:
                    model_counter[r.model] += 1
                if r.tool:
                    tool_counter[r.tool] += 1

        # 3. Assemble human-readable cause
        parts: list[str] = []

        kind_name = dominant_kind.value.replace("_", " ").title()

        parts.append(
            f"Dominant error: {kind_name} ({dominant_count}/{total}, "
            f"{kind_pct:.0f}%)"
        )

        if provider_counter:
            top_provider, prov_count = provider_counter.most_common(1)[0]
            parts.append(
                f" | Provider: {top_provider} ({prov_count}/{dominant_count})"
            )

        if model_counter:
            top_model, mod_count = model_counter.most_common(1)[0]
            parts.append(
                f" | Model: {top_model} ({mod_count}/{dominant_count})"
            )

        if tool_counter:
            top_tool, tool_count = tool_counter.most_common(1)[0]
            parts.append(
                f" | Tool: {top_tool} ({tool_count}/{dominant_count})"
            )

        # 4. Recent sample message
        if self._window:
            latest = self._window[-1]
            parts.append(f" | Sample: {latest.message[:120]}")

        # 5. Suggested action
        suggestion = self._suggest_action(dominant_kind)
        if suggestion:
            parts.append(f" | Suggestion: {suggestion}")

        return "".join(parts)

    def get_detail_report(self) -> dict:
        """Return a structured, machine-readable root-cause report.

        Returns
        -------
        dict
            Keys: ``total_errors``, ``window_errors``, ``dominant_kind``,
            ``breakdown``, ``top_provider``, ``top_model``, ``top_tool``,
            ``recent_sample``, ``suggestion``.
        """
        self._prune_window()
        result: dict[str, Any] = {
            "total_errors": self._total_errors,
            "window_errors": len(self._window),
            "dominant_kind": None,
            "breakdown": self.get_error_breakdown(),
            "top_provider": None,
            "top_model": None,
            "top_tool": None,
            "recent_sample": None,
            "suggestion": None,
        }

        if not self._window:
            return result

        kind_counter: Counter[ErrorKind] = Counter()
        for r in self._window:
            kind_counter[r.kind] += 1

        dominant_kind, _ = kind_counter.most_common(1)[0]
        result["dominant_kind"] = dominant_kind.value

        # Provider/model/tool correlation
        provider_counter: Counter[str] = Counter()
        model_counter: Counter[str] = Counter()
        tool_counter: Counter[str] = Counter()
        for r in self._window:
            if r.kind == dominant_kind:
                if r.provider:
                    provider_counter[r.provider] += 1
                if r.model:
                    model_counter[r.model] += 1
                if r.tool:
                    tool_counter[r.tool] += 1

        if provider_counter:
            result["top_provider"] = provider_counter.most_common(1)[0][0]
        if model_counter:
            result["top_model"] = model_counter.most_common(1)[0][0]
        if tool_counter:
            result["top_tool"] = tool_counter.most_common(1)[0][0]

        result["recent_sample"] = self._window[-1].message[:200]
        result["suggestion"] = self._suggest_action(dominant_kind)

        return result

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def reset(self) -> None:
        """Clear all error history, including the window and accumulated
        counters.
        """
        self._window.clear()
        self._total_errors = 0
        self._first_error_at = None
        self._breakdown.clear()
        self._last_alert_at = 0.0
        logger.info("ErrorClassifier reset — all history cleared.")

    # ========================================================================
    # Internals
    # ========================================================================

    def _add_to_window(self, record: ErrorRecord) -> None:
        """Append *record* and prune the window."""
        self._window.append(record)
        self._total_errors += 1
        self._breakdown[record.kind] += 1
        if self._first_error_at is None:
            self._first_error_at = record.timestamp
        # Enforce max entries
        while len(self._window) > self._max_entries:
            self._window.popleft()
        self._prune_window()

    def _prune_window(self) -> None:
        """Remove entries older than ``window_seconds``."""
        cutoff = time.time() - self._window_seconds
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()

    @staticmethod
    def _suggest_action(kind: ErrorKind) -> str:
        """Return a suggested remediation for a given :class:`ErrorKind`."""
        suggestions: dict[ErrorKind, str] = {
            ErrorKind.RATE_LIMIT: (
                "Wait for rate-limit window to expire; consider credential "
                "rotation or reducing request concurrency."
            ),
            ErrorKind.AUTH: (
                "Verify API keys and permissions; check for key rotation "
                "or expiration."
            ),
            ErrorKind.NETWORK: (
                "Check network connectivity, DNS resolution, and firewall "
                "rules; consider retry with exponential backoff."
            ),
            ErrorKind.CONTEXT_OVERFLOW: (
                "Reduce conversation history or enable context compression; "
                "consider switching to a larger-context model."
            ),
            ErrorKind.TOOL_ERROR: (
                "Check tool implementation and argument schemas; review "
                "tool timeout settings and error handling."
            ),
            ErrorKind.UNKNOWN: (
                "Inspect the raw error message; add a new classification "
                "pattern if this recurs."
            ),
        }
        return suggestions.get(kind, "Investigate manually.")


# ============================================================================
# Convenience: module-level singleton helper
# ============================================================================

_GLOBAL_CLASSIFIER: ErrorClassifier | None = None


def get_global_classifier(
    window_seconds: float = 300.0,
    alert_threshold: int = 5,
) -> ErrorClassifier:
    """Return (or lazily create) a module-level singleton :class:`ErrorClassifier`.

    Useful for simple setups where a single classifier is sufficient.
    """
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        _GLOBAL_CLASSIFIER = ErrorClassifier(
            window_seconds=window_seconds,
            alert_threshold=alert_threshold,
        )
    return _GLOBAL_CLASSIFIER
