"""LLM error handling — retry, jitter, 3-state circuit breaker, error classification.

Handles: rate limits (429), auth errors (401), server errors (5xx),
network errors, context length exceeded, malformed JSON tool args.

DeerFlow equivalent: LLMErrorHandlingMiddleware (368 lines)
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kairos.core.middleware import Middleware
from kairos.providers.credential import CredentialPool, RetryConfig

logger = logging.getLogger("kairos.middleware.llm_retry")


# ============================================================================
# Error Classification
# ============================================================================


class ErrorCategory(str, Enum):
    """DeerFlow-compatible error categories."""

    QUOTA = "quota"          # 余额不足 / quota exceeded
    AUTH = "auth"            # 鉴权失败
    TRANSIENT = "transient"  # 网络抖动 / 超时
    BUSY = "busy"            # 服务繁忙 / 过载
    GENERIC = "generic"      # 未知错误


@dataclass
class ClassifiedError:
    """An error with its category and whether it's retryable."""

    category: ErrorCategory
    message: str
    retryable: bool
    status_code: int | None = None


class ErrorClassifier:
    """Classifies LLM errors into DeerFlow-compatible categories with Chinese pattern support.

    Categories:
      - QUOTA:     balance exhausted, quota exceeded — NOT retryable
      - AUTH:      invalid credentials — NOT retryable
      - TRANSIENT: network timeout, connection errors — RETRYABLE
      - BUSY:      server overload, high load — RETRYABLE
      - GENERIC:   unknown — NOT retryable
    """

    # ── Chinese error patterns ───────────────────────────
    CN_QUOTA = [
        "余额不足", "额度不足", "额度已", "已用完", "quota exceeded", "insufficient quota",
        "账户欠费", "欠费", "无可用额度", "billing", "额度用", "quota",
    ]
    CN_AUTH = [
        "鉴权失败", "认证失败", "invalid api key", "unauthorized",
        "密钥无效", "token 无效", "key 无效",
    ]
    CN_BUSY = [
        "服务繁忙", "负载较高", "高负载", "服务繁忙请稍后",
        "server is busy", "overloaded", "too many requests",
        "rate limit", "请求过于频繁", "当前负载",
    ]
    CN_TRANSIENT = [
        "timeout", "超时", "connection", "网络",
        "reset by peer", "broken pipe", "connection refused",
        "service unavailable", "temporary failure",
        "暂时不可用", "connect timeout",
    ]

    @classmethod
    def classify(cls, error: Exception | str | dict) -> ClassifiedError:
        """Classify an error into a category."""
        msg = cls._extract_message(error)
        msg_lower = msg.lower()

        # Auth — must check first (unrecoverable)
        if cls._matches_any(msg_lower, cls.CN_AUTH):
            return ClassifiedError(ErrorCategory.AUTH, msg, False)

        # Quota — unrecoverable
        if cls._matches_any(msg_lower, cls.CN_QUOTA):
            return ClassifiedError(ErrorCategory.QUOTA, msg, False)

        # Busy — recoverable (retry with backoff)
        if cls._matches_any(msg_lower, cls.CN_BUSY):
            return ClassifiedError(ErrorCategory.BUSY, msg, True)

        # Transient — recoverable
        if cls._matches_any(msg_lower, cls.CN_TRANSIENT):
            return ClassifiedError(ErrorCategory.TRANSIENT, msg, True)

        # HTTP status-based fallback
        if isinstance(error, dict):
            status = error.get("status", error.get("status_code", 0))
            if status == 401 or status == 403:
                return ClassifiedError(ErrorCategory.AUTH, msg, False)
            if status == 402 or status == 429:
                return ClassifiedError(ErrorCategory.BUSY, msg, True)
            if 500 <= status < 600:
                return ClassifiedError(ErrorCategory.TRANSIENT, msg, True)

        return ClassifiedError(ErrorCategory.GENERIC, msg, False)

    @classmethod
    def _matches_any(cls, text: str, patterns: list[str]) -> bool:
        return any(p in text for p in patterns)

    @classmethod
    def _extract_message(cls, error: Exception | str | dict) -> str:
        if isinstance(error, str):
            return error
        if isinstance(error, dict):
            return str(error.get("error", "") or error.get("message", "") or error)
        return str(error)

    # ── User-facing messages ───────────────────────────

    USER_MESSAGES = {
        ErrorCategory.QUOTA: (
            "账户余额不足或配额已用完，请充值或切换 API key。"
        ),
        ErrorCategory.AUTH: (
            "API 密钥无效或已过期，请检查 config.yaml 中的 api_key。"
        ),
        ErrorCategory.BUSY: (
            "服务当前负载较高，正在自动重试…"
        ),
        ErrorCategory.TRANSIENT: (
            "网络连接临时异常，正在自动重试…"
        ),
        ErrorCategory.GENERIC: (
            "发生未知错误，请查看日志获取详情。"
        ),
    }

    @classmethod
    def user_message(cls, category: ErrorCategory) -> str:
        return cls.USER_MESSAGES.get(category, cls.USER_MESSAGES[ErrorCategory.GENERIC])


# ============================================================================
# 3-State Circuit Breaker
# ============================================================================


class CircuitState(str, Enum):
    CLOSED = "closed"        # Normal: pass calls through
    OPEN = "open"            # Fast-fail: reject immediately
    HALF_OPEN = "half_open"  # Probe: allow one test call


@dataclass
class CircuitBreaker:
    """3-state circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED.

    Thread-safe. DeerFlow-compatible.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds in OPEN before HALF_OPEN

    _failures: int = 0
    _last_failure: float = 0.0
    _state: CircuitState = CircuitState.CLOSED
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ── Transition helpers ─────────────────────────────

    def _transition_to(self, new_state: CircuitState, reason: str = "") -> None:
        old = self._state
        self._state = new_state
        if old != new_state:
            logger.warning("Circuit breaker: %s → %s%s", old.value, new_state.value,
                           f" ({reason})" if reason else "")

    # ── Public API ─────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """Current state. Automatically transitions OPEN→HALF_OPEN on timeout."""
        self._maybe_transition()
        return self._state

    def _maybe_transition(self) -> None:
        """Check if OPEN state should transition to HALF_OPEN (thread-safe)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure > self.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN, "recovery timeout")

    @property
    def is_open(self) -> bool:
        """True if circuit is OPEN (fast-fail). HALF_OPEN returns False."""
        self._maybe_transition()
        return self._state == CircuitState.OPEN

    def before_call(self) -> bool:
        """Check if a call should proceed. Returns False if circuit is OPEN."""
        self._maybe_transition()
        return self._state != CircuitState.OPEN

    def record_failure(self) -> None:
        """Record a failed call. May trip circuit to OPEN."""
        with self._lock:
            self._failures += 1
            self._last_failure = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed → back to OPEN
                self._transition_to(CircuitState.OPEN, "half-open probe failed")
                return

            if self._failures >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN, f"{self._failures} failures")

    def record_success(self) -> None:
        """Record a successful call. Resets circuit to CLOSED."""
        with self._lock:
            self._failures = 0
            if self._state != CircuitState.CLOSED:
                self._transition_to(CircuitState.CLOSED, "call succeeded")

    def reset(self) -> None:
        """Force-reset to CLOSED."""
        with self._lock:
            self._failures = 0
            self._state = CircuitState.CLOSED


# ============================================================================
# LLM Retry Middleware
# ============================================================================


class LLMRetryMiddleware(Middleware):
    """Retries failed LLM calls with jittered exponential backoff, credential rotation,
    classified error handling, and 3-state circuit breaker.

    Hook: wrap_model_call — intercepts every model call.
    """

    RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        credential_pool: CredentialPool | None = None,
        retry_config: RetryConfig | None = None,
        provider: str = "default",
        enable_circuit_breaker: bool = True,
        classifier: ErrorClassifier | None = None,
    ):
        self._pool = credential_pool or CredentialPool()
        self._config = retry_config or RetryConfig()
        self._provider = provider
        self._retry_count = 0
        self._last_error = ""
        self._last_category: ErrorCategory | None = None
        self._circuit = CircuitBreaker() if enable_circuit_breaker else None
        self._classifier = classifier or ErrorClassifier()

    def wrap_model_call(self, messages: list[dict], handler, **kwargs) -> Any:
        # ── Circuit breaker gate ────────────────────────
        if self._circuit and not self._circuit.before_call():
            self._last_category = ErrorCategory.TRANSIENT
            self._last_error = "Circuit breaker open — provider unavailable"
            return {"error": self._last_error, "retryable": False}

        last_exception = None
        credential = kwargs.get("credential")

        for attempt in range(self._config.max_retries + 1):
            credential = self._acquire_credential(credential)
            if credential is None and attempt > 0:
                break

            try:
                kwargs["credential"] = credential
                result = handler(messages, **kwargs)

                if credential:
                    self._pool.release(credential, success=True)

                if self._is_error_response(result):
                    status = self._get_status(result)
                    classified = self._classifier.classify(result)
                    self._last_category = classified.category

                    if status in self.RETRYABLE_STATUSES or classified.retryable:
                        self._handle_error(credential, status, result)
                        last_exception = Exception(f"HTTP {status}: {classified.message}")
                        if attempt < self._config.max_retries:
                            self._backoff(attempt)
                        continue
                    else:
                        # Non-retryable error → don't trip circuit, just return
                        self._last_error = classified.message
                        return result

                if self._circuit:
                    self._circuit.record_success()
                return result

            except Exception as e:
                classified = self._classifier.classify(e)
                self._last_category = classified.category

                if credential:
                    self._pool.release(credential, success=False)

                if not classified.retryable:
                    # Auth/quota errors → don't retry, don't trip circuit
                    self._last_error = classified.message
                    raise

                last_exception = e
                self._last_error = str(e)

                if self._circuit:
                    self._circuit.record_failure()

                if attempt < self._config.max_retries:
                    self._backoff(attempt)

        if self._circuit:
            self._circuit.record_failure()

        if last_exception:
            raise last_exception
        return {"error": "All retries exhausted", "retryable": False}

    # ── Credential management ──────────────────────────

    def _acquire_credential(self, preferred: Any = None) -> Any:
        if preferred and getattr(preferred, "available", False):
            return preferred
        return self._pool.acquire(self._provider)

    def _handle_error(self, credential: Any, status: int, result: Any) -> None:
        if status == 429:
            retry_after = 30.0
            if hasattr(result, "headers"):
                retry_after = float(result.headers.get("retry-after", 30))
            if credential:
                self._pool.mark_rate_limited(credential, retry_after)
            logger.warning("Rate limited, retry after %.0fs", retry_after)
        else:
            if credential:
                self._pool.release(credential, success=False)
            logger.warning("Server error %d, retrying", status)

    def _backoff(self, attempt: int) -> None:
        base = self._config.delay_for_attempt(attempt)
        jitter = base * random.uniform(0.75, 1.25)  # ±25% jitter
        time.sleep(jitter)
        self._retry_count += 1

    # ── Helpers ────────────────────────────────────────

    @staticmethod
    def _is_error_response(result: Any) -> bool:
        return isinstance(result, dict) and result.get("error") is not None

    @staticmethod
    def _get_status(result: Any) -> int:
        if isinstance(result, dict) and "error" in result:
            err = result["error"]
            if isinstance(err, dict):
                return err.get("status", err.get("status_code", 500))
        return 500

    # ── Properties ─────────────────────────────────────

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def last_category(self) -> ErrorCategory | None:
        return self._last_category

    @property
    def circuit_state(self) -> CircuitState | None:
        return self._circuit.state if self._circuit else None

    def user_friendly_error(self) -> str | None:
        """Return a user-friendly Chinese error message for the last error."""
        if self._last_category:
            return ErrorClassifier.user_message(self._last_category)
        return None

    def __repr__(self) -> str:
        circuit = f"circuit={self._circuit.state.value}" if self._circuit else "circuit=off"
        return f"LLMRetryMiddleware(retries={self._retry_count}, {circuit})"


# ============================================================================
# Tool Arg Repair Middleware
# ============================================================================


class ToolArgRepairMiddleware(Middleware):
    """Repairs malformed JSON in tool call arguments.

    Hook: wrap_tool_call — repairs args before tool execution.
    """

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        if isinstance(args, dict):
            return handler(tool_name, args, **kwargs)

        if isinstance(args, str):
            repaired = self._repair(args)
            if repaired is not None:
                logger.debug("Repaired tool args for '%s'", tool_name)
                return handler(tool_name, repaired, **kwargs)

        return handler(tool_name, args, **kwargs)

    @staticmethod
    def _repair(text: str) -> dict | None:
        strategies = [
            lambda t: json.loads(t),
            lambda t: json.loads(t.replace(",}", "}").replace(",]", "]")),
            lambda t: json.loads(t.replace("'", '"')),
            lambda t: json.loads(
                t.replace("True", "true").replace("False", "false").replace("None", "null")
            ),
            lambda t: json.loads("{" + t + "}"),
        ]
        for strategy in strategies:
            try:
                result = strategy(text)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return None

    def __repr__(self) -> str:
        return "ToolArgRepairMiddleware()"
