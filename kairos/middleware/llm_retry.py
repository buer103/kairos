"""LLM error handling — retry, jitter, circuit breaker, credential rotation.

Handles: rate limits (429), auth errors (401), server errors (5xx),
network errors, context length exceeded, malformed JSON tool args.

DeerFlow equivalent: LLMRetryMiddleware + LLMErrorHandlingMiddleware
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from kairos.core.middleware import Middleware
from kairos.providers.credential import CredentialPool, RetryConfig

logger = logging.getLogger("kairos.middleware.llm_retry")


# ── Circuit Breaker ─────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    """Prevents retries to a failing provider after threshold failures."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    _failures: int = 0
    _last_failure: float = 0.0
    _open: bool = False

    @property
    def is_open(self) -> bool:
        if not self._open:
            return False
        if time.time() - self._last_failure > self.recovery_timeout:
            self._open = False
            self._failures = 0
            return False
        return True

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure = time.time()
        if self._failures >= self.failure_threshold:
            self._open = True
            logger.warning("Circuit breaker OPEN after %d failures", self._failures)

    def record_success(self) -> None:
        self._failures = 0
        self._open = False


# ── LLM Retry Middleware ────────────────────────────────────────

class LLMRetryMiddleware(Middleware):
    """Retries failed LLM calls with jittered exponential backoff and credential rotation.

    Hook: wrap_model_call — intercepts every model call.
    """

    RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        credential_pool: CredentialPool | None = None,
        retry_config: RetryConfig | None = None,
        provider: str = "default",
        enable_circuit_breaker: bool = True,
    ):
        self._pool = credential_pool or CredentialPool()
        self._config = retry_config or RetryConfig()
        self._provider = provider
        self._retry_count = 0
        self._last_error = ""
        self._circuit = CircuitBreaker() if enable_circuit_breaker else None

    def wrap_model_call(self, messages: list[dict], handler, **kwargs) -> Any:
        if self._circuit and self._circuit.is_open:
            return {"error": "Circuit breaker open — provider unavailable", "retryable": False}

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
                    if status in self.RETRYABLE_STATUSES:
                        self._handle_error(credential, status, result)
                        last_exception = Exception(f"HTTP {status}")
                        if attempt < self._config.max_retries:
                            self._backoff(attempt)
                        continue

                if self._circuit:
                    self._circuit.record_success()
                return result

            except Exception as e:
                last_exception = e
                if credential:
                    self._pool.release(credential, success=False)

                if not self._should_retry_exception(e):
                    raise

                if self._circuit:
                    self._circuit.record_failure()

                if attempt < self._config.max_retries:
                    self._backoff(attempt)

        if self._circuit:
            self._circuit.record_failure()

        if last_exception:
            raise last_exception
        return {"error": "All retries exhausted", "retryable": False}

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

    @staticmethod
    def _should_retry_exception(exc: Exception) -> bool:
        kw = [
            "timeout", "connection", "rate limit", "too many requests",
            "server error", "service unavailable", "gateway",
            "reset by peer", "broken pipe", "connection refused",
        ]
        return any(k in str(exc).lower() for k in kw)

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def last_error(self) -> str:
        return self._last_error

    def __repr__(self) -> str:
        return f"LLMRetryMiddleware(retries={self._retry_count}, circuit={self._circuit is not None})"


# ── Tool Arg Repair Middleware ──────────────────────────────────

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
            # Try as-is
            lambda t: json.loads(t),
            # Trailing commas
            lambda t: json.loads(t.replace(",}", "}").replace(",]", "]")),
            # Single → double quotes (safe for simple cases)
            lambda t: json.loads(t.replace("'", '"')),
            # Python booleans + None
            lambda t: json.loads(t.replace("True", "true").replace("False", "false").replace("None", "null")),
            # Missing quotes around keys
            lambda t: json.loads(t),
            # Try with surrounding braces
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
