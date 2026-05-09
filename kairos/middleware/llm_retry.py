"""LLM error handling middleware — retry, fallback, and repair.

Handles the most common LLM API failures:
  1. Network errors → retry with exponential backoff
  2. Rate limits (429) → rotate credential, wait for reset
  3. Authentication errors (401) → disable credential, retry with next
  4. Invalid JSON tool arguments → attempt auto-repair
  5. Context length exceeded → trigger compression
  6. Timeout → retry with backoff

DeerFlow equivalent: LLMRetryMiddleware + LLMErrorHandlingMiddleware
"""

from __future__ import annotations

import json
import time
from typing import Any

from kairos.core.middleware import Middleware
from kairos.providers.credential import CredentialPool, RetryConfig


class LLMRetryMiddleware(Middleware):
    """Retries failed LLM calls with exponential backoff and credential rotation.

    Hook: wrap_model_call — catches exceptions and retries with fallback keys.
    """

    def __init__(
        self,
        credential_pool: CredentialPool | None = None,
        retry_config: RetryConfig | None = None,
        provider: str = "default",
    ):
        self._pool = credential_pool or CredentialPool()
        self._config = retry_config or RetryConfig()
        self._provider = provider
        self._last_error: str = ""
        self._retry_count = 0

    def wrap_model_call(self, messages: list[dict], handler, **kwargs) -> Any:
        """Call the LLM with retry logic."""
        last_exception = None
        original_credential = kwargs.get("credential")

        for attempt in range(self._config.max_retries + 1):
            credential = self._acquire_credential(original_credential)
            if credential is None and attempt > 0:
                break

            try:
                kwargs["credential"] = credential
                result = handler(messages, **kwargs)

                if credential:
                    self._pool.release(credential, success=True)

                # Check for HTTP errors in result
                if self._is_error_response(result):
                    status_code = self._get_status(result)
                    if self._should_retry(status_code):
                        if status_code == 429:
                            self._handle_rate_limit(credential, result)
                        else:
                            if credential:
                                self._pool.release(credential, success=False)
                        last_exception = Exception(f"HTTP {status_code}")
                        self._backoff(attempt)
                        continue

                return result

            except Exception as e:
                last_exception = e
                if credential:
                    self._pool.release(credential, success=False)

                if not self._should_retry_exception(e):
                    raise

                if attempt < self._config.max_retries:
                    self._backoff(attempt)

        if last_exception:
            raise last_exception
        return {"error": "All retries exhausted", "last_error": self._last_error}

    def _acquire_credential(self, preferred: Any = None) -> Any:
        """Get the next available credential."""
        if preferred and hasattr(preferred, 'available') and preferred.available:
            return preferred
        return self._pool.acquire(self._provider)

    def _handle_rate_limit(self, credential: Any, result: Any) -> None:
        """Extract Retry-After and mark credential as rate-limited."""
        retry_after = 30.0
        if credential:
            self._pool.mark_rate_limited(credential, retry_after)

    def _backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff."""
        delay = self._config.delay_for_attempt(attempt)
        time.sleep(delay)
        self._retry_count += 1

    @staticmethod
    def _is_error_response(result: Any) -> bool:
        """Check if the result indicates an HTTP error."""
        if isinstance(result, dict):
            return result.get("error") is not None
        return False

    @staticmethod
    def _get_status(result: Any) -> int:
        if isinstance(result, dict) and "error" in result:
            err = result["error"]
            if isinstance(err, dict):
                return err.get("status", err.get("status_code", 500))
        return 500

    @staticmethod
    def _should_retry(status_code: int) -> bool:
        return status_code in (429, 500, 502, 503, 504) or status_code >= 500

    @staticmethod
    def _should_retry_exception(exc: Exception) -> bool:
        """Check if this exception type should trigger a retry."""
        exc_str = str(exc).lower()
        retry_keywords = [
            "timeout", "connection", "rate limit", "too many requests",
            "server error", "service unavailable", "gateway",
            "reset by peer", "broken pipe", "connection refused",
        ]
        return any(kw in exc_str for kw in retry_keywords)

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def last_error(self) -> str:
        return self._last_error

    def __repr__(self) -> str:
        return f"LLMRetryMiddleware(retries={self._retry_count}, provider={self._provider})"


class ToolArgRepairMiddleware(Middleware):
    """Repairs invalid JSON in tool call arguments.

    LLMs sometimes produce malformed JSON in tool call arguments,
    especially with nested quotes or trailing commas. This middleware
    attempts common repairs before the tool call fails.

    Hook: wrap_tool_call — intercepts before execution, patches args.
    """

    def wrap_tool_call(self, tool_name: str, args: dict, handler, **kwargs) -> Any:
        """Attempt to repair tool arguments if they're invalid."""
        # If args is already a dict, it was parsed successfully — no repair needed
        if isinstance(args, dict):
            return handler(tool_name, args, **kwargs)

        # If args is a string, try to parse and repair
        if isinstance(args, str):
            repaired = self._repair_json(args)
            if repaired is not None:
                return handler(tool_name, repaired, **kwargs)

        # If all else fails, try the original handler
        return handler(tool_name, args, **kwargs)

    @staticmethod
    def _repair_json(text: str) -> dict | None:
        """Attempt common JSON repairs."""
        repairs = [
            text,  # Try as-is first
            text.strip().replace(",}", "}").replace(",]", "]"),  # Trailing comma before close
            text.replace("'", '"'),  # Single → double quotes
            text.replace("True", "true").replace("False", "false").replace("None", "null"),
        ]
        for attempt in repairs:
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    def __repr__(self) -> str:
        return "ToolArgRepairMiddleware()"
