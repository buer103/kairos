"""Logging infrastructure — structured logging with file rotation.

Provides:
  - agent.log (INFO+) — full conversation traces
  - errors.log (WARNING+) — errors and warnings only
  - gateway.log — gateway-specific events
  - Structured log format with session/thread context
  - Automatic log directory creation and file rotation

Usage:
    from kairos.logging import get_logger
    log = get_logger("kairos.agent")
    log.info("Agent started", session_id="abc123")
    log.error("Tool failed", tool="rag_search", error=str(e))
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class KairosFormatter(logging.Formatter):
    """Structured log formatter with JSON-like output."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from record
        for key in ("session_id", "thread_id", "tool", "provider", "plugin"):
            val = getattr(record, key, None)
            if val:
                entry[key] = val

        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])

        return json.dumps(entry, ensure_ascii=False, default=str)


class KairosLogger:
    """Centralized logging manager for Kairos."""

    _instance: KairosLogger | None = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if KairosLogger._initialized:
            return
        KairosLogger._initialized = True

        self._log_dir = Path.home() / ".kairos" / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Agent log (INFO+)
        self.agent_log = self._setup_logger(
            "kairos.agent",
            self._log_dir / "agent.log",
            level=logging.INFO,
            max_bytes=10 * 1024 * 1024,
            backup_count=5,
        )

        # Error log (WARNING+)
        self.error_log = self._setup_logger(
            "kairos.errors",
            self._log_dir / "errors.log",
            level=logging.WARNING,
            max_bytes=5 * 1024 * 1024,
            backup_count=3,
        )

        # Gateway log
        self.gateway_log = self._setup_logger(
            "kairos.gateway",
            self._log_dir / "gateway.log",
            level=logging.INFO,
            max_bytes=10 * 1024 * 1024,
            backup_count=5,
        )

        # Console log (INFO+, stderr)
        self.console_handler = logging.StreamHandler(sys.stderr)
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(KairosFormatter())

        root = logging.getLogger("kairos")
        root.setLevel(logging.DEBUG)
        root.handlers.clear()

        if os.environ.get("KAIROS_DEBUG"):
            root.addHandler(self.console_handler)

    def _setup_logger(
        self,
        name: str,
        path: Path,
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        handler = logging.handlers.RotatingFileHandler(
            str(path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(KairosFormatter())
        logger.addHandler(handler)

        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific component."""
        return logging.getLogger(f"kairos.{name}")

    def enable_console(self) -> None:
        """Enable console logging for debugging."""
        root = logging.getLogger("kairos")
        if self.console_handler not in root.handlers:
            root.addHandler(self.console_handler)

    def disable_console(self) -> None:
        root = logging.getLogger("kairos")
        if self.console_handler in root.handlers:
            root.removeHandler(self.console_handler)

    def tail(self, log_name: str = "agent", lines: int = 20) -> str:
        """Get the last N lines of a log file."""
        log_map = {
            "agent": self._log_dir / "agent.log",
            "errors": self._log_dir / "errors.log",
            "gateway": self._log_dir / "gateway.log",
        }
        path = log_map.get(log_name)
        if not path or not path.exists():
            return ""

        with path.open("r", encoding="utf-8") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])

    def log_event(
        self,
        level: str,
        message: str,
        logger: str = "agent",
        **extra,
    ) -> None:
        """Log a structured event with extra fields."""
        log = self.get_logger(logger)
        extra_fields = {}
        for k, v in extra.items():
            extra_fields[k] = v

        # Attach extra fields to a record-like object
        record = logging.LogRecord(
            name=f"kairos.{logger}",
            level=getattr(logging, level.upper(), logging.INFO),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        for k, v in extra_fields.items():
            setattr(record, k, v)

        log.handle(record)


def get_logger(name: str = "agent") -> logging.Logger:
    """Convenience: get a Kairos logger."""
    KairosLogger()  # Ensure initialized
    return logging.getLogger(f"kairos.{name}")


def log_agent_event(
    level: str,
    message: str,
    session_id: str = "",
    **extra,
) -> None:
    """Log an agent lifecycle event."""
    KairosLogger().log_event(level, message, logger="agent", session_id=session_id, **extra)


def log_tool_call(
    tool_name: str,
    args: dict,
    result: Any,
    duration_ms: float,
    session_id: str = "",
) -> None:
    """Log a tool invocation."""
    KairosLogger().log_event(
        "info",
        f"Tool: {tool_name}({_truncate(str(args), 200)}) → {duration_ms:.1f}ms",
        logger="agent",
        session_id=session_id,
        tool=tool_name,
    )


def log_error(
    message: str,
    exception: Exception | None = None,
    session_id: str = "",
    **extra,
) -> None:
    """Log an error with optional exception."""
    log = KairosLogger().error_log
    if exception:
        log.error(message, exc_info=exception, extra={"session_id": session_id, **extra} if extra else {})
    else:
        log.error(message)


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s
