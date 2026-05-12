"""Tests for finish_task tool."""
from __future__ import annotations

import pytest

from kairos.tools.builtin import finish_task
from kairos.tools.registry import get_tool, execute_tool


def test_finish_task_success():
    result = finish_task(summary="All files created", status="success")
    assert result["completed"] is True
    assert result["summary"] == "All files created"
    assert result["status"] == "success"
    assert result["action"] == "stop"


def test_finish_task_partial():
    result = finish_task(summary="3/5 tests pass", status="partial")
    assert result["completed"] is True
    assert result["status"] == "partial"


def test_finish_task_failed():
    result = finish_task(summary="Build failed", status="failed")
    assert result["completed"] is True
    assert result["status"] == "failed"


def test_finish_task_empty_summary():
    result = finish_task()
    assert result["summary"] == ""


def test_finish_task_truncates_long_summary():
    long_text = "x" * 600
    result = finish_task(summary=long_text)
    assert len(result["summary"]) == 500


def test_finish_task_registered_as_tool():
    tool = get_tool("finish_task")
    assert tool is not None
    assert tool["category"] == "control"
    assert tool["enabled"] is True


def test_finish_task_via_execute():
    result = execute_tool("finish_task", {"summary": "Done", "status": "success"})
    assert result["completed"] is True
    assert result["summary"] == "Done"
