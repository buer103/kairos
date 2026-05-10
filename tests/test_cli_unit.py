"""Tests for Kairos CLI layer: KairosConsole (Rich TUI), SKINS, _parse_field, slash commands.

Focuses on testable logic rather than Rich rendering (which requires terminal).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from kairos.cli.rich_ui import KairosConsole, SKINS
from kairos.cli import _parse_field, _handle_slash, _print_usage


# ============================================================================
# SKINS
# ============================================================================

class TestSKINS:
    """Tests for the SKINS theme dictionary."""

    def test_all_skins_have_required_keys(self):
        required = {"agent_color", "user_color", "tool_color", "error_color", "info_color", "box", "spinner"}
        for name, skin in SKINS.items():
            for key in required:
                assert key in skin, f"Skin '{name}' missing key '{key}'"

    def test_four_skins_available(self):
        assert len(SKINS) == 4
        assert "default" in SKINS
        assert "hacker" in SKINS
        assert "retro" in SKINS
        assert "minimal" in SKINS


# ============================================================================
# KairosConsole — Initialization
# ============================================================================

class TestKairosConsoleInit:
    """Tests for KairosConsole constructor and properties."""

    def test_default_init(self):
        console = KairosConsole()
        assert console.skin_name == "default"
        assert console.skin == SKINS["default"]
        assert console.verbose is False
        assert console.stream is True
        assert console._history == []

    def test_custom_init(self):
        console = KairosConsole(skin="hacker", verbose=True, stream=False)
        assert console.skin_name == "hacker"
        assert console.skin == SKINS["hacker"]
        assert console.verbose is True

    def test_unknown_skin_falls_back_to_default(self):
        console = KairosConsole(skin="nonexistent")
        assert console.skin == SKINS["default"]

    def test_set_skin_valid(self):
        console = KairosConsole()
        result = console.set_skin("retro")
        assert result is True
        assert console.skin_name == "retro"

    def test_set_skin_invalid(self):
        console = KairosConsole()
        result = console.set_skin("nonexistent")
        assert result is False
        assert console.skin_name == "default"


# ============================================================================
# KairosConsole — History
# ============================================================================

class TestKairosConsoleHistory:
    """Tests for conversation history tracking."""

    def test_history_starts_empty(self):
        console = KairosConsole()
        assert console._history == []

    def test_agent_output_appends_history(self):
        console = KairosConsole()
        console.agent_output("Hello world", confidence=0.95)
        assert len(console._history) == 1
        assert console._history[0]["role"] == "agent"
        assert console._history[0]["content"] == "Hello world"
        assert console._history[0]["confidence"] == 0.95

    def test_user_input_appends_history(self):
        console = KairosConsole()
        console.user_input("What is 2+2?")
        assert len(console._history) == 1
        assert console._history[0]["role"] == "user"

    def test_tool_call_appends_history_when_verbose(self):
        console = KairosConsole(verbose=True)
        console.tool_call("read_file", {"path": "/tmp"}, "hello", duration_ms=150)
        assert len(console._history) == 1
        assert console._history[0]["role"] == "tool"
        assert console._history[0]["name"] == "read_file"

    def test_tool_call_skipped_when_not_verbose(self):
        console = KairosConsole(verbose=False)
        console.tool_call("read_file", {"path": "/tmp"}, "hello")
        assert console._history == []


# ============================================================================
# KairosConsole — Spinner
# ============================================================================

class TestKairosConsoleSpinner:
    """Tests for spinner lifecycle."""

    def test_spinner_start_and_stop(self):
        console = KairosConsole()
        assert console._spinner is None
        console.spinner_start("Loading...")
        assert console._spinner is not None
        assert console._live is not None
        console.spinner_stop()
        assert console._spinner is None
        assert console._live is None

    def test_spinner_update(self):
        console = KairosConsole()
        console.spinner_start("Thinking...")
        console.spinner_update("Almost done...")
        assert console._spinner.text == "Almost done..."


# ============================================================================
# _parse_field (Cron)
# ============================================================================

class TestParseField:
    """Tests for the cron field parser."""

    def test_wildcard(self):
        assert _parse_field("*") == []

    def test_step(self):
        result = _parse_field("*/15")
        assert result == [0, 15, 30, 45]

    def test_single_value(self):
        assert _parse_field("5") == [5]

    def test_range(self):
        result = _parse_field("1-5")
        assert result == [1, 2, 3, 4, 5]

    def test_list(self):
        result = _parse_field("1,3,5")
        assert result == [1, 3, 5]

    def test_mixed(self):
        result = _parse_field("1,3-5,10")
        assert result == [1, 3, 4, 5, 10]


# ============================================================================
# _handle_slash (Slash commands)
# ============================================================================

class TestHandleSlash:
    """Tests for the slash command dispatcher."""

    @pytest.fixture
    def console(self):
        c = KairosConsole()
        # Mock Rich console to suppress output
        c.console = MagicMock()
        return c

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.run.return_value = {"content": "result", "confidence": 0.9}
        return agent

    @pytest.fixture
    def mock_model_config(self):
        return MagicMock(model="test-model")

    def test_help(self, console, mock_agent, mock_model_config):
        _handle_slash(console, "/help", mock_agent, mock_model_config)

    def test_history(self, console, mock_agent, mock_model_config):
        console._history = [{"role": "user", "content": "hi"}]
        _handle_slash(console, "/history", mock_agent, mock_model_config)

    def test_clear(self, console, mock_agent, mock_model_config):
        console._history = [{"role": "user", "content": "hi"}]
        _handle_slash(console, "/clear", mock_agent, mock_model_config)
        assert console._history == []

    def test_verbose_toggle(self, console, mock_agent, mock_model_config):
        _handle_slash(console, "/verbose", mock_agent, mock_model_config)
        assert console.verbose is True
        _handle_slash(console, "/verbose", mock_agent, mock_model_config)
        assert console.verbose is False

    def test_model_get(self, console, mock_agent, mock_model_config):
        _handle_slash(console, "/model", mock_agent, mock_model_config)

    def test_model_set(self, console, mock_agent, mock_model_config):
        _handle_slash(console, "/model gpt-4", mock_agent, mock_model_config)
        assert mock_model_config.model == "gpt-4"

    def test_skin_get(self, console, mock_agent, mock_model_config):
        _handle_slash(console, "/skin", mock_agent, mock_model_config)

    def test_skin_set(self, console, mock_agent, mock_model_config):
        _handle_slash(console, "/skin hacker", mock_agent, mock_model_config)
        assert console.skin_name == "hacker"

    def test_unknown_command(self, console, mock_agent, mock_model_config):
        _handle_slash(console, "/nonexistent", mock_agent, mock_model_config)

    def test_quit_exits(self, console, mock_agent, mock_model_config):
        with pytest.raises(SystemExit):
            _handle_slash(console, "/exit", mock_agent, mock_model_config)


# ============================================================================
# _print_usage
# ============================================================================

class TestPrintUsage:
    """Tests for the usage print function."""

    def test_usage_does_not_crash(self, capsys):
        _print_usage()
        captured = capsys.readouterr()
        assert "Kairos" in captured.out
        assert "Usage:" in captured.out
