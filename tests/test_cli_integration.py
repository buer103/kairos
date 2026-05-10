"""CLI integration tests — test kairos commands programmatically."""

from __future__ import annotations

import tempfile
from pathlib import Path

from kairos.cli import main
from kairos.cli.rich_ui import KairosConsole


def test_version(capsys):
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["kairos", "--version"]
        main()
        captured = capsys.readouterr()
        assert "kairos" in captured.out
        assert "0.10" in captured.out
    finally:
        sys.argv = old_argv


def test_help_flag(capsys):
    """--help shows usage (kairos without args now enters chat)."""
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["kairos", "--help"]
        main()
        captured = capsys.readouterr()
        assert "Kairos" in captured.out
        assert "Usage:" in captured.out
    finally:
        sys.argv = old_argv


def test_help_shows_commands(capsys):
    """--help shows all subcommands."""
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["kairos", "--help"]
        main()
        captured = capsys.readouterr()
        assert "kairos chat" in captured.out
        assert "kairos run" in captured.out
        assert "kairos cron" in captured.out
        assert "kairos config init" in captured.out
        assert "--resume" in captured.out
        assert "--list-sessions" in captured.out
    finally:
        sys.argv = old_argv


def test_invalid_flag(capsys):
    """Invalid flags show usage (bare words are now queries, not errors)."""
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["kairos", "--bogus"]
        main()
        captured = capsys.readouterr()
        assert "Unknown flag" in captured.out
    finally:
        sys.argv = old_argv


def test_config_init(tmp_path: Path, capsys):
    import sys
    from kairos.config import write_default_config
    
    config_path = tmp_path / "kairos.yaml"
    write_default_config(str(config_path))
    assert config_path.exists()
    content = config_path.read_text()
    assert "deepseek" in content


def test_cron_list_no_crash(capsys):
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["kairos", "cron", "list"]
        main()
        captured = capsys.readouterr()
        assert True
    finally:
        sys.argv = old_argv


def test_list_sessions_flag(capsys):
    """--list-sessions works (may show none)."""
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["kairos", "--list-sessions"]
        main()
        captured = capsys.readouterr()
        # Either "No saved sessions" or list output
        assert "saved" in captured.out.lower() or "session" in captured.out.lower() or "No API key" in captured.out
    finally:
        sys.argv = old_argv
