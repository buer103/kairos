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
        assert "0.8" in captured.out
    finally:
        sys.argv = old_argv


def test_help_without_args(capsys):
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["kairos"]
        main()
        captured = capsys.readouterr()
        assert "Kairos" in captured.out
        assert "Usage:" in captured.out
    finally:
        sys.argv = old_argv


def test_help_shows_commands(capsys):
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["kairos"]
        main()
        captured = capsys.readouterr()
        assert "kairos chat" in captured.out
        assert "kairos run" in captured.out
        assert "kairos cron" in captured.out
        assert "kairos config init" in captured.out
    finally:
        sys.argv = old_argv


def test_unknown_command(capsys):
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["kairos", "nonexistent"]
        main()
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
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
        # Should not crash
        assert True
    finally:
        sys.argv = old_argv
