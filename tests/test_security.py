"""Tests for the security layer and SecurityMiddleware."""

import pytest

from kairos.security.file_safety import FileSafetyChecker
from kairos.security.path_security import PathSecurity
from kairos.security.url_safety import URLSafety
from kairos.security.content_redact import ContentRedactor
from kairos.security.guardrails import InputGuard, OutputGuard, ToolGuard
from kairos.middleware.security_mw import SecurityMiddleware, SecurityViolation


# ============================================================================
# FileSafetyChecker
# ============================================================================


class TestFileSafetyChecker:
    def test_safe_file(self):
        fs = FileSafetyChecker()
        ok, reason = fs.is_safe("/tmp/test.txt", "path")
        assert ok or "size" in reason

    def test_dangerous_extension(self):
        fs = FileSafetyChecker()
        ok, reason = fs.is_safe("/tmp/malware.exe", "path")
        assert not ok
        assert "blocked" in reason

    def test_dangerous_script(self):
        fs = FileSafetyChecker()
        ok, reason = fs.is_safe("/tmp/script.sh", "path")
        assert not ok

    def test_sanitize_filename(self):
        fs = FileSafetyChecker()
        assert fs.sanitize_filename("normal.txt") == "normal.txt"
        result = fs.sanitize_filename("../../../etc/passwd")
        assert "../" not in result


# ============================================================================
# PathSecurity
# ============================================================================


class TestPathSecurity:
    def test_allowed_path(self):
        ps = PathSecurity(allowed_roots=["/tmp", "/home"])
        assert ps.is_path_allowed("/tmp/test.txt")

    def test_blocked_root(self):
        ps = PathSecurity(allowed_roots=["/tmp", "/home"])
        assert not ps.is_path_allowed("/etc/passwd")

    def test_path_traversal_blocked(self):
        ps = PathSecurity(allowed_roots=["/tmp"])
        assert not ps.is_path_allowed("/tmp/../../../etc/passwd")

    def test_symlink_detection(self):
        ps = PathSecurity(allowed_roots=["/tmp"])
        assert not ps.is_path_allowed("/proc/self/fd/0")

    def test_guard_read_ok(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        ps = PathSecurity(allowed_roots=[str(tmp_path)])
        result = ps.guard_read(str(f))
        assert result is not None


# ============================================================================
# URLSafety
# ============================================================================


class TestURLSafety:
    def test_safe_url(self):
        us = URLSafety()
        ok, reason = us.check_url("https://google.com")
        assert ok

    def test_file_scheme_blocked(self):
        us = URLSafety()
        ok, reason = us.check_url("file:///etc/passwd")
        assert not ok
        assert "blocked" in reason.lower()

    def test_localhost_blocked(self):
        us = URLSafety()
        ok, reason = us.check_url("http://localhost:8080/api")
        assert not ok

    def test_internal_ip_blocked(self):
        us = URLSafety()
        ok, reason = us.check_url("http://192.168.1.1/admin")
        assert not ok

    def test_internal_ip_10_blocked(self):
        us = URLSafety()
        ok, reason = us.check_url("http://10.0.0.1/api")
        assert not ok


# ============================================================================
# ContentRedactor
# ============================================================================


class TestContentRedactor:
    def test_redact_api_key(self):
        """ContentRedactor catches OpenAI key patterns with 20+ chars after sk-."""
        cr = ContentRedactor()
        # 20 chars: "a" * 20
        suffix = "a" * 20
        fake_key = "sk-" + suffix
        result = cr.redact("my key is " + fake_key)
        assert "[REDACTED]" in result
        assert fake_key not in result

    def test_redact_github_token(self):
        """ContentRedactor catches GitHub classic tokens (ghp_ + 36+ chars)."""
        cr = ContentRedactor()
        # 36 chars
        suffix = "A" * 36
        fake_token = "ghp_" + suffix
        result = cr.redact("token: " + fake_token)
        assert "[REDACTED]" in result

    def test_preserve_normal_text(self):
        cr = ContentRedactor()
        result = cr.redact("hello world")
        assert "hello world" in result

    def test_redact_dict_sensitive_keys(self):
        cr = ContentRedactor()
        data = {"name": "test", "password": "secret123"}
        result = cr.redact_dict(data)
        assert result["name"] == "test"
        assert result["password"] == "[REDACTED]"


# ============================================================================
# InputGuard
# ============================================================================


class TestInputGuard:
    def test_valid_input(self):
        ig = InputGuard()
        ok, reason = ig.validate_input("hello world")
        assert ok

    def test_too_long_input(self):
        ig = InputGuard(max_length=100)
        ok, reason = ig.validate_input("x" * 100000)
        assert not ok

    def test_long_but_ok(self):
        ig = InputGuard(max_length=1000)
        ok, reason = ig.validate_input("hello")
        assert ok


# ============================================================================
# OutputGuard
# ============================================================================


class TestOutputGuard:
    def test_valid_output(self):
        og = OutputGuard()
        ok, reason = og.validate_output("the answer is 42")
        assert ok

    def test_api_key_leak_detected(self):
        """OutputGuard catches 20+ char sk- tokens."""
        og = OutputGuard()
        suffix = "A" * 20
        fake_key = "sk-" + suffix
        ok, reason = og.validate_output("use this key: " + fake_key)
        assert not ok


# ============================================================================
# ToolGuard
# ============================================================================


class TestToolGuard:
    def test_valid_read_file(self):
        tg = ToolGuard()
        ok, reason = tg.validate_tool_args("read_file", {"path": "/tmp/test.txt"})
        assert ok

    def test_path_traversal(self):
        tg = ToolGuard()
        ok, reason = tg.validate_tool_args("read_file", {"path": "../../../etc/passwd"})
        assert not ok

    def test_valid_url_fetch(self):
        tg = ToolGuard()
        ok, reason = tg.validate_tool_args("web_fetch", {"url": "https://example.com"})
        assert ok

    def test_traversal_in_workdir(self):
        tg = ToolGuard()
        ok, reason = tg.validate_tool_args("terminal", {"command": "ls", "workdir": "../../../etc"})
        assert not ok

    def test_empty_args(self):
        tg = ToolGuard()
        ok, reason = tg.validate_tool_args("some_tool", {})
        assert ok


# ============================================================================
# SecurityMiddleware integration
# ============================================================================


class TestSecurityMiddleware:
    def test_import(self):
        mw = SecurityMiddleware(allowed_paths=["/tmp"])
        assert mw is not None

    def test_allowed_paths(self):
        mw = SecurityMiddleware(allowed_paths=["/tmp", "/var/tmp"])
        assert mw._path_security.is_path_allowed("/tmp/foo")

    def test_security_violation_exception(self):
        with pytest.raises(SecurityViolation):
            raise SecurityViolation("test violation")
