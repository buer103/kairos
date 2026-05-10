"""Tests for the 3-level interactive permission system.

Covers: PermissionLevel, ToolPolicy, PermissionManager, PermissionRequest,
        SecurityMiddleware permission integration.
"""

from __future__ import annotations

import asyncio

import pytest

from kairos.security.permission import (
    PermissionAction,
    PermissionLevel,
    PermissionManager,
    PermissionRequest,
    ToolPolicy,
)
from kairos.middleware.security_mw import SecurityMiddleware, SecurityViolation


# ============================================================================
# ToolPolicy
# ============================================================================


class TestToolPolicy:
    def test_exact_match(self):
        p = ToolPolicy("write_file")
        assert p.matches("write_file")
        assert not p.matches("read_file")

    def test_wildcard_match(self):
        p = ToolPolicy("web_*")
        assert p.matches("web_search")
        assert p.matches("web_fetch")
        assert p.matches("web_scrape")
        assert not p.matches("file_read")

    def test_glob_star_match(self):
        p = ToolPolicy("*")
        assert p.matches("anything")
        assert p.matches("write_file")

    def test_prefix_wildcard(self):
        p = ToolPolicy("*_file")
        assert p.matches("write_file")
        assert p.matches("read_file")
        assert not p.matches("terminal")


# ============================================================================
# PermissionLevel
# ============================================================================


class TestPermissionLevel:
    def test_enum_values(self):
        assert PermissionLevel.BLOCK.value == "block"
        assert PermissionLevel.ASK.value == "ask"
        assert PermissionLevel.TRUST.value == "trust"

    def test_from_string(self):
        assert PermissionLevel("block") == PermissionLevel.BLOCK
        assert PermissionLevel("ask") == PermissionLevel.ASK
        assert PermissionLevel("trust") == PermissionLevel.TRUST


# ============================================================================
# PermissionManager — basic
# ============================================================================


@pytest.fixture
def pm():
    """PermissionManager with auto-approve OFF."""
    return PermissionManager(auto_approve=False)


@pytest.fixture
def pm_auto():
    """PermissionManager with auto-approve ON."""
    return PermissionManager(auto_approve=True)


class TestPermissionManagerDefaults:
    def test_default_policies_registered(self, pm):
        """Built-in defaults should cover key tools."""
        assert pm.get_effective_level("read_file", "s1") == PermissionLevel.TRUST
        assert pm.get_effective_level("search_files", "s1") == PermissionLevel.TRUST
        assert pm.get_effective_level("write_file", "s1") == PermissionLevel.ASK
        assert pm.get_effective_level("terminal", "s1") == PermissionLevel.ASK
        assert pm.get_effective_level("patch", "s1") == PermissionLevel.ASK
        assert pm.get_effective_level("cronjob", "s1") == PermissionLevel.BLOCK

    def test_unmatched_uses_default(self, pm):
        """Unknown tools get the default policy."""
        assert pm.get_effective_level("custom_tool", "s1") == PermissionLevel.ASK


# ============================================================================
# PermissionManager — policy management
# ============================================================================


class TestPermissionManagerPolicies:
    def test_set_policy_overrides_default(self, pm):
        pm.set_policy(ToolPolicy("write_file", level=PermissionLevel.TRUST))
        assert pm.get_effective_level("write_file", "s1") == PermissionLevel.TRUST

    def test_set_policy_wildcard(self, pm):
        pm.set_policy(ToolPolicy("internal_*", level=PermissionLevel.TRUST))
        assert pm.get_effective_level("internal_audit", "s1") == PermissionLevel.TRUST
        # Other tools unchanged
        assert pm.get_effective_level("write_file", "s1") == PermissionLevel.ASK

    def test_set_default_level(self, pm):
        pm.set_default_level(PermissionLevel.TRUST)
        assert pm.get_effective_level("unknown_tool", "s1") == PermissionLevel.TRUST

    def test_session_level_override(self, pm):
        """Session-level override takes precedence over tool policies."""
        pm.set_session_level("s1", PermissionLevel.TRUST)
        # Even cronjob which is BLOCK by default
        assert pm.get_effective_level("cronjob", "s1") == PermissionLevel.TRUST
        # Other sessions unaffected
        assert pm.get_effective_level("cronjob", "s2") == PermissionLevel.BLOCK

    def test_clear_session_level(self, pm):
        pm.set_session_level("s1", PermissionLevel.TRUST)
        pm.clear_session_grants("s1")
        assert pm.get_effective_level("cronjob", "s1") == PermissionLevel.BLOCK


# ============================================================================
# PermissionManager — requests
# ============================================================================


class TestPermissionManagerRequests:
    @pytest.mark.asyncio
    async def test_trust_auto_approves(self, pm):
        pm.set_policy(ToolPolicy("read_file", level=PermissionLevel.TRUST))
        req = PermissionRequest(session_id="s1", tool_name="read_file")
        assert await pm.request(req) is True

    @pytest.mark.asyncio
    async def test_block_auto_denies(self, pm):
        pm.set_policy(ToolPolicy("dangerous", level=PermissionLevel.BLOCK))
        req = PermissionRequest(session_id="s1", tool_name="dangerous")
        assert await pm.request(req) is False

    @pytest.mark.asyncio
    async def test_auto_approve_mode(self, pm_auto):
        """Auto-approve grants ASK-level tools without callback."""
        # Auto-approve overrides ASK → trust
        req = PermissionRequest(session_id="s1", tool_name="write_file",
                                description="Write test")
        assert await pm_auto.request(req) is True  # ASK → auto-approved

    @pytest.mark.asyncio
    async def test_auto_approve_respects_block(self, pm_auto):
        """Auto-approve does NOT override BLOCK (safety first)."""
        pm_auto.set_policy(ToolPolicy("dangerous", level=PermissionLevel.BLOCK))
        req = PermissionRequest(session_id="s1", tool_name="dangerous")
        assert await pm_auto.request(req) is False  # BLOCK still blocks

    @pytest.mark.asyncio
    async def test_ask_without_callback_denies(self, pm):
        """Without a prompt callback, ASK defaults to DENY."""
        req = PermissionRequest(session_id="s1", tool_name="write_file",
                                description="Write foo.txt", path="/tmp/foo.txt")
        assert await pm.request(req) is False

    @pytest.mark.asyncio
    async def test_ask_with_callback_allows(self, pm):
        """Callback returning ALLOW_ONCE should grant."""
        async def _allow(_req):
            return PermissionAction.ALLOW_ONCE
        pm.set_prompt_callback(_allow)
        req = PermissionRequest(session_id="s1", tool_name="write_file")
        assert await pm.request(req) is True

    @pytest.mark.asyncio
    async def test_ask_with_callback_denies(self, pm):
        """Callback returning DENY should block."""
        async def _deny(_req):
            return PermissionAction.DENY
        pm.set_prompt_callback(_deny)
        req = PermissionRequest(session_id="s1", tool_name="write_file")
        assert await pm.request(req) is False

    @pytest.mark.asyncio
    async def test_session_grant_remembered(self, pm):
        """ALLOW_SESSION should remember for subsequent calls."""
        allowed = {"first": True}

        async def _session_first(_req):
            if allowed["first"]:
                allowed["first"] = False
                return PermissionAction.ALLOW_SESSION
            return PermissionAction.DENY

        pm.set_prompt_callback(_session_first)

        req1 = PermissionRequest(session_id="s1", tool_name="write_file", path="/tmp/a.txt")
        assert await pm.request(req1) is True  # first: asked, session-granted

        req2 = PermissionRequest(session_id="s1", tool_name="write_file", path="/tmp/a.txt")
        assert await pm.request(req2) is True  # second: remembered, no ask

    @pytest.mark.asyncio
    async def test_session_grant_per_path(self, pm):
        """Session grants are per (tool, path). Different path = re-ask."""
        call_count = {"n": 0}

        async def _count(_req):
            call_count["n"] += 1
            return PermissionAction.ALLOW_SESSION

        pm.set_prompt_callback(_count)

        req1 = PermissionRequest(session_id="s1", tool_name="write_file", path="/tmp/a.txt")
        assert await pm.request(req1) is True
        assert call_count["n"] == 1  # asked once

        req2 = PermissionRequest(session_id="s1", tool_name="write_file", path="/tmp/b.txt")
        assert await pm.request(req2) is True
        assert call_count["n"] == 2  # different path, asked again

        req3 = PermissionRequest(session_id="s1", tool_name="write_file", path="/tmp/a.txt")
        assert await pm.request(req3) is True
        assert call_count["n"] == 2  # same path, remembered

    @pytest.mark.asyncio
    async def test_different_sessions_isolated(self, pm):
        """Grants in one session don't leak to another."""
        async def _allow(_req):
            return PermissionAction.ALLOW_SESSION
        pm.set_prompt_callback(_allow)

        req = PermissionRequest(session_id="s1", tool_name="write_file", path="/tmp/x.txt")
        assert await pm.request(req) is True

        # Different session, no callback → defaults to DENY
        pm.set_prompt_callback(None)
        req2 = PermissionRequest(session_id="s2", tool_name="write_file", path="/tmp/x.txt")
        assert await pm.request(req2) is False

    @pytest.mark.asyncio
    async def test_callback_timeout_denies(self, pm):
        """If callback times out, should deny."""
        async def _slow(_req):
            await asyncio.sleep(10)
            return PermissionAction.ALLOW_ONCE
        pm.set_prompt_callback(_slow)
        req = PermissionRequest(session_id="s1", tool_name="write_file")
        assert await pm.request(req, timeout=0.1) is False


# ============================================================================
# SecurityMiddleware — permission integration
# ============================================================================


class TestSecurityMiddlewarePermissions:
    def test_permission_manager_accessible(self):
        """SecurityMiddleware exposes permission_manager."""
        mw = SecurityMiddleware()
        assert isinstance(mw.permission_manager, PermissionManager)

    def test_custom_permission_manager(self):
        """Can inject custom PermissionManager."""
        pm = PermissionManager(auto_approve=True)
        mw = SecurityMiddleware(permission_manager=pm)
        assert mw.permission_manager is pm
        assert mw.permission_manager._auto_approve is True

    def test_trusted_tool_allowed(self):
        """TRUST tools pass through without callback."""
        pm = PermissionManager(auto_approve=False)
        mw = SecurityMiddleware(permission_manager=pm)

        def mock_executor(name, args, **kw):
            return {"output": "ok"}

        result = mw.wrap_tool_call("read_file", {"path": "/tmp/test.txt"}, mock_executor)
        assert result == {"output": "ok"}  # executes normally

    def test_blocked_tool_denied(self):
        """BLOCK tools get permission_denied error."""
        pm = PermissionManager(auto_approve=False)
        pm.set_policy(ToolPolicy("cronjob", level=PermissionLevel.BLOCK))
        mw = SecurityMiddleware(permission_manager=pm)

        def mock_executor(name, args, **kw):
            return {"output": "should_not_reach"}

        result = mw.wrap_tool_call("cronjob", {"action": "create"}, mock_executor)
        assert result.get("kind") == "permission_denied"

    def test_input_guard_still_works(self):
        """Permission integration doesn't break existing input guards."""
        mw = SecurityMiddleware(max_input_chars=10)
        from kairos.core.state import ThreadState
        state = ThreadState()
        with pytest.raises(SecurityViolation):
            mw.before_model(state, {"user_message": "a" * 100})

    def test_dangerous_extension_still_blocked(self):
        """Permission integration doesn't break file safety."""
        mw = SecurityMiddleware(block_dangerous_files=True)
        def mock_executor(name, args, **kw):
            return {"output": "ok"}
        with pytest.raises(SecurityViolation):
            mw.wrap_tool_call("read_file", {"path": "/tmp/malware.exe"}, mock_executor)


# ============================================================================
# PermissionRequest
# ============================================================================


class TestPermissionRequest:
    def test_summary(self):
        req = PermissionRequest(tool_name="write_file", path="/tmp/foo",
                                description="Write file: /tmp/foo")
        assert "write_file" in req.summary()
        assert "/tmp/foo" in req.summary()

    def test_unique_ids(self):
        r1 = PermissionRequest()
        r2 = PermissionRequest()
        assert r1.id != r2.id
