"""Tests for sub-agent tool policy + Anthropic prompt caching."""

from __future__ import annotations

from kairos.agents.policy import SubAgentPolicy, SubAgentPolicyRegistry


# ═══════════════════════════════════════════════════════════
# SubAgentPolicy
# ═══════════════════════════════════════════════════════════


class TestSubAgentPolicy:
    """Tool access control for delegated agents."""

    def test_whitelist_allows(self):
        p = SubAgentPolicy(whitelist=["read_file", "search_files"])
        assert p.is_allowed("read_file") is True
        assert p.is_allowed("search_files") is True

    def test_whitelist_blocks(self):
        p = SubAgentPolicy(whitelist=["read_file"])
        assert p.is_allowed("terminal") is False
        assert p.is_allowed("write_file") is False

    def test_blacklist_overrides_whitelist(self):
        p = SubAgentPolicy(whitelist=["terminal"], blacklist=["terminal"])
        assert p.is_allowed("terminal") is False

    def test_empty_default_disallow(self):
        """No whitelist + default_allow=False → denies everything."""
        p = SubAgentPolicy(default_allow=False)
        assert p.is_allowed("any_tool") is False

    def test_default_allow(self):
        p = SubAgentPolicy(default_allow=True, blacklist=["dangerous"])
        assert p.is_allowed("any_tool") is True
        assert p.is_allowed("dangerous") is False

    def test_filter_tools(self):
        p = SubAgentPolicy(whitelist=["read_file"])
        schemas = [
            {"function": {"name": "read_file", "description": "Read"}},
            {"function": {"name": "write_file", "description": "Write"}},
        ]
        filtered = p.filter_tools(schemas)
        assert len(filtered) == 1
        assert filtered[0]["function"]["name"] == "read_file"

    def test_to_dict(self):
        p = SubAgentPolicy(whitelist=["a", "b"], blacklist=["c"], max_iterations=15)
        d = p.to_dict()
        assert d["whitelist"] == ["a", "b"]
        assert d["max_iterations"] == 15

    # ── Presets ────────────────────────────────────────────

    def test_read_only_preset(self):
        p = SubAgentPolicy.read_only()
        assert p.is_allowed("read_file") is True
        assert p.is_allowed("terminal") is False
        assert p.allow_delegation is False

    def test_code_assistant_preset(self):
        p = SubAgentPolicy.code_assistant()
        assert p.is_allowed("terminal") is True
        assert p.is_allowed("write_file") is True
        assert p.is_allowed("gateway_start") is False

    def test_full_access_preset(self):
        p = SubAgentPolicy.full_access()
        assert p.is_allowed("gateway_start") is False  # blacklisted
        assert p.is_allowed("any_tool") is True
        assert p.allow_delegation is True


class TestSubAgentPolicyRegistry:
    """Named policy presets."""

    def test_default_presets_registered(self):
        presets = SubAgentPolicyRegistry.list_presets()
        assert "read-only" in presets
        assert "code-assistant" in presets
        assert "full-access" in presets

    def test_get_preset(self):
        p = SubAgentPolicyRegistry.get("read-only")
        assert p is not None
        assert p.is_allowed("read_file") is True

    def test_get_nonexistent(self):
        assert SubAgentPolicyRegistry.get("nonexistent") is None

    def test_custom_preset(self):
        custom = SubAgentPolicy(whitelist=["custom_tool"])
        SubAgentPolicyRegistry.register("custom", custom)
        p = SubAgentPolicyRegistry.get("custom")
        assert p is not None
        assert p.is_allowed("custom_tool") is True

    def test_list_presets_sorted(self):
        presets = SubAgentPolicyRegistry.list_presets()
        assert presets == sorted(presets)
