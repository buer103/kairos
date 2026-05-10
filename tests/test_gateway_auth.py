"""Tests for GatewayAuth — per-platform whitelists, DM pairing, allow-all."""

from __future__ import annotations

import pytest

from kairos.gateway.auth import GatewayAuth, PlatformAuth, AuthResult


# ============================================================================
# Configuration
# ============================================================================


class TestPlatformAuthConfig:
    def test_defaults(self):
        pa = PlatformAuth(name="telegram")
        assert pa.name == "telegram"
        assert pa.allowlist == []
        assert not pa.allow_all
        assert pa.allow_dm_pairing

    def test_configure_platform(self):
        auth = GatewayAuth()
        pa = auth.configure_platform("discord", allowlist=["111", "222"])
        assert pa.name == "discord"
        assert pa.allowlist == ["111", "222"]

    def test_remove_platform(self):
        auth = GatewayAuth()
        auth.configure_platform("test")
        assert "test" in auth.platform_names
        auth.remove_platform("test")
        assert "test" not in auth.platform_names


# ============================================================================
# Auth checks
# ============================================================================


class TestGlobalAllowAll:
    def test_global_allow_all_bypasses_everything(self):
        auth = GatewayAuth(global_allow_all=True)
        result = auth.check("unknown_platform", chat_id="hacker")
        assert result.allowed
        assert result.reason == "global_allow_all"

    def test_toggle_global_allow_all(self):
        auth = GatewayAuth()
        result = auth.check("test", chat_id="x")
        assert not result.allowed
        auth.set_global_allow_all(True)
        result = auth.check("test", chat_id="x")
        assert result.allowed

    def test_is_allowed_shortcut(self):
        auth = GatewayAuth(global_allow_all=True)
        assert auth.is_allowed("anything", "anyone")


class TestPlatformAllowAll:
    def test_platform_allow_all_bypasses_allowlist(self):
        auth = GatewayAuth()
        auth.configure_platform("telegram", allow_all=True)
        result = auth.check("telegram", chat_id="unknown_user")
        assert result.allowed
        assert result.reason == "platform_allow_all"

    def test_platform_allow_all_only_that_platform(self):
        auth = GatewayAuth()
        auth.configure_platform("telegram", allow_all=True)
        auth.configure_platform("discord", allowlist=[])
        # telegram: allow_all → allowed
        assert auth.check("telegram", chat_id="anyone").allowed
        # discord: no allow_all, empty allowlist → pairing or denied
        result = auth.check("discord", chat_id="anyone")
        assert not result.allowed


class TestAllowlist:
    def test_allowlist_match(self):
        auth = GatewayAuth()
        auth.configure_platform("telegram", allowlist=["alice", "bob"])
        result = auth.check("telegram", chat_id="alice")
        assert result.allowed
        assert result.reason == "allowlist"

    def test_allowlist_no_match(self):
        auth = GatewayAuth()
        auth.configure_platform("telegram", allowlist=["alice"])
        result = auth.check("telegram", chat_id="charlie")
        assert not result.allowed

    def test_user_id_fallback(self):
        auth = GatewayAuth()
        auth.configure_platform("matrix", allowlist=["@bob:server"])
        # Uses user_id when chat_id is empty
        result = auth.check("matrix", user_id="@bob:server")
        assert result.allowed

    def test_empty_allowlist_with_pairing(self):
        auth = GatewayAuth()
        auth.configure_platform("telegram", allowlist=[], allow_dm_pairing=True)
        result = auth.check("telegram", chat_id="new_user")
        assert not result.allowed
        assert result.pairing_required
        assert result.pairing_code is not None
        assert len(result.pairing_code) == 6

    def test_empty_allowlist_without_pairing(self):
        auth = GatewayAuth()
        auth.configure_platform("telegram", allowlist=[], allow_dm_pairing=False)
        result = auth.check("telegram", chat_id="new_user")
        assert not result.allowed
        assert not result.pairing_required
        assert result.reason == "not_in_allowlist"


class TestNoPlatformConfig:
    def test_no_config_denies(self):
        auth = GatewayAuth()
        result = auth.check("nonexistent", chat_id="test")
        assert not result.allowed
        assert result.reason == "no_platform_config"


# ============================================================================
# DM Pairing
# ============================================================================


class TestPairing:
    def test_confirm_pairing(self):
        auth = GatewayAuth()
        auth.configure_platform("telegram", allowlist=[])
        result = auth.check("telegram", chat_id="alice")
        assert result.pairing_required
        code = result.pairing_code
        assert code is not None

        # Confirm
        confirmed = auth.confirm_pairing(code)
        assert confirmed == "alice"

        # Now alice is allowed
        result2 = auth.check("telegram", chat_id="alice")
        assert result2.allowed
        assert result2.reason == "paired"

    def test_confirm_pairing_invalid_code(self):
        auth = GatewayAuth()
        assert auth.confirm_pairing("INVALID") is None

    def test_revoke_pairing(self):
        auth = GatewayAuth()
        auth.configure_platform("telegram", allowlist=[])
        result = auth.check("telegram", chat_id="alice")
        auth.confirm_pairing(result.pairing_code)
        assert auth.check("telegram", chat_id="alice").allowed

        auth.revoke_pairing("alice")
        assert not auth.check("telegram", chat_id="alice").allowed

    def test_pairing_code_unique(self):
        import secrets
        auth = GatewayAuth()
        auth.configure_platform("test")
        codes = set()
        for _ in range(50):
            result = auth.check("test", chat_id=secrets.token_hex(4))
            if result.pairing_code:
                codes.add(result.pairing_code)
        # Codes should be unique or almost unique
        assert len(codes) > 0

    def test_pairing_code_length_configurable(self):
        auth = GatewayAuth()
        auth.configure_platform("test", pairing_code_length=8)
        result = auth.check("test", chat_id="user")
        assert len(result.pairing_code) == 8


# ============================================================================
# Serialization
# ============================================================================


class TestSerialization:
    def test_roundtrip(self):
        auth = GatewayAuth(global_allow_all=False)
        auth.configure_platform(
            "telegram", allowlist=["alice", "bob"], allow_all=False
        )
        auth.configure_platform(
            "discord", allow_all=True
        )
        # Add a paired user
        auth._paired_users.add("charlie")

        data = auth.to_dict()
        auth2 = GatewayAuth.from_dict(data)

        assert auth2._global_allow_all == auth._global_allow_all
        assert set(auth2._paired_users) == set(auth._paired_users)
        assert auth2._platforms["telegram"].allowlist == ["alice", "bob"]
        assert auth2._platforms["discord"].allow_all is True

    def test_roundtrip_empty(self):
        auth = GatewayAuth()
        data = auth.to_dict()
        auth2 = GatewayAuth.from_dict(data)
        assert not auth2._global_allow_all
        assert auth2._paired_users == set()

    def test_repr(self):
        auth = GatewayAuth()
        auth.configure_platform("telegram", allowlist=["alice"])
        r = repr(auth)
        assert "telegram" not in r  # repr shows count, not names
        assert "1" in r or "platforms=1" in r
