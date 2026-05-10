"""Integration tests for GatewayManager with auth layer."""

from __future__ import annotations

import asyncio
import pytest

from kairos.gateway.protocol import ConnectionState, UnifiedMessage, UnifiedResponse
from kairos.gateway.adapters.base import PlatformAdapter
from kairos.gateway.manager import GatewayManager
from kairos.gateway.auth import GatewayAuth


# ============================================================================
# Mock
# ============================================================================


class TestAdapter(PlatformAdapter):
    """Minimal adapter for integration tests."""

    def __init__(self, name: str):
        self._name = name
        self._state = ConnectionState.DISCONNECTED

    @property
    def platform_name(self) -> str:
        return self._name

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def connect(self) -> bool:
        self._state = ConnectionState.CONNECTED
        return True

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        return True

    async def receive(self) -> UnifiedMessage | None:
        return None

    def translate_incoming(self, raw: dict) -> UnifiedMessage:
        import uuid
        from kairos.gateway.protocol import ContentBlock, ContentType, MessageRole
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            role=MessageRole.USER,
            content=[ContentBlock(type=ContentType.TEXT, text=raw.get("t", ""))],
            platform=self._name,
            chat_id="t",
        )


# ============================================================================
# Tests
# ============================================================================


class TestGatewayManagerWithAuth:
    """GatewayManager route() checks auth before routing."""

    @pytest.mark.asyncio
    async def test_allowlist_allows(self):
        mgr = GatewayManager()
        mgr.auth.configure_platform("telegram", allowlist=["alice"])
        adapter = TestAdapter("telegram")
        mgr.register(adapter)
        await adapter.connect()

        result = await mgr.route("telegram", chat_id="alice")
        assert result.error is None

    @pytest.mark.asyncio
    async def test_allowlist_denies(self):
        mgr = GatewayManager()
        mgr.auth.configure_platform("telegram", allowlist=["alice"])
        adapter = TestAdapter("telegram")
        mgr.register(adapter)
        await adapter.connect()

        result = await mgr.route("telegram", chat_id="eve")
        assert result.error is not None
        assert "Auth denied" in result.error

    @pytest.mark.asyncio
    async def test_global_allow_all_bypasses(self):
        mgr = GatewayManager()
        mgr.auth.set_global_allow_all(True)
        # No platform config at all
        adapter = TestAdapter("telegram")
        mgr.register(adapter)
        await adapter.connect()

        result = await mgr.route("telegram", chat_id="anyone")
        assert result.error is None

    @pytest.mark.asyncio
    async def test_platform_allow_all_bypasses(self):
        mgr = GatewayManager()
        mgr.auth.configure_platform("discord", allow_all=True)
        adapter = TestAdapter("discord")
        mgr.register(adapter)
        await adapter.connect()

        result = await mgr.route("discord", chat_id="anyone")
        assert result.error is None

    @pytest.mark.asyncio
    async def test_pairing_returns_code(self):
        mgr = GatewayManager()
        mgr.auth.configure_platform("telegram", allowlist=[], allow_dm_pairing=True)
        adapter = TestAdapter("telegram")
        mgr.register(adapter)
        await adapter.connect()

        result = await mgr.route("telegram", chat_id="new_user")
        assert result.error is not None
        assert "pairing_required" in result.error
        assert "code:" in result.error

    @pytest.mark.asyncio
    async def test_paired_user_allowed(self):
        mgr = GatewayManager()
        mgr.auth.configure_platform("telegram", allowlist=[])
        # Simulate pairing
        mgr.auth._paired_users.add("bob")
        adapter = TestAdapter("telegram")
        mgr.register(adapter)
        await adapter.connect()

        result = await mgr.route("telegram", chat_id="bob")
        assert result.error is None

    @pytest.mark.asyncio
    async def test_user_id_auth(self):
        mgr = GatewayManager()
        mgr.auth.configure_platform("matrix", allowlist=["@alice:server"])
        adapter = TestAdapter("matrix")
        mgr.register(adapter)
        await adapter.connect()

        result = await mgr.route("matrix", chat_id="", user_id="@alice:server")
        assert result.error is None


class TestGatewayManagerAuthNoAdapter:
    @pytest.mark.asyncio
    async def test_no_adapter_with_auth(self):
        mgr = GatewayManager()
        mgr.auth.configure_platform("telegram", allowlist=["alice"])
        result = await mgr.route("telegram", chat_id="alice")
        # Auth passes, but no adapter → error about adapter
        assert "No adapter" in result.error or result.error is not None
