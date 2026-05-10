"""Tests for GatewayServer health, readiness, and detailed health endpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from kairos.core.loop import Agent
from kairos.gateway.server import GatewayServer


class TestGatewayHealth:
    """Tests for /health, /ready, /health/detailed endpoints."""

    @pytest.fixture
    def server(self):
        mock_agent = MagicMock(spec=Agent)
        mock_agent.health_status.return_value = {
            "healthy": True,
            "active_provider": 0,
            "budget": {"iterations": 0, "tokens_used": 0, "remaining": 120000},
        }
        # Mock primary provider
        mock_agent._primary_provider = MagicMock()
        mock_agent._primary_provider.config.api_key = "sk-test"
        mock_agent._credential_pool = None
        return GatewayServer(agent=mock_agent)

    @pytest.fixture
    def mock_request(self):
        """Minimal mock for aiohttp request."""
        req = MagicMock()
        return req

    def test_health_returns_ok(self, server, mock_request):
        """GET /health returns 200 with status ok."""
        import asyncio
        async def run():
            resp = await server._handle_health(mock_request)
            data = json.loads(resp.text)
            assert data["status"] == "ok"
            assert "uptime_seconds" in data
        asyncio.run(run())

    def test_ready_returns_ready_when_configured(self, server, mock_request):
        """GET /ready returns ready when agent has API key."""
        import asyncio
        async def run():
            resp = await server._handle_ready(mock_request)
            data = json.loads(resp.text)
            assert data["status"] == "ready"
            assert data["checks"]["agent_loaded"] is True
            assert data["checks"]["model_configured"] is True
        asyncio.run(run())

    def test_ready_returns_not_ready_when_no_api_key(self, server, mock_request):
        """GET /ready returns 503 when model not configured."""
        server.agent._primary_provider.config.api_key = ""
        import asyncio
        async def run():
            resp = await server._handle_ready(mock_request)
            data = json.loads(resp.text)
            assert data["status"] == "not_ready"
            assert data["checks"]["model_configured"] is False
            assert resp.status == 503
        asyncio.run(run())

    def test_ready_handles_missing_provider(self, server, mock_request):
        """GET /ready handles agent without _primary_provider."""
        del server.agent._primary_provider
        import asyncio
        async def run():
            resp = await server._handle_ready(mock_request)
            data = json.loads(resp.text)
            assert data["status"] == "not_ready"
        asyncio.run(run())

    def test_health_detailed_returns_components(self, server, mock_request):
        """GET /health/detailed returns component-level status."""
        import asyncio
        async def run():
            resp = await server._handle_health_detailed(mock_request)
            data = json.loads(resp.text)
            assert data["status"] == "healthy"
            assert "agent" in data["components"]
            assert "gateway" in data["components"]
            assert data["components"]["gateway"]["uptime_seconds"] >= 0
        asyncio.run(run())

    def test_health_detailed_degraded_on_agent_error(self, server, mock_request):
        """GET /health/detailed returns degraded when agent.health_status raises."""
        server.agent.health_status.side_effect = RuntimeError("boom")
        import asyncio
        async def run():
            resp = await server._handle_health_detailed(mock_request)
            data = json.loads(resp.text)
            assert data["components"]["agent"]["status"] == "error"
        asyncio.run(run())


class TestAgentHealthStatus:
    """Tests for Agent.health_status() called by gateway."""

    def test_health_status_returns_budget_and_providers(self):
        from kairos.providers.base import ModelConfig
        from kairos.core.loop import Agent

        agent = Agent(model=ModelConfig(api_key="sk-test"))
        status = agent.health_status()

        assert "active_provider" in status
        assert "budget" in status
        assert "providers" in status
        assert len(status["providers"]) >= 1
        assert "is_healthy" in status["providers"][0]
        assert status["providers"][0]["is_healthy"] is True
