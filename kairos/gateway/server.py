"""Gateway server — HTTP + SSE transport with session management and heartbeat."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from kairos.core.loop import Agent

logger = logging.getLogger("kairos.gateway")


class GatewayServer:
    """HTTP + SSE server for Kairos Agent with session cache and heartbeat.

    Endpoints: POST /chat, GET /chat/stream, GET /health, GET /stats, POST /sessions/clear
    """

    def __init__(
        self,
        agent: Agent,
        session_ttl: float = 3600,  # 1 hour
        request_timeout: float = 300,  # 5 minutes
        cors_origins: list[str] | None = None,
    ):
        self.agent = agent
        self._started_at = time.time()
        self._request_count = 0
        self._error_count = 0
        self._app = None
        self._runner = None
        self._session_ttl = session_ttl
        self._request_timeout = request_timeout
        self._cors_origins = cors_origins or ["*"]
        self._sessions: dict[str, dict] = {}  # session_id → {agent, last_access}

    # ── Session management ──────────────────────────────────────

    def get_or_create_session(self, session_id: str) -> Any:
        """Get or create a session-scoped agent instance."""
        now = time.time()

        # Prune expired sessions
        expired = [sid for sid, s in self._sessions.items() if now - s["last_access"] > self._session_ttl]
        for sid in expired:
            del self._sessions[sid]

        if session_id not in self._sessions:
            from kairos.core.stateful_agent import StatefulAgent
            self._sessions[session_id] = {
                "agent": StatefulAgent(
                    model=self.agent.model.config,
                    middlewares=self.agent.pipeline._layers,
                    session_id=session_id,
                ),
                "last_access": now,
            }

        self._sessions[session_id]["last_access"] = now
        return self._sessions[session_id]["agent"]

    def clear_sessions(self) -> int:
        count = len(self._sessions)
        self._sessions.clear()
        return count

    # ── Start/Stop ──────────────────────────────────────────────

    async def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError("aiohttp required. pip install aiohttp")

        self._app = web.Application()
        self._app.router.add_post("/chat", self._handle_chat)
        self._app.router.add_get("/chat/stream", self._handle_stream)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/ready", self._handle_ready)
        self._app.router.add_get("/health/detailed", self._handle_health_detailed)
        self._app.router.add_get("/stats", self._handle_stats)
        self._app.router.add_get("/metrics", self._handle_metrics)
        self._app.router.add_post("/sessions/clear", self._handle_clear_sessions)

        # Graceful shutdown handlers
        self._app.on_shutdown.append(self._on_shutdown)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        logger.info("Kairos Gateway listening on http://%s:%d", host, port)

    async def _on_shutdown(self, app) -> None:
        """Graceful shutdown: drain pending sessions, log stats."""
        logger.info(
            "Gateway shutting down — processed %d requests, %d errors, %d active sessions",
            self._request_count, self._error_count, len(self._sessions),
        )
        # Clean up sessions
        self.clear_sessions()

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    # ── Handlers ────────────────────────────────────────────────

    async def _handle_chat(self, request) -> Any:
        from aiohttp import web

        self._request_count += 1

        try:
            body = await request.json()
        except Exception:
            return self._cors(web.json_response({"error": "Invalid JSON"}, status=400))

        text = body.get("message", body.get("text", ""))
        if not text:
            return self._cors(web.json_response({"error": "No message provided"}, status=400))

        session_id = body.get("session_id", body.get("chat_id", ""))
        agent = self.get_or_create_session(session_id) if session_id else self.agent

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(agent.run, text),
                timeout=self._request_timeout,
            )
        except asyncio.TimeoutError:
            self._error_count += 1
            return self._cors(web.json_response(
                {"error": "Request timeout", "content": "The request took too long."},
                status=504,
            ))
        except Exception as e:
            self._error_count += 1
            logger.error("Chat error: %s", e)
            return self._cors(web.json_response({"error": str(e)}, status=500))

        return self._cors(web.json_response({
            "content": result.get("content", ""),
            "confidence": result.get("confidence"),
            "evidence": result.get("evidence", []),
            "interrupted": result.get("interrupted", False),
        }))

    async def _handle_stream(self, request) -> Any:
        from aiohttp import web

        text = request.query.get("message", request.query.get("text", ""))
        if not text:
            return self._cors(web.json_response({"error": "No message provided"}, status=400))

        session_id = request.query.get("session_id", "")
        agent = self.get_or_create_session(session_id) if session_id else self.agent

        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        # CORS for SSE
        resp.headers["Access-Control-Allow-Origin"] = "*"
        await resp.prepare(request)

        try:
            if hasattr(agent, "chat_stream"):
                for event in agent.chat_stream(text):
                    data = json.dumps(event, ensure_ascii=False)
                    await resp.write(f"data: {data}\n\n".encode())
            else:
                result = await asyncio.to_thread(agent.run, text)
                data = json.dumps({
                    "type": "done", "content": result.get("content", ""),
                    "confidence": result.get("confidence"),
                    "evidence": result.get("evidence", []),
                }, ensure_ascii=False)
                await resp.write(f"data: {data}\n\n".encode())
        except Exception as e:
            data = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            await resp.write(f"data: {data}\n\n".encode())

        await resp.write_eof()
        return resp

    async def _handle_health(self, request) -> Any:
        """Liveness probe — is the process alive? Always returns 200 if the server is running."""
        from aiohttp import web
        return self._cors(web.json_response({
            "status": "ok",
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "sessions": len(self._sessions),
        }))

    async def _handle_ready(self, request) -> Any:
        """Readiness probe — can the agent serve requests? Checks model availability."""
        from aiohttp import web

        checks: dict[str, bool] = {
            "agent_loaded": self.agent is not None,
        }

        # Check model provider is configured
        try:
            provider = getattr(self.agent, "_primary_provider", None)
            if provider and hasattr(provider, "config"):
                checks["model_configured"] = bool(provider.config.api_key)
            else:
                checks["model_configured"] = False
        except Exception:
            checks["model_configured"] = False

        # Check credential pool
        try:
            pool = getattr(self.agent, "_credential_pool", None)
            if pool:
                stats = pool.stats()
                checks["credential_pool"] = stats.get("default", {}).get("available_keys", 0) > 0
            else:
                checks["credential_pool"] = True  # No pool = using single key, assume OK
        except Exception:
            checks["credential_pool"] = False

        ready = all(checks.values())
        status = 200 if ready else 503

        return self._cors(web.json_response({
            "status": "ready" if ready else "not_ready",
            "checks": checks,
            "uptime_seconds": round(time.time() - self._started_at, 1),
        }, status=status))

    async def _handle_health_detailed(self, request) -> Any:
        """Detailed health check — component-level status for monitoring systems."""
        from aiohttp import web

        components: dict[str, dict[str, Any]] = {}

        # Agent status
        try:
            health = self.agent.health_status() if hasattr(self.agent, "health_status") else {}
            components["agent"] = {
                "status": "healthy" if health.get("healthy", True) else "degraded",
                **health,
            }
        except Exception as e:
            components["agent"] = {"status": "error", "error": str(e)}

        # Gateway status
        components["gateway"] = {
            "status": "healthy",
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "requests": self._request_count,
            "errors": self._error_count,
            "sessions": len(self._sessions),
            "error_rate": round(self._error_count / max(self._request_count, 1), 3),
        }

        overall = all(
            c.get("status") in ("healthy", "ok") for c in components.values()
        )

        return self._cors(web.json_response({
            "status": "healthy" if overall else "degraded",
            "components": components,
        }))

    async def _handle_stats(self, request) -> Any:
        from aiohttp import web
        return self._cors(web.json_response({
            "requests": self._request_count,
            "errors": self._error_count,
            "uptime": round(time.time() - self._started_at, 1),
            "sessions": len(self._sessions),
            "error_rate": round(self._error_count / max(self._request_count, 1), 3),
        }))

    async def _handle_metrics(self, request) -> Any:
        """Prometheus-compatible /metrics endpoint."""
        from aiohttp import web
        from kairos.observability.metrics import get_metrics

        reg = get_metrics()
        reg.set_gauge("active_sessions", float(len(self._sessions)))
        reg.set_gauge("uptime_seconds", round(time.time() - self._started_at, 1))
        reg.update_process_metrics()

        resp = web.Response(
            text=reg.render(),
            content_type="text/plain; version=0.0.4; charset=utf-8",
        )
        return self._cors(resp)

    async def _handle_clear_sessions(self, request) -> Any:
        from aiohttp import web
        count = self.clear_sessions()
        return self._cors(web.json_response({"cleared": count}))

    @staticmethod
    def _cors(response: Any) -> Any:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
