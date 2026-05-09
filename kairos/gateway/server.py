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
        self._app.router.add_get("/stats", self._handle_stats)
        self._app.router.add_post("/sessions/clear", self._handle_clear_sessions)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        logger.info("Kairos Gateway listening on http://%s:%d", host, port)

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
        from aiohttp import web
        return self._cors(web.json_response({
            "status": "ok",
            "uptime": round(time.time() - self._started_at, 1),
            "sessions": len(self._sessions),
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
