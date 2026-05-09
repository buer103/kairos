"""Gateway server — HTTP + SSE transport for Kairos Agent.

Provides:
  - POST /chat — synchronous chat
  - GET /chat/stream — SSE streaming chat
  - GET /health — health check
  - GET /stats — agent stats
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from kairos.core.loop import Agent
from kairos.gateway.protocol import UnifiedMessage, UnifiedResponse, MessageRole, ContentBlock, ContentType


class GatewayServer:
    """HTTP + SSE server for Kairos Agent.

    Usage:
        agent = Agent(model=ModelConfig(api_key="..."))
        gateway = GatewayServer(agent=agent)
        gateway.start(host="0.0.0.0", port=8000)

    Dependencies: aiohttp (pip install aiohttp)
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self._started_at = time.time()
        self._request_count = 0
        self._app = None
        self._runner = None

    async def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the HTTP server (non-blocking)."""
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for GatewayServer. Install with: pip install aiohttp"
            )

        self._app = web.Application()
        self._app.router.add_post("/chat", self._handle_chat)
        self._app.router.add_get("/chat/stream", self._handle_stream)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/stats", self._handle_stats)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        print(f"Kairos Gateway listening on http://{host}:{port}")

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._runner:
            await self._runner.cleanup()

    async def process(self, message: UnifiedMessage) -> UnifiedResponse:
        """Process a unified message through the agent."""
        self._request_count += 1
        result = self.agent.run(message.text)
        response = UnifiedResponse(
            text=result.get("content", ""),
            confidence=result.get("confidence"),
            evidence=result.get("evidence", []),
            metadata={"request_id": message.id, "platform": message.platform},
        )
        return response

    # ── HTTP Handlers ──────────────────────────────────────────

    async def _handle_chat(self, request) -> Any:
        try:
            from aiohttp import web
        except ImportError:
            pass

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        text = body.get("message", body.get("text", ""))
        if not text:
            return web.json_response({"error": "No message provided"}, status=400)

        msg = UnifiedMessage.from_text(
            text,
            platform=body.get("platform", "http"),
            chat_id=body.get("chat_id", ""),
        )

        response = await self.process(msg)
        return web.json_response(response.to_dict())

    async def _handle_stream(self, request) -> Any:
        try:
            from aiohttp import web
        except ImportError:
            pass

        text = request.query.get("message", request.query.get("text", ""))
        if not text:
            return web.json_response({"error": "No message provided"}, status=400)

        response_obj = web.StreamResponse()
        response_obj.headers["Content-Type"] = "text/event-stream"
        response_obj.headers["Cache-Control"] = "no-cache"
        response_obj.headers["Connection"] = "keep-alive"
        await response_obj.prepare(request)

        try:
            # Use real streaming if agent supports it
            if hasattr(self.agent, "chat_stream"):
                for event in self.agent.chat_stream(text):
                    data = json.dumps(event, ensure_ascii=False)
                    await response_obj.write(f"data: {data}\n\n".encode())
            else:
                # Fallback to synchronous
                result = self.agent.run(text)
                data = json.dumps({
                    "type": "done",
                    "content": result.get("content", ""),
                    "confidence": result.get("confidence"),
                    "evidence": result.get("evidence", []),
                }, ensure_ascii=False)
                await response_obj.write(f"data: {data}\n\n".encode())
        except Exception as e:
            data = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            await response_obj.write(f"data: {data}\n\n".encode())

        await response_obj.write_eof()
        return response_obj

    async def _handle_health(self, request) -> Any:
        try:
            from aiohttp import web
        except ImportError:
            pass
        return web.json_response({"status": "ok", "uptime": time.time() - self._started_at})

    async def _handle_stats(self, request) -> Any:
        try:
            from aiohttp import web
        except ImportError:
            pass
        return web.json_response({
            "requests": self._request_count,
            "uptime": time.time() - self._started_at,
            "agent_max_iterations": self.agent.max_iterations,
        })
