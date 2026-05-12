"""
06 — Gateway Server
===================

Run Kairos as a multi-platform HTTP server. Users can interact with your
agent through Telegram, WeChat, Slack, Discord, CLI, and more —
all through a single unified API.

The Gateway:
  - Exposes a POST /chat endpoint for programmatic access
  - Exposes GET /chat/stream for SSE streaming
  - Exposes GET /health, /ready for monitoring
  - Routes messages through platform adapters

Requires:
    pip install aiohttp           # or: pip install kairos-agent[gateway]

Run:
    python examples/06_gateway_server.py
    # Then in another terminal:
    curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" \
         -d '{"message": "Hello Kairos!"}'
"""

import asyncio
import os
from kairos import Agent
from kairos.providers.base import ModelConfig
from kairos.gateway import (
    GatewayServer,
    CLIAdapter,
    UnifiedMessage,
    UnifiedResponse,
    MessageRole,
)

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-your-key-here"

# ── 1. Create your agent ────────────────────────────────────────
agent = Agent(
    model=ModelConfig(api_key=API_KEY),
    agent_name="GatewayBot",
    role_description="You are a helpful assistant accessible via HTTP API.",
)

# ── 2. Choose platform adapters ─────────────────────────────────
# CLI adapter is for terminal/curl access.
# Add TelegramAdapter, WeChatAdapter, etc. for messaging platforms.
adapters = [
    CLIAdapter(),
    # TelegramAdapter(bot_token=os.getenv("TELEGRAM_BOT_TOKEN")),
    # WeChatAdapter(app_id=..., app_secret=..., token=...),
]

# ── 3. Start the gateway ────────────────────────────────────────
gateway = GatewayServer(agent=agent, adapters=adapters)

print("Kairos Gateway starting on http://0.0.0.0:8080")
print()
print("Try these endpoints:")
print("  GET  http://localhost:8080/health")
print("  POST http://localhost:8080/chat  (body: {\"message\": \"Hello\"})")
print("  GET  http://localhost:8080/chat/stream?message=Hello")
print()

try:
    asyncio.run(gateway.start(host="0.0.0.0", port=8080))
except KeyboardInterrupt:
    print("\nGateway stopped.")
