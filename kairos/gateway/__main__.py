"""Gateway entry point — start the HTTP+SSE API server.

Usage:
    python -m kairos.gateway                    # defaults: 0.0.0.0:8080
    python -m kairos.gateway --port 9000        # custom port
    python -m kairos.gateway --host 127.0.0.1   # localhost only

Environment:
    DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY   # model credentials
    KAIROS_CONFIG                                           # path to config.yaml
    KAIROS_LOG_LEVEL                                         # DEBUG/INFO/WARNING
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

logging.basicConfig(
    level=getattr(logging, os.getenv("KAIROS_LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kairos.gateway")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Kairos Gateway Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--provider", default=None, help="Provider name (deepseek/openai/anthropic/gemini)")
    parser.add_argument("--model", default=None, help="Model name override")
    args = parser.parse_args()

    # Determine provider and API key
    provider_name = (
        args.provider
        or os.getenv("KAIROS_PROVIDER", "deepseek")
    )

    from kairos.providers.registry import ProviderRegistry
    registry = ProviderRegistry()
    profile = registry.get(provider_name)
    if profile is None:
        logger.error("Unknown provider: %s. Available: %s", provider_name, registry.list_names())
        sys.exit(1)

    api_key = os.getenv(profile.env_api_key, "")
    if not api_key:
        logger.error(
            "No API key found. Set %s environment variable.",
            profile.env_api_key,
        )
        sys.exit(1)

    logger.info("Starting Kairos Gateway with provider=%s model=%s", provider_name, args.model or profile.default_model)

    model_config = profile.make_config(api_key=api_key, model=args.model)

    # Build agent
    from kairos.core.loop import Agent
    agent = Agent(
        model=model_config,
        enable_subagents=True,
    )

    # Start gateway server
    from kairos.gateway.server import GatewayServer
    server = GatewayServer(
        agent=agent,
        session_ttl=float(os.getenv("KAIROS_SESSION_TTL", "3600")),
        request_timeout=float(os.getenv("KAIROS_REQUEST_TIMEOUT", "300")),
    )

    # Graceful shutdown on SIGTERM/SIGINT
    loop = asyncio.get_running_loop()

    def _shutdown():
        logger.info("Received shutdown signal")
        asyncio.ensure_future(server.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    await server.start(host=args.host, port=args.port)
    logger.info("Gateway ready on http://%s:%d", args.host, args.port)
    logger.info("Endpoints: POST /chat, GET /chat/stream, GET /health, GET /ready, GET /health/detailed, GET /stats")

    # Keep running until stopped
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
