"""Entry point: python -m kairos.web"""

import asyncio
import os
import sys

from kairos.web import WebServer
from kairos.core.stateful_agent import StatefulAgent
from kairos.providers.base import ModelConfig


def main():
    host = os.environ.get("KAIROS_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("KAIROS_WEB_PORT", "8080"))

    # Build agent
    api_key = os.environ.get("KAIROS_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ No API key found. Set KAIROS_API_KEY or DEEPSEEK_API_KEY.")
        sys.exit(1)

    base_url = os.environ.get("KAIROS_BASE_URL", "https://api.deepseek.com/v1")
    model_name = os.environ.get("KAIROS_MODEL", "deepseek-chat")

    model = ModelConfig(api_key=api_key, base_url=base_url, model=model_name)
    agent = StatefulAgent(model=model)

    server = WebServer(agent=agent, host=host, port=port)

    async def run():
        await server.start()
        print(f"\n✨ Kairos Web UI: http://{host}:{port}\n")
        # Keep running
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


if __name__ == "__main__":
    main()
