"""
02 — Custom Tool
================

Register your own business-specific tool and let the agent call it.
This is the core pattern for building domain agents (diagnostics,
data queries, API integrations, etc.).

Pattern: @register_tool decorator → Agent picks it up automatically.

Run:
    python examples/02_custom_tool.py
"""

import os
from kairos import Agent
from kairos.providers.base import ModelConfig
from kairos.tools.registry import register_tool

# ── 1. Define your business function ─────────────────────────────
# This could query your database, call your internal API, etc.

def fetch_order_status(order_id: str) -> dict:
    """Simulate fetching order status from your backend."""
    # In real code: requests.get(f"https://api.your-company.com/orders/{order_id}")
    orders = {
        "ORD-1234": {"status": "shipped", "eta": "2026-05-15", "tracking": "1Z999AA123"},
        "ORD-5678": {"status": "processing", "eta": "2026-05-18"},
        "ORD-9012": {"status": "delivered", "eta": "2026-05-10"},
    }
    return orders.get(order_id, {"status": "not_found", "error": f"Unknown order: {order_id}"})


# ── 2. Register it as a tool ────────────────────────────────────
# The decorator auto-registers into the global tool registry.
# The Agent will see this tool and can call it when relevant.

@register_tool(
    name="fetch_order_status",
    description="Fetch the status and tracking info for an order by its ID (e.g., ORD-1234).",
    parameters={
        "order_id": {
            "type": "string",
            "description": "The order ID to look up, e.g. ORD-1234",
        },
    },
    category="business",
)
def tool_fetch_order_status(order_id: str) -> str:
    result = fetch_order_status(order_id)
    return (
        f"Order {order_id}: {result['status'].upper()}"
        + (f", tracking: {result['tracking']}" if "tracking" in result else "")
        + (f", error: {result['error']}" if "error" in result else "")
    )


# ── 3. Create an agent and ask about orders ──────────────────────
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-your-key-here"

agent = Agent(
    model=ModelConfig(api_key=API_KEY),
    agent_name="OrderBot",
    role_description=(
        "You are a customer support agent. "
        "Use fetch_order_status to check order information. "
        "Be helpful and concise."
    ),
)

result = agent.run("What is the status of order ORD-1234?")
print(result["content"])
