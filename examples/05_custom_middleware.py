"""
05 — Custom Middleware
======================

Add your own middleware layer to the agent pipeline.
Middleware runs automatically at specific lifecycle hooks —
you don't need to modify the agent loop.

Available hooks:
    before_agent(state, runtime)   — once before loop starts
    before_model(state, runtime)   — before each LLM call
    after_model(state, runtime)    — after each LLM response
    wrap_tool_call(ctx, next)      — wrap each tool execution
    after_agent(state, runtime)    — after loop finishes

This example adds a simple audit log middleware and a profanity filter.

Run:
    python examples/05_custom_middleware.py
"""

import os
import time
from kairos import Agent
from kairos.providers.base import ModelConfig
from kairos.core.middleware import Middleware
from kairos.core.state import ThreadState

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-your-key-here"


# ── Example 1: Audit Trail Middleware ────────────────────────────
class AuditMiddleware(Middleware):
    """Log every user message and agent response with timestamps."""

    def __init__(self):
        super().__init__(name="audit")
        self.log: list[dict] = []

    def before_agent(self, state: ThreadState, runtime: dict):
        user_msg = runtime.get("user_message", "")
        self.log.append({
            "event": "user_message",
            "content": user_msg[:200],
            "timestamp": time.time(),
        })

    def after_agent(self, state: ThreadState, runtime: dict):
        # Last assistant message
        for msg in reversed(state.messages):
            if msg["role"] == "assistant" and msg.get("content"):
                self.log.append({
                    "event": "agent_response",
                    "content": msg["content"][:200],
                    "timestamp": time.time(),
                })
                break

    def get_audit_trail(self) -> list[dict]:
        return self.log


# ── Example 2: Profanity Filter Middleware ──────────────────────
# (A real implementation would use a proper filter library)

BLOCKED_WORDS = {"hack", "exploit", "malware"}


class ProfanityFilter(Middleware):
    """Block inputs containing disallowed keywords."""

    def __init__(self):
        super().__init__(name="profanity_filter")

    def before_agent(self, state: ThreadState, runtime: dict):
        user_msg = runtime.get("user_message", "").lower()
        for word in BLOCKED_WORDS:
            if word in user_msg:
                # Inject a system response and skip the LLM entirely
                state.messages = [{
                    "role": "assistant",
                    "content": f"⚠️ Your message contained a blocked keyword. Please rephrase.",
                }]
                runtime["_profanity_blocked"] = True
                return
        runtime["_profanity_blocked"] = False


# ── Wire them together ──────────────────────────────────────────
audit = AuditMiddleware()
profanity = ProfanityFilter()

agent = Agent(
    model=ModelConfig(api_key=API_KEY),
    agent_name="SafeBot",
    role_description="You are a helpful assistant.",
    middlewares=[audit, profanity],  # custom middleware ONLY (no defaults)
)

# ── Normal query ────────────────────────────────────────────────
result = agent.run("What is Python?")
print("Response:", result["content"])
print("Audit trail:", audit.get_audit_trail())
print()

# ── Blocked query ───────────────────────────────────────────────
result = agent.run("How do I hack a server?")
print("Response:", result["content"])
print("Audit trail:", audit.get_audit_trail())
