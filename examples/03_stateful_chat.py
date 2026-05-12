"""
03 — Stateful Chat
==================

Multi-turn conversation with session persistence.
StatefulAgent remembers context across messages and can save/load sessions.

Use this for chatbots, customer support agents, or any long-running conversation.

Run:
    python examples/03_stateful_chat.py
"""

import os
from kairos import StatefulAgent
from kairos.providers.base import ModelConfig

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-your-key-here"

# ── StatefulAgent: maintains conversation history across calls ──
agent = StatefulAgent(
    model=ModelConfig(api_key=API_KEY),
    agent_name="MemoryBot",
    role_description="You remember what the user tells you and can recall it later.",
)

# ── Turn 1 ──────────────────────────────────────────────────────
print("=== Turn 1 ===")
result = agent.chat("My name is Alice and I live in Tokyo.")
print(result["content"])

# ── Turn 2 — agent remembers the name from turn 1 ────────────────
print("\n=== Turn 2 ===")
result = agent.chat("What is my name and where do I live?")
print(result["content"])

# ── Save session for later ──────────────────────────────────────
session_name = "alice-session"
agent.save_session(session_name)
print(f"\n✅ Session saved as: {session_name}")

# ── Later (in another process / after restart) ──────────────────
new_agent = StatefulAgent(model=ModelConfig(api_key=API_KEY))
new_agent.load_session(session_name)

print("\n=== Resumed Session ===")
result = new_agent.chat("What city did I say I live in?")
print(result["content"])

# ── List all saved sessions ─────────────────────────────────────
sessions = agent.list_sessions()
print(f"\nSaved sessions: {len(sessions)}")
for s in sessions:
    print(f"  ● {s['name']}  ({s.get('message_count', 0)} messages)")
