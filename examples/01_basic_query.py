"""
01 — Basic Query
================

The simplest possible Kairos agent: ask a question, get an answer.
No middleware, no tools, no configuration — just a model and a prompt.

This is your starting point. Everything else builds on this pattern.

Run:
    export DEEPSEEK_API_KEY=sk-your-key
    python examples/01_basic_query.py
"""

import os
from kairos import Agent
from kairos.providers.base import ModelConfig

# ── Replace with your own API key, or set DEEPSEEK_API_KEY env var ──
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-your-key-here"

# ── Create the agent ─────────────────────────────────────────────
# ModelConfig accepts any OpenAI-compatible endpoint.
# Default: DeepSeek (base_url="https://api.deepseek.com", model="deepseek-chat")
agent = Agent(
    model=ModelConfig(api_key=API_KEY),
    agent_name="BasicBot",
    role_description="You are a concise assistant who answers in plain language.",
)

# ── Run a query ──────────────────────────────────────────────────
result = agent.run("What is the capital of France? Answer in one sentence.")

# result is a dict with these keys:
#   "content"      — the LLM's text response
#   "confidence"   — float 0-1 (requires ConfidenceScorer middleware)
#   "evidence"     — list of evidence steps (requires EvidenceTracker)
#   "trace_context" — TraceContext for full-chain observability

print(result["content"])
print(f"\nConfidence: {result.get('confidence', 'N/A')}")
print(f"Trace ID:   {result['trace_context'].trace_id[:8] if result.get('trace_context') else 'N/A'}")
