"""Pluggable context engines for Kairos.

Context engines control how the agent's message history is compressed
to fit within the model's context window. The default engine is
ContextCompressor v3 — plug in your own by implementing ContextEngine.
"""
from kairos.engines.protocol import ContextEngine

__all__ = ["ContextEngine"]
