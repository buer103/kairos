"""
Kairos — The right tool, at the right moment.
An AI agent framework inheriting from Hermes and DeerFlow.
"""

__version__ = "0.1.0"

from kairos.core.loop import Agent
from kairos.tools.registry import register_tool

__all__ = ["Agent", "register_tool", "__version__"]
