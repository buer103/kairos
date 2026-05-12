"""Core package — Agent Loop, StatefulAgent, MiddlewarePipeline, ThreadPaths.

The heart of Kairos. Provides:
- ``Agent`` — one-shot query agent with full middleware pipeline
- ``StatefulAgent`` — multi-turn chat with session persistence
- ``MiddlewarePipeline`` — 20-layer lifecycle hook framework
- ``ThreadPaths`` — isolated workspace directories per thread
- ``ThreadDataState`` — typed runtime state container
"""

from kairos.core.loop import Agent
from kairos.core.stateful_agent import StatefulAgent
from kairos.core.middleware import Middleware, MiddlewarePipeline
from kairos.core.paths import ThreadPaths
from kairos.core.thread_state import ThreadDataState
