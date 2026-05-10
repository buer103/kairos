"""Tests for MiddlewarePipeline hook execution order.

Verifies DeerFlow-compatible execution semantics:
  - before_agent / before_model: forward (0→N)
  - after_agent / after_model:   reverse (N→0)
  - wrap_model_call / wrap_tool_call: forward wrapping
"""

from __future__ import annotations

from kairos.core.middleware import Middleware, MiddlewarePipeline


# ============================================================================
# Spy middleware: records call order
# ============================================================================

class Spy(Middleware):
    """Records hook calls in a shared list."""

    def __init__(self, name: str, log: list[str]):
        self.name = name
        self._log = log

    def before_agent(self, state, runtime):
        self._log.append(f"{self.name}.before_agent")
        return None

    def after_agent(self, state, runtime):
        self._log.append(f"{self.name}.after_agent")
        return None

    def before_model(self, state, runtime):
        self._log.append(f"{self.name}.before_model")
        return None

    def after_model(self, state, runtime):
        self._log.append(f"{self.name}.after_model")
        return None

    def wrap_model_call(self, messages, handler, **kwargs):
        self._log.append(f"{self.name}.wrap_model_call_enter")
        result = handler(messages, **kwargs)
        self._log.append(f"{self.name}.wrap_model_call_exit")
        return result

    def wrap_tool_call(self, tool_name, args, handler, **kwargs):
        self._log.append(f"{self.name}.wrap_tool_call_enter")
        result = handler(tool_name, args, **kwargs)
        self._log.append(f"{self.name}.wrap_tool_call_exit")
        return result


# ============================================================================
# Tests
# ============================================================================


class TestPipelineExecutionOrder:
    """Verify DeerFlow-compatible execution semantics."""

    def test_before_agent_forward(self):
        """before_agent executes 0→N (forward)."""
        log: list[str] = []
        A = Spy("A", log)
        B = Spy("B", log)
        C = Spy("C", log)
        pipeline = MiddlewarePipeline([A, B, C])

        pipeline.before_agent({}, {})

        assert log == ["A.before_agent", "B.before_agent", "C.before_agent"]

    def test_before_model_forward(self):
        """before_model executes 0→N (forward)."""
        log: list[str] = []
        A = Spy("A", log)
        B = Spy("B", log)
        C = Spy("C", log)
        pipeline = MiddlewarePipeline([A, B, C])

        pipeline.before_model({}, {})

        assert log == ["A.before_model", "B.before_model", "C.before_model"]

    def test_after_agent_reverse(self):
        """after_agent executes N→0 (reverse) — DeerFlow-compatible.

        Critical: Memory must save before Sandbox releases.
        If A=ThreadData(0), B=Sandbox(1), C=Memory(2),
        after_agent should be C→B→A not A→B→C.
        """
        log: list[str] = []
        A = Spy("A", log)
        B = Spy("B", log)
        C = Spy("C", log)
        pipeline = MiddlewarePipeline([A, B, C])

        pipeline.after_agent({}, {})

        assert log == ["C.after_agent", "B.after_agent", "A.after_agent"]

    def test_after_model_reverse(self):
        """after_model executes N→0 (reverse) — DeerFlow-compatible.

        ClarificationMiddleware is always last and must process first
        in after_model to handle Command(goto=END) before other layers.
        """
        log: list[str] = []
        A = Spy("A", log)
        B = Spy("B", log)
        C = Spy("C", log)
        pipeline = MiddlewarePipeline([A, B, C])

        pipeline.after_model({}, {})

        assert log == ["C.after_model", "B.after_model", "A.after_model"]

    def test_wrap_model_call_nesting(self):
        """wrap_model_call nests forward: outer wraps inner wraps model."""
        log: list[str] = []
        inner = Spy("inner", log)
        outer = Spy("outer", log)
        pipeline = MiddlewarePipeline([inner, outer])

        def model(messages, **kw):
            log.append("model")
            return "response"

        result = pipeline.wrap_model_call([], model)

        # Forward wrapping: outer(inner(model)) — outer enters first
        assert result == "response"
        assert log == [
            "outer.wrap_model_call_enter",
            "inner.wrap_model_call_enter",
            "model",
            "inner.wrap_model_call_exit",
            "outer.wrap_model_call_exit",
        ]

    def test_wrap_tool_call_nesting(self):
        """wrap_tool_call uses REVERSED order — security layers at end wrap outermost."""
        log: list[str] = []
        inner = Spy("inner", log)
        outer = Spy("outer", log)
        pipeline = MiddlewarePipeline([inner, outer])

        def tool(name, args, **kw):
            log.append("tool")
            return "result"

        result = pipeline.wrap_tool_call("test", {}, tool)

        assert result == "result"
        # Reversed: inner wraps outer wraps tool (inner enters first)
        assert log == [
            "inner.wrap_tool_call_enter",
            "outer.wrap_tool_call_enter",
            "tool",
            "outer.wrap_tool_call_exit",
            "inner.wrap_tool_call_exit",
        ]

    def test_sandbox_before_memory_after(self):
        """Real-world scenario: Sandbox(0), Memory(1).

        after_agent must call Memory → Sandbox (reverse) so memory
        saves before sandbox releases.
        """
        log: list[str] = []

        class Sandbox(Spy):
            def after_agent(self, state, runtime):
                log.append("sandbox.release")

        class Memory(Spy):
            def after_agent(self, state, runtime):
                log.append("memory.save")

        pipeline = MiddlewarePipeline([Sandbox("s", log), Memory("m", log)])
        pipeline.after_agent({}, {})

        assert log == ["memory.save", "sandbox.release"]

    def test_empty_pipeline_all_hooks(self):
        """Empty pipeline doesn't crash on any hook."""
        pipeline = MiddlewarePipeline([])
        # All should be no-ops
        pipeline.before_agent({}, {})
        pipeline.after_agent({}, {})
        pipeline.before_model({}, {})
        pipeline.after_model({}, {})

    def test_single_layer_does_not_reverse(self):
        """Single layer: forward == reverse, both work."""
        log: list[str] = []
        A = Spy("A", log)
        pipeline = MiddlewarePipeline([A])

        pipeline.after_agent({}, {})
        assert log == ["A.after_agent"]


class TestPipelineWrapOrderSingleLayer:
    """Single-layer wrap nesting."""

    def test_wrap_model_single(self):
        log: list[str] = []
        A = Spy("A", log)
        pipeline = MiddlewarePipeline([A])

        def model(messages, **kw):
            log.append("model")
            return "ok"

        result = pipeline.wrap_model_call([], model)
        assert result == "ok"
        # Single layer: outer enters, inner, model, inner exits, outer exits
        assert log == ["A.wrap_model_call_enter", "model", "A.wrap_model_call_exit"]

    def test_wrap_tool_single(self):
        log: list[str] = []
        A = Spy("A", log)
        pipeline = MiddlewarePipeline([A])

        def tool(name, args, **kw):
            log.append("tool")
            return "ok"

        result = pipeline.wrap_tool_call("t", {}, tool)
        assert result == "ok"
        # Single layer: reversed is same as forward
        assert log == ["A.wrap_tool_call_enter", "tool", "A.wrap_tool_call_exit"]


class TestPipelineAdd:
    def test_add_returns_self(self):
        pipeline = MiddlewarePipeline()
        result = pipeline.add(Spy("X", []))
        assert result is pipeline

    def test_add_appends(self):
        log: list[str] = []
        pipeline = MiddlewarePipeline([Spy("A", log)])
        pipeline.add(Spy("B", log))
        pipeline.before_agent({}, {})
        assert log == ["A.before_agent", "B.before_agent"]
