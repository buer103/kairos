"""Test suite for the Kairos Hook system — 22 lifecycle hook points.

Covers: registration, emission, priority ordering, error isolation,
thread safety, singleton pattern, integration with agent loop.
"""
from __future__ import annotations

import threading
import time

import pytest

from kairos.hooks import (
    HookPoint,
    HookRegistry,
    get_hook_registry,
    reset_hook_registry,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test starts with a fresh registry."""
    reset_hook_registry()
    yield
    reset_hook_registry()


@pytest.fixture
def registry():
    return get_hook_registry()


# ── All Hook Points Coverage ──────────────────────────────────


ALL_HOOK_POINTS = list(HookPoint)
AGENT_HOOKS = [HookPoint.AGENT_START, HookPoint.AGENT_END, HookPoint.AGENT_INTERRUPT]
MODEL_HOOKS = [HookPoint.BEFORE_MODEL, HookPoint.AFTER_MODEL, HookPoint.MODEL_ERROR]
TOOL_HOOKS = [
    HookPoint.BEFORE_TOOL,
    HookPoint.AFTER_TOOL,
    HookPoint.TOOL_ERROR,
    HookPoint.TOOL_RETRY,
]
MESSAGE_HOOKS = [HookPoint.BEFORE_MESSAGE, HookPoint.AFTER_MESSAGE]
SESSION_HOOKS = [HookPoint.SESSION_SAVE, HookPoint.SESSION_LOAD]
MIDDLEWARE_HOOKS = [HookPoint.MIDDLEWARE_CHAIN_START, HookPoint.MIDDLEWARE_CHAIN_END]
COMPRESS_HOOKS = [HookPoint.BEFORE_COMPRESSION, HookPoint.AFTER_COMPRESSION]
MEMORY_HOOKS = [HookPoint.MEMORY_SAVE, HookPoint.MEMORY_LOAD]
GATEWAY_HOOKS = [HookPoint.GATEWAY_MESSAGE_RECEIVED, HookPoint.GATEWAY_RESPONSE_SENT]
SKILL_HOOKS = [HookPoint.SKILL_LOADED, HookPoint.SKILL_CREATED]


def test_22_hook_points_total():
    """Verify we have exactly 24 hook points."""
    assert len(ALL_HOOK_POINTS) == 24, f"Expected 24, got {len(ALL_HOOK_POINTS)}"


@pytest.mark.parametrize("hook", ALL_HOOK_POINTS)
def test_each_hook_registers_and_emits(registry, hook):
    """Every hook point accepts registrations and fires callbacks."""
    received = []

    @registry.on(hook)
    def cb(**ctx):
        received.append(ctx)

    registry.emit(hook, key="val")
    assert len(received) == 1
    assert received[0].get("key") == "val"


@pytest.mark.parametrize("hook", ALL_HOOK_POINTS)
def test_each_hook_with_priority(registry, hook):
    """Every hook respects priority ordering (lower = earlier)."""
    order = []

    @registry.on(hook, priority=200)
    def late(**ctx):
        order.append("late")

    @registry.on(hook, priority=50)
    def early(**ctx):
        order.append("early")

    registry.emit(hook)
    assert order == ["early", "late"]


@pytest.mark.parametrize("hook", ALL_HOOK_POINTS)
def test_each_hook_unregister(registry, hook):
    """Every hook supports unregistration."""
    received = []

    def cb(**ctx):
        received.append(1)

    registry.register(hook, cb)
    assert registry.listeners(hook) == 1

    removed = registry.unregister(hook, cb)
    assert removed is True
    assert registry.listeners(hook) == 0

    registry.emit(hook)
    assert received == []


# ── Multi-subscriber ──────────────────────────────────────────


def test_multiple_subscribers_same_hook(registry):
    """Multiple callbacks on the same hook all fire."""
    results = []

    @registry.on(HookPoint.BEFORE_TOOL)
    def cb1(tool_name, **ctx):
        results.append(f"cb1:{tool_name}")

    @registry.on(HookPoint.BEFORE_TOOL)
    def cb2(tool_name, **ctx):
        results.append(f"cb2:{tool_name}")

    registry.emit(HookPoint.BEFORE_TOOL, tool_name="read_file")
    assert len(results) == 2
    assert "cb1:read_file" in results
    assert "cb2:read_file" in results


# ── Error Isolation ───────────────────────────────────────────


def test_error_isolation_one_failing_callback(registry):
    """One failing callback does not prevent others from running."""
    results = []

    @registry.on(HookPoint.BEFORE_TOOL)
    def bad(**ctx):
        raise RuntimeError("boom")

    @registry.on(HookPoint.BEFORE_TOOL)
    def good(**ctx):
        results.append("ok")
        return "ok"

    returned = registry.emit(HookPoint.BEFORE_TOOL, tool_name="test")
    assert results == ["ok"]
    assert "ok" in returned


def test_error_isolation_all_fail(registry):
    """When all callbacks fail, emit still returns empty list."""

    @registry.on(HookPoint.BEFORE_TOOL)
    def bad1(**ctx):
        raise ValueError("oops")

    @registry.on(HookPoint.BEFORE_TOOL)
    def bad2(**ctx):
        raise TypeError("boom")

    returned = registry.emit(HookPoint.BEFORE_TOOL)
    assert returned == []


# ── Return Value Collection ───────────────────────────────────


def test_emit_collects_non_none_results(registry):
    """emit() returns list of non-None callback return values."""

    @registry.on(HookPoint.BEFORE_TOOL)
    def returns_something(**ctx):
        return 42

    @registry.on(HookPoint.BEFORE_TOOL)
    def returns_none(**ctx):
        return None

    @registry.on(HookPoint.BEFORE_TOOL)
    def returns_string(**ctx):
        return "hello"

    results = registry.emit(HookPoint.BEFORE_TOOL)
    assert 42 in results
    assert "hello" in results
    assert None not in results
    assert len(results) == 2


# ── Priority Edge Cases ───────────────────────────────────────


def test_same_priority_maintains_registration_order(registry):
    """Callbacks with same priority fire in registration order."""
    order = []

    @registry.on(HookPoint.BEFORE_TOOL, priority=100)
    def first(**ctx):
        order.append(1)

    @registry.on(HookPoint.BEFORE_TOOL, priority=100)
    def second(**ctx):
        order.append(2)

    registry.emit(HookPoint.BEFORE_TOOL)
    assert order == [1, 2]


def test_mixed_priorities_correct_order(registry):
    """Mixed priorities fire properly: low→high."""
    order = []

    @registry.on(HookPoint.BEFORE_TOOL, priority=10)
    def a(**ctx):
        order.append("a")

    @registry.on(HookPoint.BEFORE_TOOL, priority=5)
    def b(**ctx):
        order.append("b")

    @registry.on(HookPoint.BEFORE_TOOL, priority=20)
    def c(**ctx):
        order.append("c")

    registry.emit(HookPoint.BEFORE_TOOL)
    assert order == ["b", "a", "c"]


# ── Thread Safety ─────────────────────────────────────────────


def test_concurrent_registration(registry):
    """Multiple threads can register callbacks without corruption."""
    errors = []
    barrier = threading.Barrier(5)

    def register_thread(thread_id):
        try:
            for i in range(50):
                hook = ALL_HOOK_POINTS[i % len(ALL_HOOK_POINTS)]
                registry.register(hook, lambda **ctx: None, priority=100)
        except Exception as e:
            errors.append(str(e))
        barrier.wait()

    threads = [
        threading.Thread(target=register_thread, args=(i,)) for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent register: {errors}"
    # At least some registered
    total = sum(registry.listeners(h) for h in ALL_HOOK_POINTS)
    assert total > 0


def test_concurrent_emit_and_register(registry):
    """Emit during registration does not deadlock."""
    errors = []
    start = threading.Event()
    emitter_done = threading.Event()

    def emitter():
        start.wait()
        for _ in range(100):
            try:
                registry.emit(ALL_HOOK_POINTS[0], value=1)
            except Exception as e:
                errors.append(str(e))
        emitter_done.set()

    def registrar():
        start.wait()
        for i in range(100):
            try:
                hook = ALL_HOOK_POINTS[i % len(ALL_HOOK_POINTS)]
                registry.register(hook, lambda **ctx: None)
            except Exception as e:
                errors.append(str(e))

    t1 = threading.Thread(target=emitter)
    t2 = threading.Thread(target=registrar)
    t1.start()
    t2.start()
    start.set()
    emitter_done.wait(timeout=10)
    t2.join(timeout=5)

    assert errors == [], f"Concurrent errors: {errors}"


# ── Singleton ─────────────────────────────────────────────────


def test_get_hook_registry_returns_same_instance():
    """get_hook_registry() returns the same singleton."""
    r1 = get_hook_registry()
    r2 = get_hook_registry()
    assert r1 is r2


def test_reset_hook_registry_creates_new_instance():
    """reset_hook_registry() clears the singleton."""
    r1 = get_hook_registry()
    r1.register(HookPoint.BEFORE_TOOL, lambda **ctx: None)
    assert r1.listeners(HookPoint.BEFORE_TOOL) == 1

    reset_hook_registry()
    r2 = get_hook_registry()
    assert r2 is not r1
    assert r2.listeners(HookPoint.BEFORE_TOOL) == 0


# ── list_hooks ────────────────────────────────────────────────


def test_list_hooks_shows_registered_only(registry):
    """list_hooks() only returns hooks with listeners."""
    registry.register(HookPoint.BEFORE_TOOL, lambda **ctx: None)
    registry.register(HookPoint.AFTER_TOOL, lambda **ctx: None)

    listing = registry.list_hooks()
    assert "tool:before" in listing
    assert "tool:after" in listing
    assert "model:before" not in listing
    assert listing["tool:before"] == 1
    assert listing["tool:after"] == 1


def test_list_hooks_empty_when_cleared(registry):
    """list_hooks() is empty after clear()."""
    registry.register(HookPoint.BEFORE_TOOL, lambda **ctx: None)
    registry.clear()
    assert registry.list_hooks() == {}


# ── emit_async ────────────────────────────────────────────────


def test_emit_async_delegates_to_emit(registry):
    """emit_async produces same results as emit."""
    received = []

    @registry.on(HookPoint.BEFORE_TOOL)
    def cb(**ctx):
        received.append(ctx.get("x"))

    registry.emit_async(HookPoint.BEFORE_TOOL, x=99)
    assert received == [99]


# ── Context Propagation ───────────────────────────────────────


def test_context_is_isolated_between_emits(registry):
    """Each emit() call gets a fresh context."""
    results = []

    @registry.on(HookPoint.BEFORE_TOOL)
    def cb(**ctx):
        results.append(ctx.get("tool_name", "missing"))

    registry.emit(HookPoint.BEFORE_TOOL, tool_name="tool_a")
    registry.emit(HookPoint.BEFORE_TOOL, tool_name="tool_b")
    assert results == ["tool_a", "tool_b"]


def test_extra_context_ignored_silently(registry):
    """Extra kwargs not consumed by callback cause no error."""
    received = []

    @registry.on(HookPoint.BEFORE_TOOL)
    def cb(tool_name, **ctx):
        received.append(tool_name)

    registry.emit(
        HookPoint.BEFORE_TOOL,
        tool_name="test",
        extra_field="unused",
        another=123,
    )
    assert received == ["test"]


# ── Unregister edge cases ─────────────────────────────────────


def test_unregister_nonexistent_returns_false(registry):
    """Unregistering a callback that was never registered returns False."""

    def phantom(**ctx):
        pass

    assert registry.unregister(HookPoint.BEFORE_TOOL, phantom) is False


def test_unregister_from_empty_hook_returns_false(registry):
    """Unregistering from a hook with no listeners returns False."""

    def phantom(**ctx):
        pass

    assert registry.listeners(HookPoint.BEFORE_TOOL) == 0
    assert registry.unregister(HookPoint.BEFORE_TOOL, phantom) is False


def test_clear_preserves_registry_functionality(registry):
    """After clear(), registry still works for new registrations."""
    registry.register(HookPoint.BEFORE_TOOL, lambda **ctx: None)
    registry.clear()

    results = []

    @registry.on(HookPoint.BEFORE_TOOL)
    def cb(**ctx):
        results.append(1)

    registry.emit(HookPoint.BEFORE_TOOL)
    assert results == [1]


# ── Hook Categories ───────────────────────────────────────────


def test_agent_hooks_are_three():
    assert len(AGENT_HOOKS) == 3


def test_model_hooks_are_three():
    assert len(MODEL_HOOKS) == 3


def test_tool_hooks_are_four():
    assert len(TOOL_HOOKS) == 4


def test_message_hooks_are_two():
    assert len(MESSAGE_HOOKS) == 2


def test_session_hooks_are_two():
    assert len(SESSION_HOOKS) == 2


def test_middleware_hooks_are_two():
    assert len(MIDDLEWARE_HOOKS) == 2


def test_compress_hooks_are_two():
    assert len(COMPRESS_HOOKS) == 2


def test_memory_hooks_are_two():
    assert len(MEMORY_HOOKS) == 2


def test_gateway_hooks_are_two():
    assert len(GATEWAY_HOOKS) == 2


def test_skill_hooks_are_two():
    assert len(SKILL_HOOKS) == 2


# ── Decorator pattern ─────────────────────────────────────────


def test_decorator_returns_original_function(registry):
    """@registry.on returns the decorated function unchanged."""

    @registry.on(HookPoint.BEFORE_TOOL)
    def my_func(**ctx):
        return "original"

    assert my_func() == "original"


def test_decorator_default_priority(registry):
    """Decorator without explicit priority defaults to 100."""
    registry.register(HookPoint.BEFORE_TOOL, lambda **ctx: None)
    with registry._lock:
        priorities = [p for p, _ in registry._hooks[HookPoint.BEFORE_TOOL]]
    assert 100 in priorities


# ── Emit with no listeners ───────────────────────────────────


@pytest.mark.parametrize("hook", ALL_HOOK_POINTS)
def test_emit_with_no_listeners_returns_empty(registry, hook):
    """Emitting a hook with no callbacks returns empty list."""
    assert registry.emit(hook) == []


# ── Stress test ───────────────────────────────────────────────


def test_1000_emits_on_same_hook(registry):
    """Hook system handles many emits without issues."""
    counter = [0]

    @registry.on(HookPoint.BEFORE_TOOL)
    def inc(**ctx):
        counter[0] += 1

    for _ in range(1000):
        registry.emit(HookPoint.BEFORE_TOOL)

    assert counter[0] == 1000


def test_all_22_hooks_emit_concurrently(registry):
    """All 22 hooks can be emitted concurrently via threads without deadlock."""
    errors = []

    def emit_hook(hook):
        try:
            registry.emit(hook, test=True)
        except Exception as e:
            errors.append(f"{hook.value}: {e}")

    # Register one callback per hook
    for h in ALL_HOOK_POINTS:
        registry.register(h, lambda **ctx: None)

    threads = [
        threading.Thread(target=emit_hook, args=(h,)) for h in ALL_HOOK_POINTS
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors: {errors}"
