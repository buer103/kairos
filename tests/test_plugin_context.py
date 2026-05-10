"""Tests for PluginContext + PluginRegistry — Hermes-compatible plugin registration."""

from __future__ import annotations

import pytest

from kairos.plugins.context import PluginContext
from kairos.plugins.registry import (
    PluginRegistry,
    HookEntry,
    PlatformEntry,
    CliCommandEntry,
    reset_plugin_registry,
    get_plugin_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry between tests."""
    reset_plugin_registry()


# ============================================================================
# PluginRegistry
# ============================================================================


class TestPluginRegistryHooks:
    def test_add_and_get_hook(self):
        reg = PluginRegistry()
        called = []

        def cb(**kw):
            called.append(kw)

        reg.add_hook("before_model", cb, priority=10)
        entries = reg.get_hooks("before_model")
        assert len(entries) == 1
        assert entries[0].priority == 10

    def test_invoke_hooks(self):
        reg = PluginRegistry()
        results = []

        def cb(**kw):
            results.append(kw.get("msg", ""))

        reg.add_hook("test_hook", cb)
        reg.invoke_hooks("test_hook", msg="hello")
        assert results == ["hello"]

    def test_hooks_ordered_by_priority(self):
        reg = PluginRegistry()
        order = []

        def hi(**kw):
            order.append("hi")

        def lo(**kw):
            order.append("lo")

        reg.add_hook("test", lo, priority=0)
        reg.add_hook("test", hi, priority=10)
        entries = reg.get_hooks("test")
        assert entries[0].priority == 10
        assert entries[1].priority == 0

    def test_remove_hook(self):
        reg = PluginRegistry()

        def cb(**kw):
            pass

        reg.add_hook("test", cb)
        assert reg.remove_hook("test", cb)
        assert reg.get_hooks("test") == []


class TestPluginRegistryPlatforms:
    def test_add_and_get_platform(self):
        reg = PluginRegistry()

        def factory(cfg):
            pass

        reg.add_platform("irc", "IRC", factory, emoji="💬")
        p = reg.get_platform("irc")
        assert p.name == "irc"
        assert p.label == "IRC"
        assert p.emoji == "💬"

    def test_platforms_dict(self):
        reg = PluginRegistry()
        reg.add_platform("a", "A", lambda c: None)
        reg.add_platform("b", "B", lambda c: None)
        assert len(reg.platforms) == 2


class TestPluginRegistryCLI:
    def test_add_cli_command(self):
        reg = PluginRegistry()

        def handler(args):
            pass

        reg.add_cli_command("deploy", handler, "Deploy the app")
        cmd = reg.get_cli_command("deploy")
        assert cmd.name == "deploy"
        assert cmd.description == "Deploy the app"


class TestPluginRegistrySummary:
    def test_summary_counts(self):
        reg = PluginRegistry()

        def dummy(**kw):
            pass

        reg.add_hook("h1", dummy)
        reg.add_hook("h2", dummy)
        reg.add_platform("x", "X", dummy)
        reg.add_cli_command("c", dummy)
        reg.add_skill("s", "/tmp/s")
        reg.add_middleware("m", dummy)
        reg.add_memory("mem", dummy)
        reg.add_provider("p", dummy)

        s = reg.summary()
        assert s["hooks"] == 2
        assert s["platforms"] == 1
        assert s["cli_commands"] == 1
        assert s["skills"] == 1
        assert s["middleware"] == 1
        assert s["memory"] == 1
        assert s["providers"] == 1


# ============================================================================
# PluginContext
# ============================================================================


class TestPluginContextRegistration:
    """All register_* methods work through PluginContext."""

    def test_plugin_name(self):
        ctx = PluginContext("myplugin")
        assert ctx.plugin_name == "myplugin"

    def test_register_hook(self):
        ctx = PluginContext("test")

        def cb(**kw):
            pass

        ctx.register_hook("before_model", cb, priority=5)
        reg = get_plugin_registry()
        entries = reg.get_hooks("before_model")
        assert len(entries) == 1
        assert "hook:before_model" in ctx.registrations

    def test_register_platform(self):
        ctx = PluginContext("irc_plugin")

        def factory(cfg):
            pass

        ctx.register_platform("irc", "IRC", factory, emoji="💬")
        reg = get_plugin_registry()
        p = reg.get_platform("irc")
        assert p.label == "IRC"
        assert "platform:irc" in ctx.registrations

    def test_register_cli_command(self):
        ctx = PluginContext("deploy_plugin")

        def handler(args):
            pass

        ctx.register_cli_command("deploy", handler, "Deploy now")
        reg = get_plugin_registry()
        cmd = reg.get_cli_command("deploy")
        assert cmd.description == "Deploy now"
        assert "cli_cmd:deploy" in ctx.registrations

    def test_register_skill(self):
        ctx = PluginContext("skill_plugin")
        ctx.register_skill("my-skill", "/tmp/skills/my-skill", "A test skill")
        reg = get_plugin_registry()
        assert reg.get_skill_path("my-skill") == "/tmp/skills/my-skill"
        assert "skill:my-skill" in ctx.registrations

    def test_register_middleware(self):
        ctx = PluginContext("mw_plugin")

        def factory():
            return "mw_instance"

        ctx.register_middleware("my_mw", factory, position="before:Clarification")
        reg = get_plugin_registry()
        mw = reg.get_middleware("my_mw")
        assert mw.position == "before:Clarification"
        assert "middleware:my_mw" in ctx.registrations

    def test_register_memory(self):
        ctx = PluginContext("redis_plugin")

        def factory():
            return "redis_backend"

        ctx.register_memory("redis", factory)
        reg = get_plugin_registry()
        assert "redis" in reg.memory_providers
        assert "memory:redis" in ctx.registrations

    def test_register_provider(self):
        ctx = PluginContext("mistral_plugin")

        def factory(cfg):
            return "mistral_adapter"

        ctx.register_provider("mistral", factory, model_names=["mistral-large"])
        reg = get_plugin_registry()
        assert "mistral" in reg.provider_names
        assert "provider:mistral" in ctx.registrations


class TestPluginContextTracking:
    """Registrations are tracked in order."""

    def test_tracks_all_registrations(self):
        ctx = PluginContext("full")

        def dummy(**kw):
            pass

        ctx.register_hook("h", dummy)
        ctx.register_platform("x", "X", dummy)
        ctx.register_skill("s", "/tmp/s")
        ctx.register_middleware("m", dummy)
        ctx.register_memory("mem", dummy)
        ctx.register_provider("p", dummy)

        assert len(ctx.registrations) == 6
        assert "hook:h" in ctx.registrations
        assert "platform:x" in ctx.registrations
        assert "skill:s" in ctx.registrations

    def test_repr(self):
        ctx = PluginContext("myplug")

        def dummy(**kw):
            pass

        ctx.register_hook("h", dummy)
        r = repr(ctx)
        assert "myplug" in r
        assert "registrations=1" in r


class TestPluginRegistrySingleton:
    def test_same_instance(self):
        a = get_plugin_registry()
        b = get_plugin_registry()
        assert a is b

    def test_reset_works(self):
        a = get_plugin_registry()

        def dummy(**kw):
            pass

        a.add_hook("test", dummy)
        assert len(a.get_hooks("test")) == 1
        reset_plugin_registry()
        b = get_plugin_registry()
        assert b.get_hooks("test") == []


class TestPluginRegistryClear:
    def test_clear_removes_all(self):
        reg = PluginRegistry()

        def dummy(**kw):
            pass

        reg.add_hook("h", dummy)
        reg.add_platform("p", "P", dummy)
        reg.clear()
        assert reg.summary()["hooks"] == 0
        assert reg.summary()["platforms"] == 0
