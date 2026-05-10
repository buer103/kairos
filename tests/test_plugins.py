"""Tests for Kairos plugin system: PluginManifest, PluginManager, built-in plugins.

Covers: plugins.py (300 lines), plugins/builtin/* (3 plugins).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kairos.plugins import PluginManifest, PluginManager
from kairos.plugins.builtin import register_all, _BUILTIN_PLUGINS


# ============================================================================
# PluginManifest
# ============================================================================

class TestPluginManifest:
    """Tests for PluginManifest dataclass."""

    def test_minimal_manifest(self):
        m = PluginManifest(name="test-plugin")
        assert m.name == "test-plugin"
        assert m.version == "0.1.0"
        assert m.type == "tool"
        assert m.loaded is False
        assert m.enabled is True

    def test_full_manifest(self):
        m = PluginManifest(
            name="my-plugin",
            version="2.1.0",
            type="middleware",
            description="A test plugin",
            entry_point="my_plugin.register",
            dependencies=["requests>=2.28"],
            enabled=False,
        )
        assert m.type == "middleware"
        assert m.entry_point == "my_plugin.register"
        assert "requests>=2.28" in m.dependencies

    def test_to_dict(self):
        m = PluginManifest(name="test", type="tool", description="desc")
        d = m.to_dict()
        assert d["name"] == "test"
        assert d["type"] == "tool"
        assert d["loaded"] is False
        assert d["enabled"] is True


# ============================================================================
# PluginManager — Parsing
# ============================================================================

class TestPluginManagerParsing:
    """Tests for YAML-lite parsing and manifest discovery."""

    def test_parse_yaml_lite(self):
        text = "name: my-plugin\nversion: 1.0.0\ntype: tool\ndescription: Hello world"
        result = PluginManager._parse_yaml_lite(text)
        assert result["name"] == "my-plugin"
        assert result["version"] == "1.0.0"
        assert result["type"] == "tool"

    def test_parse_yaml_lite_quoted_values(self):
        text = 'name: "my plugin"\nversion: "2.0"\ndescription: "test"'
        result = PluginManager._parse_yaml_lite(text)
        assert result["name"] == "my plugin"
        assert result["version"] == "2.0"

    def test_parse_yaml_lite_skips_comments(self):
        text = "# comment\nname: ok\n# another comment\ntype: tool"
        result = PluginManager._parse_yaml_lite(text)
        assert result["name"] == "ok"

    def test_parse_deps_string(self):
        result = PluginManager._parse_deps("requests>=2.28, click, typer>=0.9")
        assert len(result) == 3
        assert "requests>=2.28" in result
        assert "click" in result

    def test_parse_deps_empty(self):
        assert PluginManager._parse_deps("") == []
        assert PluginManager._parse_deps(None) == []

    def test_parse_deps_list(self):
        result = PluginManager._parse_deps(["a", "b"])
        assert result == ["a", "b"]

    def test_parse_manifest_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin_dir = Path(tmp) / "test-plugin"
            plugin_dir.mkdir()
            manifest_path = plugin_dir / "PLUGIN.md"
            manifest_path.write_text(
                "---\nname: my-plugin\nversion: 1.2.3\ntype: middleware\n"
                "description: A middleware plugin\nentry_point: my_mod.func\n"
                "dependencies: dep1,dep2\n---\n\n# Plugin docs\n"
            )

            pm = PluginManager(plugin_dirs=[Path(tmp)])
            discovered = pm.discover()

            assert len(discovered) == 1
            m = discovered[0]
            assert m.name == "my-plugin"
            assert m.version == "1.2.3"
            assert m.type == "middleware"
            assert m.entry_point == "my_mod.func"
            assert m.dependencies == ["dep1", "dep2"]

    def test_parse_manifest_no_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin_dir = Path(tmp) / "bad-plugin"
            plugin_dir.mkdir()
            (plugin_dir / "PLUGIN.md").write_text("# No frontmatter\n\nJust docs.")

            pm = PluginManager(plugin_dirs=[Path(tmp)])
            discovered = pm.discover()
            assert discovered == []

    def test_discover_skips_missing_dir(self):
        pm = PluginManager(plugin_dirs=[Path("/nonexistent/path")])
        discovered = pm.discover()
        assert discovered == []

    def test_add_search_path(self):
        pm = PluginManager()
        initial_count = len(pm._search_paths)
        pm.add_search_path("/custom/plugins")
        assert len(pm._search_paths) == initial_count + 1


# ============================================================================
# PluginManager — Registry
# ============================================================================

class TestPluginManagerRegistry:
    """Tests for plugin registration and retrieval."""

    @pytest.fixture
    def pm(self):
        return PluginManager(plugin_dirs=[])

    def test_register_and_get(self, pm):
        pm.register("tool", "my_tool", lambda x: x)
        tools = pm.get_plugins("tool")
        assert "my_tool" in tools

    def test_get_plugin(self, pm):
        pm.register("middleware", "mw1", "middleware_obj")
        result = pm.get_plugin("middleware", "mw1")
        assert result == "middleware_obj"

    def test_get_plugin_missing(self, pm):
        assert pm.get_plugin("tool", "nonexistent") is None

    def test_get_plugins_returns_copy(self, pm):
        pm.register("tool", "t1", "v1")
        tools = pm.get_plugins("tool")
        tools["t2"] = "v2"
        assert "t2" not in pm._registry["tool"]

    def test_get_plugins_unknown_type(self, pm):
        result = pm.get_plugins("nonexistent_type")
        assert result == {}


# ============================================================================
# PluginManager — Lifecycle Hooks
# ============================================================================

class TestPluginManagerHooks:
    """Tests for plugin lifecycle hooks."""

    @pytest.fixture
    def pm(self):
        return PluginManager(plugin_dirs=[])

    def test_register_and_trigger_hook(self, pm):
        results = []

        def my_hook(**kwargs):
            results.append(kwargs.get("data"))

        pm.register_hook("on_startup", my_hook)
        pm.trigger_hook("on_startup", data="hello")

        assert results == ["hello"]

    def test_multiple_hooks_same_event(self, pm):
        called = []

        pm.register_hook("event", lambda **kw: called.append(1))
        pm.register_hook("event", lambda **kw: called.append(2))

        pm.trigger_hook("event")
        assert called == [1, 2]

    def test_hook_error_does_not_block_others(self, pm):
        called = []

        def broken(**kw):
            raise RuntimeError("boom")

        def works(**kw):
            called.append("ok")

        pm.register_hook("test", broken)
        pm.register_hook("test", works)

        results = pm.trigger_hook("test")
        assert called == ["ok"]

    def test_trigger_unknown_event(self, pm):
        results = pm.trigger_hook("nonexistent")
        assert results == []


# ============================================================================
# PluginManager — Management
# ============================================================================

class TestPluginManagerManagement:
    """Tests for enable/disable/list/stats."""

    @pytest.fixture
    def pm(self):
        return PluginManager(plugin_dirs=[])

    def test_list_plugins_empty(self, pm):
        assert pm.list_plugins() == []

    def test_list_plugins_filtered_by_type(self, pm):
        pm._plugins["a"] = PluginManifest(name="a", type="tool")
        pm._plugins["b"] = PluginManifest(name="b", type="middleware")
        pm._plugins["c"] = PluginManifest(name="c", type="tool")

        tools = pm.list_plugins(plugin_type="tool")
        assert len(tools) == 2

        all_plugins = pm.list_plugins()
        assert len(all_plugins) == 3

    def test_enable_disable(self, pm):
        pm._plugins["p1"] = PluginManifest(name="p1", enabled=False)
        assert pm.enable("p1") is True
        assert pm._plugins["p1"].enabled is True

        assert pm.disable("p1") is True
        assert pm._plugins["p1"].enabled is False

    def test_enable_unknown(self, pm):
        assert pm.enable("nonexistent") is False

    def test_disable_unknown(self, pm):
        assert pm.disable("nonexistent") is False

    def test_stats(self, pm):
        pm._plugins["a"] = PluginManifest(name="a", type="tool", loaded=True)
        pm._plugins["b"] = PluginManifest(name="b", type="middleware")

        stats = pm.stats()
        assert stats["total_discovered"] == 2
        assert stats["total_loaded"] == 1
        assert stats["by_type"]["tool"] == 1
        assert stats["by_type"]["middleware"] == 1


# ============================================================================
# PluginManager — Load
# ============================================================================

class TestPluginManagerLoad:
    """Tests for plugin loading (load, load_all, _load_entry_point, _load_package)."""

    @pytest.fixture
    def pm(self):
        return PluginManager(plugin_dirs=[])

    def test_load_unknown_plugin(self, pm):
        assert pm.load("nonexistent") is False

    def test_load_already_loaded(self, pm):
        pm._plugins["p1"] = PluginManifest(name="p1", loaded=True)
        assert pm.load("p1") is False

    def test_load_disabled(self, pm):
        pm._plugins["p1"] = PluginManifest(name="p1", enabled=False)
        assert pm.load("p1") is False

    def test_load_entry_point(self, pm):
        def my_func(manager, manifest):
            return {"my_tool": lambda: 42}

        manifest = PluginManifest(
            name="p1", type="tool",
            entry_point="test_module.my_func",
            path=Path("/tmp/p1"),
        )
        pm._plugins["p1"] = manifest

        mock_module = MagicMock()
        mock_module.my_func = my_func

        with patch("importlib.import_module", return_value=mock_module):
            result = pm.load("p1")

        assert result is True
        assert manifest.loaded is True
        assert "my_tool" in pm._registry["tool"]

    def test_load_entry_point_no_func(self, pm):
        manifest = PluginManifest(
            name="p1", type="tool",
            entry_point="test_module.missing_func",
            path=Path("/tmp/p1"),
        )
        pm._plugins["p1"] = manifest

        mock_module = MagicMock(spec=[])  # no attributes

        with patch("importlib.import_module", return_value=mock_module):
            result = pm.load("p1")

        # Should still succeed (module was imported, no crash)
        assert result is True

    def test_load_package(self, pm):
        manifest = PluginManifest(
            name="p1", type="middleware",
            entry_point="",  # no entry_point → use _load_package
            path=Path("/tmp/p1"),
        )
        pm._plugins["p1"] = manifest

        mock_module = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            result = pm.load("p1")

        assert result is True
        assert manifest.loaded is True
        assert pm._registry["middleware"]["p1"] is mock_module

    def test_load_all_filters_by_type(self, pm):
        pm._plugins["a"] = PluginManifest(name="a", type="tool")
        pm._plugins["b"] = PluginManifest(name="b", type="middleware")

        # Mock load to always succeed
        with patch.object(pm, "load", return_value=True) as mock_load:
            count = pm.load_all(plugin_type="tool")
            assert count == 1
            mock_load.assert_called_once_with("a")

    def test_load_all_auto_discovers(self, pm):
        # No plugins in registry → should call discover
        with patch.object(pm, "discover", return_value=[]):
            count = pm.load_all()
            assert count == 0


# ============================================================================
# Built-in Plugins
# ============================================================================

class TestBuiltinPlugins:
    """Tests for built-in plugin registration."""

    def test_register_all(self):
        pm = PluginManager(plugin_dirs=[])

        # register_all calls register on each built-in plugin
        with patch("kairos.plugins.builtin.memory_plugin.register") as mock_mem:
            with patch("kairos.plugins.builtin.context_plugin.register") as mock_ctx:
                with patch("kairos.plugins.builtin.provider_plugin.register") as mock_prov:
                    register_all(pm)

        mock_mem.assert_called_once_with(pm)
        mock_ctx.assert_called_once_with(pm)
        mock_prov.assert_called_once_with(pm)

    def test_builtin_plugins_list(self):
        assert len(_BUILTIN_PLUGINS) == 3
        assert "kairos.plugins.builtin.memory_plugin" in _BUILTIN_PLUGINS
        assert "kairos.plugins.builtin.context_plugin" in _BUILTIN_PLUGINS
        assert "kairos.plugins.builtin.provider_plugin" in _BUILTIN_PLUGINS

    def test_memory_plugin_register(self):
        from kairos.plugins.builtin.memory_plugin import register
        pm = PluginManager(plugin_dirs=[])
        register(pm)
        assert "sqlite" in pm._registry["memory"]
        assert "dict" in pm._registry["memory"]
        assert pm._registry["memory"]["sqlite"]["version"] == "1.0.0"

    def test_context_plugin_register(self):
        from kairos.plugins.builtin.context_plugin import register
        pm = PluginManager(plugin_dirs=[])
        register(pm)
        assert "trajectory_compressor" in pm._registry["middleware"]
        assert "importance_scorer" in pm._registry["middleware"]

    def test_provider_plugin_register(self):
        from kairos.plugins.builtin.provider_plugin import register
        pm = PluginManager(plugin_dirs=[])
        register(pm)
        assert "anthropic" in pm._registry["provider"]
        assert "gemini" in pm._registry["provider"]
        assert "deepseek" in pm._registry["provider"]
        assert "models" in pm._registry["provider"]["anthropic"]
        assert len(pm._registry["provider"]["anthropic"]["models"]) > 0

    def test_provider_plugin_helpers(self):
        from kairos.plugins.builtin.provider_plugin import get_available_providers
        providers = get_available_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "gemini" in providers
        assert "deepseek" in providers
        assert "openrouter" in providers
        assert "groq" in providers
        assert "qwen" in providers
        assert providers["openai"]["default_model"] == "gpt-4o"
        assert providers["deepseek"]["default_model"] == "deepseek-chat"
        assert providers["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"
        assert providers["groq"]["env_api_key"] == "GROQ_API_KEY"
        assert providers["qwen"]["env_api_key"] == "DASHSCOPE_API_KEY"
        assert providers["anthropic"]["requires_native_sdk"] is True
