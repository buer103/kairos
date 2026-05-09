"""Kairos built-in plugins package.

Plugins are auto-discovered by PluginManager when it scans
~/.kairos/plugins/ and the kairos/plugins/builtin/ directory.

Each plugin defines: PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_TYPE
and a register(manager) function.
"""

from importlib import import_module

_BUILTIN_PLUGINS = [
    "kairos.plugins.builtin.memory_plugin",
    "kairos.plugins.builtin.context_plugin",
    "kairos.plugins.builtin.provider_plugin",
]


def register_all(manager) -> None:
    """Auto-register all built-in plugins with the given PluginManager."""
    for module_name in _BUILTIN_PLUGINS:
        try:
            mod = import_module(module_name)
            if hasattr(mod, "register"):
                mod.register(manager)
        except Exception:
            pass  # Plugin may have unmet deps — skip gracefully
