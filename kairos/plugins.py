"""Plugin system — extensible architecture for Kairos.

Plugins are Python packages placed in ~/.kairos/plugins/<name>/
Each plugin has a PLUGIN.md manifest and an optional __init__.py entry point.

Plugin types:
  - tool: register custom tools
  - middleware: custom middleware layers
  - provider: custom model providers
  - memory: custom memory backends
  - gateway: custom platform adapters
  - hook: lifecycle hooks

Manifest format (PLUGIN.md):
  ```yaml
  ---
  name: my-plugin
  version: 1.0.0
  type: tool
  description: My custom tools
  entry_point: my_plugin.register
  dependencies:
    - requests>=2.28
  ---
  ```

Usage:
    pm = PluginManager()
    pm.discover()
    pm.load_all()
    tools = pm.get_plugins("tool")
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# Parser for YAML frontmatter in PLUGIN.md
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class PluginManifest:
    """A plugin's metadata from its PLUGIN.md file."""

    name: str
    version: str = "0.1.0"
    type: str = "tool"  # tool, middleware, provider, memory, gateway, hook
    description: str = ""
    entry_point: str = ""  # "module.function" to call on load
    dependencies: list[str] = field(default_factory=list)
    path: Path = field(default_factory=Path)
    loaded: bool = False
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "type": self.type,
            "description": self.description,
            "entry_point": self.entry_point,
            "dependencies": self.dependencies,
            "loaded": self.loaded,
            "enabled": self.enabled,
        }


class PluginManager:
    """Discovers, loads, and manages plugins.

    Plugin directories scanned:
      1. ~/.kairos/plugins/  — user plugins
      2. /etc/kairos/plugins/ — system plugins
      3. Any path added via add_search_path()
    """

    def __init__(self, plugin_dirs: list[Path] | None = None):
        self._search_paths: list[Path] = plugin_dirs or [
            Path.home() / ".kairos" / "plugins",
        ]
        self._plugins: dict[str, PluginManifest] = {}
        self._registry: dict[str, dict[str, Any]] = {
            "tool": {},
            "middleware": {},
            "provider": {},
            "memory": {},
            "gateway": {},
            "hook": {},
        }
        self._hooks: dict[str, list[Callable]] = {}

    # ── Discovery ────────────────────────────────────────

    def add_search_path(self, path: str | Path) -> None:
        self._search_paths.append(Path(path))

    def discover(self) -> list[PluginManifest]:
        """Scan all search paths for PLUGIN.md files."""
        discovered = []
        for base in self._search_paths:
            if not base.exists():
                continue
            for manifest_path in base.glob("*/PLUGIN.md"):
                try:
                    manifest = self._parse_manifest(manifest_path)
                    if manifest:
                        discovered.append(manifest)
                        self._plugins[manifest.name] = manifest
                except Exception as e:
                    get_logger("plugins").warning(
                        f"Failed to parse plugin manifest: {manifest_path}: {e}"
                    )
        return discovered

    def _parse_manifest(self, path: Path) -> PluginManifest | None:
        content = path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return None

        frontmatter = match.group(1)
        fields = self._parse_yaml_lite(frontmatter)

        name = fields.get("name", path.parent.name)
        return PluginManifest(
            name=name,
            version=fields.get("version", "0.1.0"),
            type=fields.get("type", "tool"),
            description=fields.get("description", ""),
            entry_point=fields.get("entry_point", ""),
            dependencies=self._parse_deps(fields.get("dependencies", "")),
            path=path.parent,
        )

    @staticmethod
    def _parse_yaml_lite(text: str) -> dict[str, str]:
        """Simple YAML key: value parser (no pyyaml dependency)."""
        result = {}
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip().strip('"').strip("'")
        return result

    @staticmethod
    def _parse_deps(deps_str: str) -> list[str]:
        if not deps_str:
            return []
        if isinstance(deps_str, list):
            return deps_str
        # "requests>=2.28, click" → ["requests>=2.28", "click"]
        return [d.strip() for d in deps_str.split(",") if d.strip()]

    # ── Loading ──────────────────────────────────────────

    def load(self, name: str) -> bool:
        """Load a single plugin by name."""
        manifest = self._plugins.get(name)
        if not manifest or manifest.loaded:
            return False

        if not manifest.enabled:
            return False

        try:
            get_logger("plugins").info(f"Loading plugin: {name} (type={manifest.type})")

            if manifest.entry_point:
                self._load_entry_point(manifest)
            else:
                self._load_package(manifest)

            manifest.loaded = True
            return True
        except Exception as e:
            get_logger("plugins").error(f"Failed to load plugin {name}: {e}")
            return False

    def load_all(self, plugin_type: str | None = None) -> int:
        """Load all discovered plugins, optionally filtered by type. Returns count loaded."""
        if not self._plugins:
            self.discover()

        count = 0
        for manifest in self._plugins.values():
            if plugin_type and manifest.type != plugin_type:
                continue
            if self.load(manifest.name):
                count += 1
        return count

    def _load_entry_point(self, manifest: PluginManifest) -> None:
        """Call the entry point function: 'module.function'."""
        module_name, _, func_name = manifest.entry_point.rpartition(".")

        # Add plugin dir to path
        sys.path.insert(0, str(manifest.path.parent))

        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name, None)
            if func and callable(func):
                result = func(self, manifest)
                if isinstance(result, dict):
                    self._registry[manifest.type].update(result)
            elif module:
                # Module-level side effects (e.g., @register_tool decorators)
                self._registry[manifest.type][manifest.name] = module
        finally:
            if str(manifest.path.parent) in sys.path:
                sys.path.remove(str(manifest.path.parent))

    def _load_package(self, manifest: PluginManifest) -> None:
        """Load the plugin as a package (__init__.py)."""
        sys.path.insert(0, str(manifest.path.parent))

        try:
            module = importlib.import_module(manifest.name)
            self._registry[manifest.type][manifest.name] = module
        finally:
            if str(manifest.path.parent) in sys.path:
                sys.path.remove(str(manifest.path.parent))

    # ── Registry ─────────────────────────────────────────

    def register(self, plugin_type: str, name: str, obj: Any) -> None:
        """Manually register a plugin object."""
        self._registry.setdefault(plugin_type, {})[name] = obj

    def get_plugins(self, plugin_type: str) -> dict[str, Any]:
        """Get all loaded plugins of a given type."""
        return dict(self._registry.get(plugin_type, {}))

    def get_plugin(self, plugin_type: str, name: str) -> Any:
        """Get a specific plugin by type and name."""
        return self._registry.get(plugin_type, {}).get(name)

    # ── Hooks ────────────────────────────────────────────

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a lifecycle hook callback."""
        self._hooks.setdefault(event, []).append(callback)

    def trigger_hook(self, event: str, **kwargs) -> list[Any]:
        """Trigger all callbacks for a lifecycle event."""
        results = []
        for cb in self._hooks.get(event, []):
            try:
                results.append(cb(**kwargs))
            except Exception as e:
                get_logger("plugins").error(f"Hook {event} failed: {e}")
        return results

    # ── Management ───────────────────────────────────────

    def list_plugins(self, plugin_type: str | None = None) -> list[dict]:
        """List all discovered plugins."""
        plugins = self._plugins.values()
        if plugin_type:
            plugins = [p for p in plugins if p.type == plugin_type]
        return [p.to_dict() for p in plugins]

    def enable(self, name: str) -> bool:
        """Enable a disabled plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a plugin (won't load on next discover)."""
        if name in self._plugins:
            self._plugins[name].enabled = False
            return True
        return False

    def stats(self) -> dict[str, Any]:
        """Return plugin statistics."""
        by_type = {}
        for p in self._plugins.values():
            by_type[p.type] = by_type.get(p.type, 0) + 1
        return {
            "total_discovered": len(self._plugins),
            "total_loaded": sum(1 for p in self._plugins.values() if p.loaded),
            "by_type": by_type,
            "search_paths": [str(p) for p in self._search_paths],
        }


# Lazy import to avoid circular dependency
from kairos.logging import get_logger
