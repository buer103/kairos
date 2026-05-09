"""Skin engine — YAML-based theme system for Kairos Rich TUI.

Supports:
  - Built-in skins (10+ themes)
  - Custom YAML skin files (~/.config/kairos/skins/)
  - Hot-reload (detect file changes)
  - Color scheme generation (mono, complementary, analogous)
  - Dark/Light mode detection

Usage:
    engine = SkinEngine()
    engine.load_skin("ocean")  # built-in
    engine.load_skin("~/.config/kairos/skins/my-theme.yaml")  # custom
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from rich.style import Style
from rich.box import Box, ROUNDED, HEAVY, HEAVY_EDGE, SIMPLE, MINIMAL, DOUBLE, SQUARE

# ═══════════════════════════════════════════════════════════════════════════
# Built-in skins
# ═══════════════════════════════════════════════════════════════════════════

BUILTIN_SKINS: dict[str, dict[str, Any]] = {
    "default": {
        "name": "Default",
        "description": "Clean, readable default theme",
        "agent_color": "cyan",
        "user_color": "green",
        "tool_color": "yellow",
        "error_color": "red",
        "info_color": "blue",
        "success_color": "green",
        "warning_color": "yellow",
        "dim_color": "dim",
        "box": "rounded",
        "spinner": "dots",
        "panel_padding": 1,
        "code_theme": "monokai",
    },
    "hacker": {
        "name": "Hacker",
        "description": "Matrix-inspired green-on-black",
        "agent_color": "bright_green",
        "user_color": "green",
        "tool_color": "bright_yellow",
        "error_color": "bright_red",
        "info_color": "cyan",
        "success_color": "bright_green",
        "warning_color": "bright_yellow",
        "dim_color": "dim",
        "box": "minimal",
        "spinner": "line",
        "panel_padding": 1,
        "code_theme": "native",
    },
    "retro": {
        "name": "Retro",
        "description": "80s terminal aesthetic",
        "agent_color": "bright_cyan",
        "user_color": "bright_magenta",
        "tool_color": "bright_yellow",
        "error_color": "red",
        "info_color": "white",
        "success_color": "bright_green",
        "warning_color": "bright_yellow",
        "dim_color": "dim",
        "box": "simple",
        "spinner": "arc",
        "panel_padding": 1,
        "code_theme": "fruity",
    },
    "minimal": {
        "name": "Minimal",
        "description": "Bare essentials, low visual noise",
        "agent_color": "white",
        "user_color": "dim white",
        "tool_color": "dim yellow",
        "error_color": "red",
        "info_color": "dim blue",
        "success_color": "dim green",
        "warning_color": "dim yellow",
        "dim_color": "bright_black",
        "box": "minimal",
        "spinner": "dots",
        "panel_padding": 0,
        "code_theme": "native",
    },
    "ocean": {
        "name": "Ocean",
        "description": "Deep blue ocean colors",
        "agent_color": "deep_sky_blue1",
        "user_color": "sea_green2",
        "tool_color": "gold1",
        "error_color": "indian_red",
        "info_color": "steel_blue",
        "success_color": "medium_sea_green",
        "warning_color": "light_goldenrod1",
        "dim_color": "grey50",
        "box": "rounded",
        "spinner": "bouncingBall",
        "panel_padding": 1,
        "code_theme": "material",
    },
    "sunset": {
        "name": "Sunset",
        "description": "Warm orange/purple palette",
        "agent_color": "dark_orange",
        "user_color": "medium_purple",
        "tool_color": "wheat1",
        "error_color": "deep_pink2",
        "info_color": "plum2",
        "success_color": "green_yellow",
        "warning_color": "orange1",
        "dim_color": "grey58",
        "box": "rounded",
        "spinner": "dots",
        "panel_padding": 1,
        "code_theme": "dracula",
    },
    "forest": {
        "name": "Forest",
        "description": "Natural greens and browns",
        "agent_color": "chartreuse3",
        "user_color": "dark_khaki",
        "tool_color": "light_goldenrod3",
        "error_color": "firebrick1",
        "info_color": "dark_sea_green4",
        "success_color": "medium_spring_green",
        "warning_color": "yellow4",
        "dim_color": "grey50",
        "box": "rounded",
        "spinner": "growVertical",
        "panel_padding": 1,
        "code_theme": "emacs",
    },
    "midnight": {
        "name": "Midnight",
        "description": "Dark mode with subtle accents",
        "agent_color": "grey78",
        "user_color": "grey58",
        "tool_color": "grey42",
        "error_color": "grey89 on red",
        "info_color": "grey66",
        "success_color": "grey78",
        "warning_color": "grey50",
        "dim_color": "grey35",
        "box": "simple",
        "spinner": "dots",
        "panel_padding": 1,
        "code_theme": "native",
    },
    "neon": {
        "name": "Neon",
        "description": "Cyberpunk neon aesthetic",
        "agent_color": "magenta",
        "user_color": "bright_cyan",
        "tool_color": "bright_green",
        "error_color": "red on black",
        "info_color": "bright_blue",
        "success_color": "bright_green",
        "warning_color": "bright_yellow",
        "dim_color": "dim",
        "box": "heavy",
        "spinner": "bouncingBar",
        "panel_padding": 1,
        "code_theme": "one-dark",
    },
    "mono": {
        "name": "Monochrome",
        "description": "Black and white, high contrast",
        "agent_color": "bold white",
        "user_color": "dim white",
        "tool_color": "italic white",
        "error_color": "underline white",
        "info_color": "white",
        "success_color": "bold white",
        "warning_color": "dim white",
        "dim_color": "bright_black",
        "box": "minimal",
        "spinner": "simpleDotsScrolling",
        "panel_padding": 0,
        "code_theme": "bw",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Box name → Box object mapping
# ═══════════════════════════════════════════════════════════════════════════

_BOX_MAP: dict[str, Box] = {
    "rounded": ROUNDED,
    "heavy": HEAVY,
    "heavy_edge": HEAVY_EDGE,
    "simple": SIMPLE,
    "minimal": MINIMAL,
    "double": DOUBLE,
    "square": SQUARE,
}


def _resolve_box(name: str) -> Box:
    """Resolve a box name to a Rich Box object."""
    return _BOX_MAP.get(name.lower(), ROUNDED)


# ═══════════════════════════════════════════════════════════════════════════
# Skin Engine
# ═══════════════════════════════════════════════════════════════════════════


class SkinEngine:
    """YAML-based theme engine with hot-reload support."""

    def __init__(self, skins_dir: str = "~/.config/kairos/skins"):
        self._skins_dir = Path(skins_dir).expanduser()
        self._skins_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_skins: dict[str, dict[str, Any]] = dict(BUILTIN_SKINS)
        self._current_name = "default"
        self._current: dict[str, Any] = BUILTIN_SKINS["default"]
        self._file_timestamps: dict[str, float] = {}
        self._load_custom_skins()

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_custom_skins(self) -> None:
        """Scan skins_dir for YAML skin files and load them."""
        if not self._skins_dir.exists():
            return

        for skin_file in self._skins_dir.glob("*.yaml"):
            try:
                skin = self._load_yaml_skin(skin_file)
                if skin:
                    name = skin.get("name", skin_file.stem)
                    self._loaded_skins[name] = skin
                    self._file_timestamps[name] = skin_file.stat().st_mtime
            except Exception:
                pass  # Skip broken files

        for skin_file in self._skins_dir.glob("*.yml"):
            try:
                skin = self._load_yaml_skin(skin_file)
                if skin:
                    name = skin.get("name", skin_file.stem)
                    self._loaded_skins[name] = skin
                    self._file_timestamps[name] = skin_file.stat().st_mtime
            except Exception:
                pass

    def _load_yaml_skin(self, path: Path) -> dict[str, Any] | None:
        """Load a YAML skin file. Returns None if missing or invalid."""
        if not path.exists():
            return None

        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except ImportError:
            # Fallback: minimal line-based parser for simple YAML
            data = self._parse_simple_yaml(path)

        if not isinstance(data, dict):
            return None

        # Ensure required fields with defaults
        if "name" not in data:
            data["name"] = path.stem
        data.setdefault("agent_color", "cyan")
        data.setdefault("user_color", "green")
        data.setdefault("tool_color", "yellow")
        data.setdefault("error_color", "red")
        data.setdefault("info_color", "blue")
        data.setdefault("success_color", "green")
        data.setdefault("warning_color", "yellow")
        data.setdefault("dim_color", "dim")
        data.setdefault("box", "rounded")
        data.setdefault("spinner", "dots")
        data.setdefault("panel_padding", 1)

        return data

    @staticmethod
    def _parse_simple_yaml(path: Path) -> dict[str, Any]:
        """Minimal YAML parser for key: value lines (no nested structures)."""
        data: dict[str, Any] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line and not line.startswith("-"):
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if not value or value == "null":
                        continue
                    # Try cast to int/float/bool
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
                        value = float(value)
                    data[key] = value
        return data

    # ── Skin management ───────────────────────────────────────────────────

    def load_skin(self, name: str) -> bool:
        """Load a skin by name. Checks built-in first, then custom files.

        If name is a file path (contains / or \\) or ends with .yaml/.yml,
        treats it as a file path.
        """
        # File path
        if "/" in name or "\\" in name or name.endswith((".yaml", ".yml")):
            path = Path(name).expanduser()
            if path.exists():
                skin = self._load_yaml_skin(path)
                if skin:
                    skin_name = skin.get("name", path.stem)
                    self._loaded_skins[skin_name] = skin
                    self._current = skin
                    self._current_name = skin_name
                    return True
            return False

        # Check built-in + loaded
        if name in self._loaded_skins:
            self._current = self._loaded_skins[name]
            self._current_name = name
            return True

        return False

    def reload(self, name: str | None = None) -> bool:
        """Reload current skin (or a specific one) from disk."""
        if name:
            return self.load_skin(name)

        # Reload current
        self._load_custom_skins()
        if self._current_name in self._loaded_skins:
            self._current = self._loaded_skins[self._current_name]
            return True
        return False

    def check_hot_reload(self) -> bool:
        """Check if any loaded custom skins changed on disk. Reload if so."""
        changed = False
        for name, mtime in list(self._file_timestamps.items()):
            skin_file = self._skins_dir / f"{name}.yaml"
            if not skin_file.exists():
                skin_file = self._skins_dir / f"{name}.yml"
            if skin_file.exists():
                new_mtime = skin_file.stat().st_mtime
                if new_mtime > mtime:
                    skin = self._load_yaml_skin(skin_file)
                    if skin:
                        self._loaded_skins[name] = skin
                        self._file_timestamps[name] = new_mtime
                        changed = True
        if changed and self._current_name in self._loaded_skins:
            self._current = self._loaded_skins[self._current_name]
        return changed

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def current(self) -> dict[str, Any]:
        return self._current

    @property
    def current_name(self) -> str:
        return self._current_name

    def get(self, key: str, default: Any = None) -> Any:
        return self._current.get(key, default)

    def style(self, key: str) -> str:
        """Get a color/style string for Rich markup."""
        return self._current.get(key, "white")

    def box(self) -> Box:
        return _resolve_box(self._current.get("box", "rounded"))

    def spinner(self) -> str:
        return self._current.get("spinner", "dots")

    def panel_padding(self) -> int:
        return int(self._current.get("panel_padding", 1))

    # ── Listing ───────────────────────────────────────────────────────────

    def list_skins(self) -> list[dict[str, Any]]:
        """List all available skins with metadata."""
        skins = []
        for name, skin in self._loaded_skins.items():
            skins.append({
                "name": name,
                "label": skin.get("name", name),
                "description": skin.get("description", ""),
                "source": "built-in" if name in BUILTIN_SKINS else "custom",
                "loaded": name == self._current_name,
            })
        return sorted(skins, key=lambda s: (not s["loaded"], s["name"]))

    def export_current(self) -> dict[str, Any]:
        """Export current skin as a YAML-compatible dict."""
        return dict(self._current)
