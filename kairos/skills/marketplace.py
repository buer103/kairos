"""Skill Marketplace — install skills from GitHub, URL, or local directory.

Usage:
    mp = SkillMarketplace(skill_manager)
    mp.install("github.com/user/repo")
    mp.install("~/my-skills/custom-skill")
    mp.install("https://example.com/skill.tar.gz")
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kairos.skills.manager import SkillManager


class SkillMarketplace:
    """Install skills from various sources into the skills directory.

    Sources supported:
      - github.com/user/repo          (clone the repo)
      - github.com/user/repo/path     (clone repo, use sub-path)
      - local directory path          (copy skill dir)
      - URL to .tar.gz or .zip        (download + extract)
      - huggingface://org/skill       (download from HF Hub)
    """

    def __init__(self, skill_manager: SkillManager | None = None):
        self._manager = skill_manager or SkillManager()
        self._dir = self._manager._dir
        self._marketplace_dir = self._dir / "marketplace"
        self._marketplace_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────

    def install(self, source: str, name: str | None = None) -> dict[str, Any]:
        """Install a skill from a source. Returns installation result."""
        source = source.strip()

        # Detect source type
        if source.startswith("github.com/") or source.startswith("https://github.com/"):
            return self._install_github(source, name)
        elif source.startswith("huggingface://"):
            return self._install_huggingface(source, name)
        elif source.startswith("http://") or source.startswith("https://"):
            return self._install_url(source, name)
        elif os.path.isdir(os.path.expanduser(source)):
            return self._install_local(source, name)
        else:
            # Try as GitHub shorthand
            if "/" in source and not source.startswith("/"):
                return self._install_github(source, name)
            return {"success": False, "error": f"Unrecognized source: {source}"}

    def uninstall(self, name: str) -> dict[str, Any]:
        """Uninstall a marketplace skill (move to backup)."""
        # Find in marketplace dir
        for skill_dir in self._marketplace_dir.iterdir():
            if skill_dir.is_dir() and skill_dir.name == name:
                return self._remove_skill(name, skill_dir)

        # Try subdirectories
        for part1 in self._marketplace_dir.iterdir():
            if not part1.is_dir():
                continue
            for part2 in part1.iterdir():
                if part2.is_dir() and part2.name == name:
                    return self._remove_skill(name, part2)
                for part3 in part2.iterdir():
                    if part3.is_dir() and part3.name == name:
                        return self._remove_skill(name, part3)

        return {"success": False, "error": f"Skill '{name}' not found in marketplace"}

    def list_marketplace(self) -> list[dict[str, Any]]:
        """List all installed marketplace skills."""
        skills = []
        for skill_md in sorted(self._marketplace_dir.glob("**/SKILL.md")):
            try:
                content = skill_md.read_text(encoding="utf-8")
                name = skill_md.parent.name
                description = ""
                version = ""
                source = ""

                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("description:"):
                        description = line.split(":", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("version:"):
                        version = line.split(":", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("source:"):
                        source = line.split(":", 1)[1].strip().strip('"').strip("'")

                skills.append({
                    "name": name,
                    "description": description,
                    "version": version,
                    "source": source,
                    "path": str(skill_md.parent),
                })
            except Exception:
                pass
        return skills

    def update(self, name: str) -> dict[str, Any]:
        """Update a marketplace skill to the latest version."""
        # Find the skill
        skill_path = None
        source = None
        for skill_md in self._marketplace_dir.glob(f"**/{name}/SKILL.md"):
            skill_path = skill_md
            break

        if not skill_path:
            return {"success": False, "error": f"Skill '{name}' not found in marketplace"}

        # Read source URL
        try:
            content = skill_path.read_text(encoding="utf-8")
            for line in content.split("\n"):
                if line.strip().startswith("source:"):
                    source = line.split(":", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass

        if not source:
            return {"success": False, "error": "No source URL found in skill metadata"}

        # Uninstall old, install fresh
        self.uninstall(name)
        return self.install(source, name)

    def update_all(self) -> list[dict[str, Any]]:
        """Update all marketplace skills."""
        results = []
        for skill in self.list_marketplace():
            result = self.update(skill["name"])
            results.append(result)
        return results

    # ── Install Methods ────────────────────────────────────────

    def _install_github(self, source: str, name: str | None) -> dict[str, Any]:
        """Install from a GitHub repository."""
        # Normalize: github.com/user/repo or github.com/user/repo/path
        clean = source.replace("https://", "").replace("http://", "").rstrip("/")
        if clean.startswith("github.com/"):
            clean = clean[11:]
        parts = clean.split("/")
        if len(parts) < 2:
            return {"success": False, "error": f"Invalid GitHub source: {source}"}

        user = parts[0]
        repo = parts[1]
        subpath = "/".join(parts[2:]) if len(parts) > 2 else ""

        gh_url = f"https://github.com/{user}/{repo}.git"
        skill_name = name or (subpath.split("/")[-1] if subpath else repo)
        dest_dir = self._marketplace_dir / "github.com" / user / skill_name

        if dest_dir.exists():
            return {"success": False, "error": f"Skill '{skill_name}' already installed at {dest_dir}"}

        # Clone to temp dir
        tmp = tempfile.mkdtemp(prefix="kairos-skill-")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", gh_url, tmp],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            if result.returncode != 0:
                return {"success": False, "error": f"Git clone failed: {result.stderr}"}

            skill_source = Path(tmp)
            if subpath:
                skill_source = skill_source / subpath
                if not skill_source.exists():
                    return {"success": False, "error": f"Sub-path not found: {subpath}"}

            # Find or create SKILL.md
            skill_md = skill_source / "SKILL.md"
            if not skill_md.exists():
                return {"success": False, "error": f"No SKILL.md found in {skill_source}"}

            # Copy skill to marketplace
            dest_dir.mkdir(parents=True, exist_ok=True)
            self._copy_dir(skill_source, dest_dir)

            # Add source metadata to SKILL.md
            self._add_source_metadata(dest_dir / "SKILL.md", f"github.com/{user}/{repo}")
            if subpath:
                self._add_source_metadata(dest_dir / "SKILL.md", f"github.com/{user}/{repo}/{subpath}")

            # Reindex
            self._manager.scan()
            self._manager.mark_used(skill_name)

            return {
                "success": True,
                "name": skill_name,
                "path": str(dest_dir),
                "source": f"github.com/{user}/{repo}",
            }
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _install_huggingface(self, source: str, name: str | None) -> dict[str, Any]:
        """Install from HuggingFace Hub."""
        # huggingface://org/skill-name
        clean = source.replace("huggingface://", "").rstrip("/")
        parts = clean.split("/")
        if len(parts) < 2:
            return {"success": False, "error": f"Invalid HF source: {source}"}

        org = parts[0]
        skill_name = name or parts[1]
        dest_dir = self._marketplace_dir / "huggingface.co" / org / skill_name

        if dest_dir.exists():
            return {"success": False, "error": f"Skill '{skill_name}' already installed"}

        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                f"{org}/{skill_name}",
                local_dir=str(dest_dir),
                repo_type="space",
            )
        except ImportError:
            # Fallback: download via requests
            import urllib.request
            url = f"https://huggingface.co/{org}/{skill_name}/resolve/main/SKILL.md"
            try:
                urllib.request.urlretrieve(url, str(dest_dir / "SKILL.md"))
            except Exception as e:
                shutil.rmtree(dest_dir, ignore_errors=True)
                return {"success": False, "error": f"HF download failed: {e}"}
        except Exception as e:
            shutil.rmtree(dest_dir, ignore_errors=True)
            return {"success": False, "error": f"HF install failed: {e}"}

        self._add_source_metadata(dest_dir / "SKILL.md", f"huggingface://{org}/{skill_name}")
        self._manager.scan()
        self._manager.mark_used(skill_name)

        return {
            "success": True,
            "name": skill_name,
            "path": str(dest_dir),
            "source": f"huggingface://{org}/{skill_name}",
        }

    def _install_url(self, source: str, name: str | None) -> dict[str, Any]:
        """Install from a URL to .tar.gz or .zip."""
        import urllib.request

        parsed = urlparse(source)
        filename = os.path.basename(parsed.path) or "skill"
        skill_name = name or filename.rsplit(".", 1)[0].replace(".tar", "")

        dest_dir = self._marketplace_dir / "url" / skill_name
        if dest_dir.exists():
            return {"success": False, "error": f"Skill '{skill_name}' already installed"}

        tmp = tempfile.mkdtemp(prefix="kairos-skill-dl-")
        try:
            dl_path = os.path.join(tmp, filename)
            urllib.request.urlretrieve(source, dl_path)

            # Extract
            extract_dir = os.path.join(tmp, "extracted")
            os.makedirs(extract_dir, exist_ok=True)

            if dl_path.endswith(".tar.gz") or dl_path.endswith(".tgz"):
                with tarfile.open(dl_path, "r:gz") as tf:
                    tf.extractall(extract_dir)
            elif dl_path.endswith(".zip"):
                with zipfile.ZipFile(dl_path, "r") as zf:
                    zf.extractall(extract_dir)
            else:
                return {"success": False, "error": f"Unsupported archive format: {dl_path}"}

            # Find SKILL.md in extracted content
            skill_source = None
            for p in Path(extract_dir).rglob("SKILL.md"):
                skill_source = p.parent
                break

            if not skill_source:
                return {"success": False, "error": "No SKILL.md found in archive"}

            dest_dir.mkdir(parents=True, exist_ok=True)
            self._copy_dir(skill_source, dest_dir)
            self._add_source_metadata(dest_dir / "SKILL.md", source)
            self._manager.scan()
            self._manager.mark_used(skill_name)

            return {
                "success": True,
                "name": skill_name,
                "path": str(dest_dir),
                "source": source,
            }
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _install_local(self, source: str, name: str | None) -> dict[str, Any]:
        """Install from a local directory."""
        src_path = Path(os.path.expanduser(source)).resolve()
        if not src_path.is_dir():
            return {"success": False, "error": f"Not a directory: {source}"}

        # Find SKILL.md
        skill_md = src_path / "SKILL.md"
        if not skill_md.exists():
            return {"success": False, "error": f"No SKILL.md found in {source}"}

        skill_name = name or src_path.name
        dest_dir = self._marketplace_dir / "local" / skill_name

        if dest_dir.exists():
            return {"success": False, "error": f"Skill '{skill_name}' already installed"}

        dest_dir.mkdir(parents=True, exist_ok=True)
        self._copy_dir(src_path, dest_dir)
        self._add_source_metadata(dest_dir / "SKILL.md", f"local://{src_path}")
        self._manager.scan()
        self._manager.mark_used(skill_name)

        return {
            "success": True,
            "name": skill_name,
            "path": str(dest_dir),
            "source": f"local://{src_path}",
        }

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _copy_dir(src: Path, dst: Path) -> None:
        """Copy a directory tree ignoring .git and __pycache__."""
        for item in src.rglob("*"):
            if ".git" in item.parts or "__pycache__" in item.parts:
                continue
            rel = item.relative_to(src)
            target = dst / rel
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)

    @staticmethod
    def _add_source_metadata(skill_md: Path, source: str) -> None:
        """Insert source metadata into SKILL.md frontmatter."""
        if not skill_md.exists():
            return
        try:
            content = skill_md.read_text(encoding="utf-8")
            if "source:" in content:
                return  # Already has source

            # Insert after version or name line
            lines = content.split("\n")
            insert_at = None
            for i, line in enumerate(lines):
                if line.strip().startswith("version:"):
                    insert_at = i + 1
                    break
            if insert_at is None:
                for i, line in enumerate(lines):
                    if line.strip().startswith("name:"):
                        insert_at = i + 1
                        break

            if insert_at:
                lines.insert(insert_at, f"source: {source}")
                skill_md.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    def _remove_skill(self, name: str, skill_dir: Path) -> dict[str, Any]:
        """Remove a skill from marketplace (archive it)."""
        # Archive to backup dir before removing
        self._manager.delete(name)
        if skill_dir.exists():
            shutil.rmtree(skill_dir, ignore_errors=True)

        # Remove parent empty dirs
        parent = skill_dir.parent
        while parent != self._marketplace_dir:
            try:
                if not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                break
            parent = parent.parent

        self._manager.scan()
        return {"success": True, "name": name, "removed": str(skill_dir)}
