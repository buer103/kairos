"""Tests for skill management: SkillManager lifecycle + SkillMarketplace."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kairos.skills.manager import SkillManager, SkillStatus, SkillEntry
from kairos.skills.marketplace import SkillMarketplace


class TestSkillManager:
    """Skill lifecycle management."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a SkillManager with a temp skills directory."""
        return SkillManager(skills_dir=tmp_path / "skills")

    def test_initial_state(self, manager):
        """Fresh SkillManager has empty index."""
        assert manager.stats()["total"] == 0

    def test_create_skill(self, manager):
        """Creating a skill registers it in the index."""
        path = manager.create(
            name="test-skill",
            content="# Test Content\n\nThis is a test skill.",
            description="A test skill",
        )
        assert path.exists()
        assert manager.stats()["total"] == 1
        assert manager.stats()["active"] == 1

    def test_get_skill(self, manager):
        """Can retrieve a skill entry."""
        manager.create("my-skill", "# Content", "desc")
        entry = manager.get("my-skill")
        assert entry is not None
        assert entry.name == "my-skill"
        assert entry.description == "desc"

    def test_update_skill(self, manager):
        """Updating creates a backup and modifies content."""
        manager.create("upd-skill", "# Old", "old desc")
        path = manager.update("upd-skill", content="# New", description="new desc")

        assert path is not None
        content = path.read_text()
        assert "# New" in content
        assert "new desc" in content or "description: new desc" in content

    def test_update_nonexistent(self, manager):
        """Updating a nonexistent skill returns None."""
        assert manager.update("ghost", content="x") is None

    def test_delete_and_archive(self, manager):
        """Deleting archives the skill."""
        manager.create("del-skill", "# To delete")
        assert manager.stats()["total"] == 1

        ok = manager.delete("del-skill")
        assert ok
        assert manager.get("del-skill").status == SkillStatus.ARCHIVED
        assert manager.stats()["archived"] == 1
        assert manager.stats()["active"] == 0

    def test_delete_nonexistent(self, manager):
        """Deleting a nonexistent skill returns False."""
        assert manager.delete("ghost") is False

    def test_mark_used(self, manager):
        """mark_used updates last_used_at and use_count."""
        manager.create("used-skill", "# Content")
        assert manager.mark_used("used-skill")
        entry = manager.get("used-skill")
        assert entry.use_count == 1
        assert entry.last_used_at is not None

    def test_mark_used_unknown(self, manager):
        """mark_used on unknown skill returns False."""
        assert manager.mark_used("ghost") is False

    def test_mark_stale(self, manager):
        """Stale detection flags unused skills."""
        manager.create("old-skill", "# Old")
        entry = manager.get("old-skill")
        # Set last_used_at to 60 days ago
        entry.last_used_at = time.time() - 60 * 86400
        manager._save_index()

        stale = manager.mark_stale(days=30)
        assert "old-skill" in stale
        assert manager.get("old-skill").status == SkillStatus.STALE

    def test_mark_used_reactivates_stale(self, manager):
        """Using a stale skill reactivates it."""
        manager.create("revive", "# Revive")
        entry = manager.get("revive")
        entry.last_used_at = time.time() - 60 * 86400
        manager._save_index()
        manager.mark_stale(days=30)
        assert manager.get("revive").status == SkillStatus.STALE

        manager.mark_used("revive")
        assert manager.get("revive").status == SkillStatus.ACTIVE

    def test_scan_discovers_new_skills(self, manager):
        """Scan finds SKILL.md files on disk and adds them to index."""
        # Manually create a SKILL.md file outside of manager.create()
        # Note: built-in skills are auto-copied on init, so scan may find existing ones
        pre_count = len(manager.get_all_skills())
        skill_dir = manager._dir / "manual-scan-skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: manual-scan-skill\ndescription: Found by scan\n---\n\n# Manual\n"
        )

        result = manager.scan()
        assert result["added"] >= 1
        assert manager.get("manual-scan-skill") is not None

    def test_reindex_removes_missing(self, manager):
        """Reindex removes entries whose files are gone."""
        manager.create("temp-skill", "# Temp")
        assert manager.get("temp-skill") is not None

        # Remove the file manually
        manager.get("temp-skill").path.unlink()
        manager.get("temp-skill").path.parent.rmdir()

        result = manager.reindex()
        assert result["removed"] >= 1
        assert manager.get("temp-skill") is None

    def test_clean_removes_old_archives(self, manager):
        """Clean removes archived entries older than threshold."""
        manager.create("clean-me", "# Clean")
        manager.delete("clean-me")

        stats = manager.stats()
        assert stats["backups"] >= 1

        # Make backups artificially old by touching with old mtime
        for item in manager._backup_dir.iterdir():
            old_time = time.time() - 365 * 86400  # 1 year ago
            os.utime(item, (old_time, old_time))

        # Clean with 0 days (everything is old)
        result = manager.clean(days=0)
        assert result["cleaned"] >= 1

    def test_get_all_skills(self, manager):
        """get_all_skills returns active/stale skills with metadata."""
        manager.create("alpha", "# A", "First")
        manager.create("beta", "# B", "Second")

        # Make beta stale
        entry = manager.get("beta")
        entry.last_used_at = time.time() - 60 * 86400
        manager._save_index()
        manager.mark_stale(days=30)

        all_skills = manager.get_all_skills()
        assert len(all_skills) == 2  # Both active and stale
        names = [s["name"] for s in all_skills]
        assert "alpha" in names
        assert "beta" in names

    def test_get_skill_content(self, manager):
        """get_skill_content returns full skill data."""
        manager.create("full", "# Full Content Here\n\nSome text.", "Full desc")

        content = manager.get_skill_content("full")
        assert content is not None
        assert content["name"] == "full"
        assert "Full Content Here" in content["content"]
        assert content["description"] == "Full desc"
        assert "skill_dir" in content
        assert "linked_files" in content

    def test_get_skill_content_unavailable(self, manager):
        """get_skill_content returns None for missing/archived skills."""
        assert manager.get_skill_content("ghost") is None

    def test_list_files(self, manager):
        """list_files returns file listing for a skill."""
        manager.create("files-skill", "# Files", "desc")
        # Create a references file
        skill_dir = manager.get("files-skill").path.parent
        refs = skill_dir / "references"
        refs.mkdir(exist_ok=True)
        (refs / "guide.md").write_text("# Guide")

        files = manager.list_files("files-skill")
        assert files is not None
        assert "files" in files
        assert "references" in files["files"]

    def test_list_files_access_file(self, manager):
        """list_files with file_path returns file content."""
        manager.create("file-skill", "# Files", "desc")
        skill_dir = manager.get("file-skill").path.parent
        refs = skill_dir / "references"
        refs.mkdir(exist_ok=True)
        (refs / "guide.md").write_text("# Guide Content")

        result = manager.list_files("file-skill", "references/guide.md")
        assert result is not None
        assert "Guide Content" in result["content"]

    def test_list_files_prevents_traversal(self, manager):
        """list_files blocks path traversal attempts."""
        manager.create("safe-skill", "# Safe")
        result = manager.list_files("safe-skill", "../../etc/passwd")
        assert result is not None
        assert "error" in result

    def test_list_categories(self, manager):
        """list_categories returns unique categories."""
        manager.create("cat-skill", "# C", category="testing")
        categories = manager.list_categories()
        assert "testing" in categories

    def test_resolve_forwarding(self, manager):
        """resolve_forwarding finds the absorption target."""
        manager.create("old-skill", "# Old")
        manager.delete("old-skill", absorbed_into="new-skill")

        target = manager.resolve_forwarding("old-skill")
        assert target is not None
        assert "new-skill" in target

    def test_index_persistence(self, tmp_path):
        """Index survives recreation of SkillManager."""
        m1 = SkillManager(skills_dir=tmp_path / "skills")
        m1.create("persist", "# P", "Persistent")
        assert m1.stats()["total"] == 1

        m2 = SkillManager(skills_dir=tmp_path / "skills")
        assert m2.stats()["total"] == 1
        assert m2.get("persist") is not None


class TestSkillMarketplace:
    """Skill installation from various sources."""

    @pytest.fixture
    def mp(self, tmp_path):
        """Create a marketplace with temp skills dir."""
        manager = SkillManager(skills_dir=tmp_path / "skills")
        return SkillMarketplace(manager)

    def test_install_local_skill(self, mp):
        """Installing from a local directory copies the skill."""
        # Create a local skill directory
        local = Path(mp._dir) / "test-local"
        local = local.resolve()
        src = local.parent / "my-local-skill"
        src.mkdir(parents=True, exist_ok=True)
        (src / "SKILL.md").write_text(
            "---\nname: my-local-skill\ndescription: Local test skill\n---\n\n# Content\n"
        )

        result = mp.install(str(src))
        assert result["success"]
        assert result["name"] == "my-local-skill"

    def test_install_already_exists(self, mp):
        """Installing a skill that already exists fails."""
        src = Path(mp._dir).parent / "dup-skill"
        src.mkdir(parents=True, exist_ok=True)
        (src / "SKILL.md").write_text("---\nname: dup-skill\n---\n\n# Dup\n")

        mp.install(str(src))
        result = mp.install(str(src))
        assert not result["success"]
        assert "already installed" in result["error"]

    def test_uninstall_skill(self, mp):
        """Uninstalling removes from marketplace."""
        src = Path(mp._dir).parent / "uninstall-me"
        src.mkdir(parents=True, exist_ok=True)
        (src / "SKILL.md").write_text("---\nname: uninstall-me\n---\n\n# Gone\n")

        mp.install(str(src))
        result = mp.uninstall("uninstall-me")
        assert result["success"]

    def test_uninstall_nonexistent(self, mp):
        """Uninstalling a nonexistent skill fails."""
        result = mp.uninstall("ghost")
        assert not result["success"]

    def test_list_marketplace(self, mp):
        """list_marketplace returns installed marketplace skills."""
        src = Path(mp._dir).parent / "market-skill"
        src.mkdir(parents=True, exist_ok=True)
        (src / "SKILL.md").write_text(
            "---\nname: market-skill\nversion: 1.0.0\ndescription: Market test\n---\n\n# Market\n"
        )

        mp.install(str(src))
        skills = mp.list_marketplace()
        assert len(skills) == 1
        assert skills[0]["name"] == "market-skill"
        assert skills[0]["version"] == "1.0.0"

    def test_install_invalid_source(self, mp):
        """Invalid source returns error."""
        result = mp.install("not-a-valid-source-format")
        assert not result["success"]

    def test_install_nonexistent_local(self, mp):
        """Nonexistent local path returns error."""
        result = mp.install("/tmp/definitely-does-not-exist-skill-dir")
        assert not result["success"]
