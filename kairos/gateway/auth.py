"""Gateway authorization — per-platform whitelists, DM pairing, allow-all mode.

Hermes-compatible multi-layer auth:
  1. Global allow-all: if True, skip all auth
  2. Per-platform allow-all: if True, skip that platform's auth
  3. Per-platform allowlist: only listed chat_ids can interact
  4. DM pairing: unlisted users receive a pairing request
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("kairos.gateway.auth")


# ============================================================================
# Models
# ============================================================================


@dataclass
class AuthResult:
    """Result of an authorization check."""

    allowed: bool
    reason: str = ""
    pairing_required: bool = False
    pairing_code: str | None = None


@dataclass
class PlatformAuth:
    """Per-platform authorization config."""

    name: str
    allowlist: list[str] = field(default_factory=list)     # Allowed chat_ids
    allow_all: bool = False                                  # Skip auth for this platform
    allow_dm_pairing: bool = True                            # Show pairing to unlisted users
    pairing_code_length: int = 6                             # Length of pairing code


# ============================================================================
# GatewayAuth
# ============================================================================


class GatewayAuth:
    """Multi-platform authorization layer for the Kairos Gateway.

    Usage:
        auth = GatewayAuth()
        auth.configure_platform("telegram", allowlist=["123", "456"])
        result = auth.check("telegram", chat_id="123", user_id="alice")
    """

    def __init__(self, global_allow_all: bool = False):
        self._platforms: dict[str, PlatformAuth] = {}
        self._global_allow_all = global_allow_all
        self._pending_pairings: dict[str, str] = {}   # pairing_code → chat_id
        self._paired_users: set[str] = set()           # Set of paired chat_ids

    # ── Configuration ───────────────────────────────────────

    def configure_platform(
        self,
        name: str,
        allowlist: list[str] | None = None,
        allow_all: bool = False,
        allow_dm_pairing: bool = True,
        pairing_code_length: int = 6,
    ) -> PlatformAuth:
        """Configure auth for a platform. Overwrites existing config."""
        auth = PlatformAuth(
            name=name,
            allowlist=allowlist or [],
            allow_all=allow_all,
            allow_dm_pairing=allow_dm_pairing,
            pairing_code_length=pairing_code_length,
        )
        self._platforms[name] = auth
        return auth

    def remove_platform(self, name: str) -> None:
        """Remove auth config for a platform."""
        self._platforms.pop(name, None)

    def set_global_allow_all(self, value: bool) -> None:
        """Enable/disable global allow-all mode (skip all auth)."""
        self._global_allow_all = value

    # ── Auth check ─────────────────────────────────────────

    def check(
        self,
        platform: str,
        chat_id: str = "",
        user_id: str = "",
    ) -> AuthResult:
        """Check if a chat/user is authorized to interact.

        Returns AuthResult with allowed=True/False and pairing info.
        """
        # 1. Global allow-all
        if self._global_allow_all:
            return AuthResult(allowed=True, reason="global_allow_all")

        # 2. Platform config
        pa = self._platforms.get(platform)
        if not pa:
            logger.warning("No auth config for platform '%s', denying", platform)
            return AuthResult(allowed=False, reason="no_platform_config")

        # 3. Platform allow-all
        if pa.allow_all:
            return AuthResult(allowed=True, reason="platform_allow_all")

        # 4. Allowlist check
        identifier = chat_id or user_id
        if identifier in pa.allowlist:
            return AuthResult(allowed=True, reason="allowlist")

        # 5. Already paired
        if identifier in self._paired_users:
            return AuthResult(allowed=True, reason="paired")

        # 6. DM pairing
        if pa.allow_dm_pairing:
            import secrets
            code = self._generate_pairing_code(pa.pairing_code_length)
            self._pending_pairings[code] = identifier
            logger.info(
                "Pairing required for %s/%s, code=%s", platform, identifier, code,
            )
            return AuthResult(
                allowed=False,
                reason="pairing_required",
                pairing_required=True,
                pairing_code=code,
            )

        return AuthResult(allowed=False, reason="not_in_allowlist")

    # ── Pairing ────────────────────────────────────────────

    def confirm_pairing(self, code: str) -> str | None:
        """Confirm a pairing code. Returns the chat_id if valid."""
        chat_id = self._pending_pairings.pop(code, None)
        if chat_id:
            self._paired_users.add(chat_id)
            logger.info("Pairing confirmed: %s", chat_id)
        return chat_id

    def revoke_pairing(self, chat_id: str) -> None:
        """Revoke a previously paired user."""
        self._paired_users.discard(chat_id)
        # Also remove from all platform allowlists
        for pa in self._platforms.values():
            if chat_id in pa.allowlist:
                pa.allowlist.remove(chat_id)

    @staticmethod
    def _generate_pairing_code(length: int) -> str:
        import secrets
        import string
        alphabet = string.ascii_uppercase + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    # ── Properties ─────────────────────────────────────────

    @property
    def platform_names(self) -> list[str]:
        return list(self._platforms.keys())

    def get_platform(self, name: str) -> PlatformAuth | None:
        return self._platforms.get(name)

    def is_allowed(self, platform: str, chat_id: str = "", user_id: str = "") -> bool:
        """Quick boolean check."""
        return self.check(platform, chat_id, user_id).allowed

    # ── Serialization ──────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_allow_all": self._global_allow_all,
            "platforms": {
                name: {
                    "allowlist": pa.allowlist,
                    "allow_all": pa.allow_all,
                    "allow_dm_pairing": pa.allow_dm_pairing,
                }
                for name, pa in self._platforms.items()
            },
            "paired_users": list(self._paired_users),
            "pending_pairings": len(self._pending_pairings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GatewayAuth:
        auth = cls(global_allow_all=data.get("global_allow_all", False))
        for name, cfg in data.get("platforms", {}).items():
            auth.configure_platform(
                name=name,
                allowlist=cfg.get("allowlist", []),
                allow_all=cfg.get("allow_all", False),
                allow_dm_pairing=cfg.get("allow_dm_pairing", True),
            )
        auth._paired_users = set(data.get("paired_users", []))
        return auth

    def __repr__(self) -> str:
        return (
            f"GatewayAuth(platforms={len(self._platforms)}, "
            f"paired={len(self._paired_users)}, "
            f"global_allow_all={self._global_allow_all})"
        )
