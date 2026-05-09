"""Device Pairing — secure device-to-agent pairing with QR codes.

Supports multiple auth flows:
  - QR code pairing (scan to connect)
  - Verification code (enter code to authorize)
  - OAuth 2.0 device flow (RFC 8628)
  - Manual token entry

For platforms like Telegram/WeChat/Slack that require device registration,
this module handles the pairing lifecycle.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PairingState(str, Enum):
    PENDING = "pending"       # Waiting for user to confirm
    CONFIRMED = "confirmed"   # User approved the pairing
    REJECTED = "rejected"     # User denied the pairing
    EXPIRED = "expired"       # Pairing request timed out
    COMPLETED = "completed"   # Pairing successfully established
    ERROR = "error"           # Pairing failed


@dataclass
class PairingRequest:
    """A single pairing request with unique code and timeout."""

    id: str
    platform: str
    code: str  # Human-readable verification code
    secret: str  # HMAC secret for verification
    state: PairingState = PairingState.PENDING
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0
    confirmed_at: float = 0
    device_info: dict[str, str] = field(default_factory=dict)
    credentials: dict[str, str] = field(default_factory=dict)  # Stored after completion

    def __post_init__(self):
        if not self.expires_at:
            self.expires_at = self.created_at + 600  # 10-minute default

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.expires_at - time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "code": self.code,
            "state": self.state.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "remaining_s": round(self.remaining_seconds, 1),
            "device_info": self.device_info,
        }

    def to_qr_data(self, base_url: str = "") -> str:
        """Generate a QR code data string (URL format)."""
        if base_url:
            return f"{base_url}/pair?id={self.id}&code={self.code}"
        return json.dumps({
            "type": "kairos_pairing",
            "id": self.id,
            "code": self.code,
            "platform": self.platform,
        })

    def verify_code(self, code: str) -> bool:
        """Verify if a provided code matches."""
        return hmac.compare_digest(code, self.code)


class PairingManager:
    """Manage device pairing lifecycle.

    Usage:
        pm = PairingManager()
        request = pm.create_request("telegram", device_info={"bot_name": "mybot"})

        # User scans QR or enters code
        qr_url = request.to_qr_data("https://kairos.example.com")

        # Verify and complete:
        if pm.verify("telegram", code):
            pm.complete(request.id, credentials={"bot_token": "xxx"})
    """

    def __init__(
        self,
        default_timeout: float = 600,  # 10 minutes
        max_active_per_platform: int = 5,
        code_length: int = 6,
    ):
        self._default_timeout = default_timeout
        self._max_active = max_active_per_platform
        self._code_length = code_length
        self._requests: dict[str, PairingRequest] = {}  # id → request
        self._credentials: dict[str, dict[str, str]] = {}  # platform → credentials

    # ── Request lifecycle ─────────────────────────────────────────────────

    def create_request(
        self,
        platform: str,
        device_info: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> PairingRequest:
        """Create a new pairing request. Returns the request with code + QR data."""
        # Clean expired requests
        self._clean_expired()

        # Limit active requests per platform
        active = [
            r for r in self._requests.values()
            if r.platform == platform and r.state == PairingState.PENDING
        ]
        if len(active) >= self._max_active:
            # Cancel oldest pending
            oldest = min(active, key=lambda r: r.created_at)
            oldest.state = PairingState.EXPIRED

        request_id = uuid.uuid4().hex[:16]
        code = self._generate_code()
        secret = secrets.token_hex(16)

        request = PairingRequest(
            id=request_id,
            platform=platform,
            code=code,
            secret=secret,
            expires_at=time.time() + (timeout or self._default_timeout),
            device_info=device_info or {},
        )
        self._requests[request_id] = request
        return request

    def verify(self, request_id: str, code: str) -> PairingState:
        """Verify a pairing code. Returns the resulting state."""
        request = self._requests.get(request_id)
        if not request:
            return PairingState.ERROR

        if request.is_expired:
            request.state = PairingState.EXPIRED
            return PairingState.EXPIRED

        if request.state != PairingState.PENDING:
            return request.state

        if request.verify_code(code):
            request.state = PairingState.CONFIRMED
            request.confirmed_at = time.time()
            return PairingState.CONFIRMED

        return PairingState.PENDING

    def reject(self, request_id: str) -> bool:
        """Reject a pairing request."""
        request = self._requests.get(request_id)
        if not request:
            return False
        request.state = PairingState.REJECTED
        return True

    def complete(
        self,
        request_id: str,
        credentials: dict[str, str],
    ) -> PairingRequest | None:
        """Complete a confirmed pairing request and store credentials.

        Args:
            request_id: The request ID
            credentials: Platform-specific credentials (e.g., bot_token, api_key)

        Returns the updated request or None if not found/invalid state.
        """
        request = self._requests.get(request_id)
        if not request:
            return None

        if request.state != PairingState.CONFIRMED:
            return None

        request.state = PairingState.COMPLETED
        request.credentials = credentials
        self._credentials[request.platform] = credentials
        return request

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending pairing request."""
        request = self._requests.get(request_id)
        if not request:
            return False
        if request.state == PairingState.PENDING:
            request.state = PairingState.EXPIRED
            return True
        return False

    # ── Query ─────────────────────────────────────────────────────────────

    def get_request(self, request_id: str) -> PairingRequest | None:
        """Get a pairing request by ID."""
        return self._requests.get(request_id)

    def get_credentials(self, platform: str) -> dict[str, str] | None:
        """Get stored credentials for a platform."""
        return self._credentials.get(platform)

    def list_pending(self, platform: str | None = None) -> list[PairingRequest]:
        """List all pending pairing requests, optionally filtered by platform."""
        self._clean_expired()
        pending = [
            r for r in self._requests.values()
            if r.state == PairingState.PENDING
        ]
        if platform:
            pending = [r for r in pending if r.platform == platform]
        return pending

    def stats(self) -> dict[str, Any]:
        """Get pairing statistics."""
        self._clean_expired()
        states = {s.value: 0 for s in PairingState}
        for r in self._requests.values():
            states[r.state.value] = states.get(r.state.value, 0) + 1

        platforms = set(r.platform for r in self._requests.values())
        return {
            "total_requests": len(self._requests),
            "by_state": states,
            "credentials_stored": list(self._credentials.keys()),
            "platforms": list(platforms),
        }

    # ── OAuth Device Flow ─────────────────────────────────────────────────

    def oauth_device_flow(
        self,
        platform: str,
        device_code_url: str = "",
        poll_interval: int = 5,
    ) -> dict[str, Any]:
        """RFC 8628 Device Authorization Grant for OAuth platforms.

        Returns device_code data suitable for polling the authorization server.
        """
        device_code = secrets.token_urlsafe(32)
        user_code = self._generate_code(upper=True)

        return {
            "device_code": device_code,
            "user_code": user_code,
            "verification_uri": device_code_url,
            "verification_uri_complete": f"{device_code_url}?code={user_code}" if device_code_url else "",
            "expires_in": int(self._default_timeout),
            "interval": poll_interval,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _generate_code(self, upper: bool = False) -> str:
        """Generate a human-readable verification code."""
        if upper:
            chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        else:
            chars = "0123456789"
            # For numeric-only codes, use secure randbelow
        code = "".join(
            chars[secrets.randbelow(len(chars))]
            for _ in range(self._code_length)
        )
        return code

    def _clean_expired(self) -> int:
        """Remove expired or completed requests. Returns count cleaned."""
        now = time.time()
        to_remove = [
            rid for rid, r in self._requests.items()
            if r.is_expired or r.state in (PairingState.COMPLETED, PairingState.REJECTED)
        ]
        for rid in to_remove:
            del self._requests[rid]
        return len(to_remove)

    def __repr__(self) -> str:
        return (
            f"PairingManager(requests={len(self._requests)}, "
            f"credentials={list(self._credentials.keys())})"
        )
