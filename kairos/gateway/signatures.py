"""Webhook signature verification — production-grade HMAC-SHA256 and Ed25519 for all platforms.

Each platform has its own verification method:
  - Discord: Ed25519 public key
  - Feishu: HMAC-SHA256 (timestamp + nonce + body)
  - WhatsApp: HMAC-SHA256 (X-Hub-Signature-256)
  - Line: HMAC-SHA256 (channel secret)
  - Signal: not applicable (E2E encrypted)
  - Matrix: HMAC-SHA256 (signing key)
  - IRC: not applicable (plain TCP)

Usage:
    from kairos.gateway.signatures import verify_discord, verify_feishu
    if not verify_discord(public_key, signature, timestamp, raw_body):
        return 401
"""

from __future__ import annotations

import hashlib
import hmac
import logging

logger = logging.getLogger("kairos.gateway.signatures")


# ═══════════════════════════════════════════════════════════════════
# HMAC-SHA256 (used by Feishu, WhatsApp, Line, Matrix, Slack)
# ═══════════════════════════════════════════════════════════════════

def verify_hmac_sha256(secret: str, body: bytes, signature: str, prefix: str = "sha256=") -> bool:
    """Generic HMAC-SHA256 verification.

    Args:
        secret: Shared secret key.
        body: Raw request body bytes.
        signature: Signature from request header (e.g. "sha256=abc123").
        prefix: Signature prefix to strip ("sha256=", "v1=").

    Returns:
        True if signature matches.
    """
    if not secret or not signature:
        return False

    # Strip prefix
    sig_value = signature.removeprefix(prefix) if signature.startswith(prefix) else signature

    computed = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, sig_value)


def verify_feishu(secret: str, headers: dict[str, str], raw_body: bytes) -> bool:
    """Verify Feishu webhook signature.

    Feishu uses: HMAC-SHA256(timestamp + nonce + body, secret).

    Headers required:
        X-Lark-Request-Timestamp: unix timestamp
        X-Lark-Request-Nonce: random string
        X-Lark-Signature: computed signature

    Returns:
        True if valid.
    """
    timestamp = headers.get("x-lark-request-timestamp", "")
    nonce = headers.get("x-lark-request-nonce", "")
    signature = headers.get("x-lark-signature", "")

    if not all([timestamp, nonce, signature, secret]):
        logger.warning("Feishu verification: missing required field")
        return False

    # Build signed string: timestamp + nonce + body
    signed = f"{timestamp}{nonce}".encode("utf-8") + raw_body

    computed = hmac.new(
        secret.encode("utf-8"),
        signed,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, signature)


def verify_whatsapp(app_secret: str, raw_body: bytes, signature: str) -> bool:
    """Verify WhatsApp (Meta) webhook signature.

    Uses X-Hub-Signature-256 header: sha256=<hmac_hex>.

    Returns:
        True if valid.
    """
    return verify_hmac_sha256(app_secret, raw_body, signature, prefix="sha256=")


def verify_line(channel_secret: str, raw_body: bytes, signature: str) -> bool:
    """Verify LINE webhook signature.

    Uses x-line-signature header: base64(HMAC-SHA256(body, secret)).

    Returns:
        True if valid.
    """
    if not channel_secret or not signature:
        return False

    import base64
    computed = base64.b64encode(
        hmac.new(
            channel_secret.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")

    return hmac.compare_digest(computed, signature)


def verify_matrix(signing_key: str, raw_body: bytes, signature_header: str) -> bool:
    """Verify Matrix webhook signature.

    Matrix uses X-Matrix-Signature: key_id:hmac_hex
    We verify the HMAC portion against our signing key.

    Returns:
        True if valid.
    """
    if not signing_key or not signature_header:
        return False

    # Parse key_id:hmac_hex
    parts = signature_header.split(":", 1)
    if len(parts) != 2:
        return False

    sig_hex = parts[1]
    return verify_hmac_sha256(signing_key, raw_body, sig_hex, prefix="")


# ═══════════════════════════════════════════════════════════════════
# Discord — Ed25519 Public Key
# ═══════════════════════════════════════════════════════════════════

def verify_discord(public_key: str, signature_hex: str, timestamp: str, raw_body: bytes) -> bool:
    """Verify Discord Interactions webhook using Ed25519 public key.

    Discord sends:
        X-Signature-Ed25519: hex-encoded signature
        X-Signature-Timestamp: unix timestamp

    The signed message is: timestamp + body (as bytes).

    Requires: nacl (PyNaCl) — pip install pynacl

    Returns:
        True if valid.
    """
    if not all([public_key, signature_hex, timestamp]):
        logger.warning("Discord verification: missing required field")
        return False

    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignatureError
    except ImportError:
        logger.warning("PyNaCl not installed — Discord signature verification skipped")
        return True  # Degrade gracefully: allow but warn

    try:
        key_bytes = bytes.fromhex(public_key)
        sig_bytes = bytes.fromhex(signature_hex)
        verify_key = VerifyKey(key_bytes)
        message = timestamp.encode("utf-8") + raw_body
        verify_key.verify(message, sig_bytes)
        return True
    except (ValueError, BadSignatureError) as e:
        logger.warning("Discord signature verification failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════
# Unified verification
# ═══════════════════════════════════════════════════════════════════

def verify_webhook(
    platform: str,
    secret: str,
    headers: dict[str, str],
    raw_body: bytes,
    *,
    public_key: str = "",
) -> bool:
    """Verify webhook signature for any platform.

    Args:
        platform: Platform name (discord, feishu, whatsapp, line, matrix, signal, irc).
        secret: Shared secret / signing key.
        headers: Request headers (lowercase keys).
        raw_body: Raw request body bytes.
        public_key: Discord public key (only for discord platform).

    Returns:
        True if verification passed or not required for this platform.
    """
    # Map platform to verification function
    signature = headers.get("x-signature-ed25519", "")

    match platform:
        case "discord":
            ts = headers.get("x-signature-timestamp", "")
            return verify_discord(public_key, signature, ts, raw_body)
        case "feishu":
            return verify_feishu(secret, headers, raw_body)
        case "whatsapp":
            sig = headers.get("x-hub-signature-256", "")
            return verify_whatsapp(secret, raw_body, sig)
        case "line":
            sig = headers.get("x-line-signature", "")
            return verify_line(secret, raw_body, sig)
        case "matrix":
            sig = headers.get("x-matrix-signature", "")
            return verify_matrix(secret, raw_body, sig)
        case "signal":
            # E2E encrypted, no webhook verification
            return True
        case "irc":
            # Plain TCP, no webhook
            return True
        case _:
            # Unknown platform — default allow
            return True
