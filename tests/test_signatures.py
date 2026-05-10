"""Tests for webhook signature verification across all platforms."""

from __future__ import annotations

import hashlib
import hmac
import json
import base64

import pytest

from kairos.gateway.signatures import (
    verify_hmac_sha256,
    verify_feishu,
    verify_whatsapp,
    verify_line,
    verify_matrix,
    verify_discord,
    verify_webhook,
)


class TestHMACSHA256:
    """Tests for generic HMAC-SHA256 verification."""

    def test_valid_signature(self):
        secret = "test_secret"
        body = b'{"hello": "world"}'
        computed = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        sig = f"sha256={computed}"

        assert verify_hmac_sha256(secret, body, sig) is True

    def test_invalid_signature(self):
        assert verify_hmac_sha256("secret", b"body", "sha256=bad") is False

    def test_empty_secret(self):
        assert verify_hmac_sha256("", b"body", "sha256=abc") is False

    def test_empty_signature(self):
        assert verify_hmac_sha256("secret", b"body", "") is False


class TestFeishu:
    """Tests for Feishu webhook verification."""

    def test_valid_feishu_signature(self):
        secret = "feishu_secret"
        body = b'{"event": "test"}'
        timestamp = "1700000000"
        nonce = "abc123"

        # Compute expected signature
        signed = f"{timestamp}{nonce}".encode() + body
        computed = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()

        headers = {
            "x-lark-request-timestamp": timestamp,
            "x-lark-request-nonce": nonce,
            "x-lark-signature": computed,
        }
        assert verify_feishu(secret, headers, body) is True

    def test_invalid_feishu(self):
        headers = {
            "x-lark-request-timestamp": "1",
            "x-lark-request-nonce": "2",
            "x-lark-signature": "bad",
        }
        assert verify_feishu("secret", headers, b"body") is False

    def test_missing_field(self):
        headers = {}
        assert verify_feishu("secret", headers, b"body") is False


class TestWhatsapp:
    """Tests for WhatsApp (Meta) webhook verification."""

    def test_valid_whatsapp(self):
        secret = "whatsapp_app_secret"
        body = b'{"entry": []}'
        computed = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert verify_whatsapp(secret, body, f"sha256={computed}") is True

    def test_invalid_whatsapp(self):
        assert verify_whatsapp("secret", b"body", "sha256=wrong") is False


class TestLine:
    """Tests for LINE webhook verification."""

    def test_valid_line(self):
        secret = "line_channel_secret"
        body = b'{"events": []}'
        computed = base64.b64encode(
            hmac.new(secret.encode(), body, hashlib.sha256).digest()
        ).decode()
        assert verify_line(secret, body, computed) is True

    def test_invalid_line(self):
        assert verify_line("secret", b"body", "bad_base64") is False


class TestMatrix:
    """Tests for Matrix webhook verification."""

    def test_valid_matrix(self):
        secret = "matrix_key"
        body = b'{"event": "test"}'
        hmac_hex = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        header = f"key_id:{hmac_hex}"
        assert verify_matrix(secret, body, header) is True

    def test_invalid_matrix_no_colon(self):
        assert verify_matrix("k", b"b", "badformat") is False


class TestVerifyWebhook:
    """Tests for unified verify_webhook dispatcher."""

    def test_feishu_with_valid_sig(self):
        secret = "s"
        body = b"hello"
        ts = "1"
        nonce = "n"
        signed = f"{ts}{nonce}".encode() + body
        computed = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        headers = {
            "x-lark-request-timestamp": ts,
            "x-lark-request-nonce": nonce,
            "x-lark-signature": computed,
        }
        assert verify_webhook("feishu", secret, headers, body) is True

    def test_signal_always_true(self):
        assert verify_webhook("signal", "", {}, b"") is True

    def test_irc_always_true(self):
        assert verify_webhook("irc", "", {}, b"") is True

    def test_unknown_platform_true(self):
        assert verify_webhook("unknown_platform", "", {}, b"") is True

    def test_line_with_valid_sig(self):
        secret = "s"
        body = b"hello"
        computed = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
        headers = {"x-line-signature": computed}
        assert verify_webhook("line", secret, headers, body) is True
