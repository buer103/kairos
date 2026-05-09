"""
URL Safety
==========

Validates URLs for safety, preventing:

* Use of dangerous schemes (``file://``, ``data://``, ``javascript://``, …)
* Server-Side Request Forgery (SSRF) — blocks internal/private IP ranges
* Access to localhost and link-local addresses
* Excessively long URLs (> 4096 chars)
* IP addresses embedded in hostnames

Zero external dependencies.  Python 3.12+ (match/case for IP dispatch).
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import ClassVar
from urllib.parse import urlparse


class URLSafety:
    """Validate URLs for safety — scheme, host, and SSRF checks.

    Static class — all methods are classmethods or staticmethods for easy
    use without instantiation.

    Usage::

        safe, reason = URLSafety.check_url("https://example.com/resource")
        if not safe:
            raise SecurityError(reason)
    """

    # Schemes we never allow
    BLOCKED_SCHEMES: ClassVar[frozenset[str]] = frozenset({
        "file",
        "data",
        "ftp",
        "javascript",
        "vbscript",
        "about",
        "chrome",
        "chrome-extension",
        "jar",
        "gopher",
    })

    # Hostnames / IPs that always resolve to local resources
    BLOCKED_HOSTNAMES: ClassVar[frozenset[str]] = frozenset({
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "[::1]",
        "::1",
        "0000::1",
        "0",
    })

    # IPv4 private / reserved ranges (CIDR strings for ipaddress)
    _PRIVATE_IPV4_RANGES: ClassVar[list[str]] = [
        "10.0.0.0/8",          # Class A private
        "172.16.0.0/12",       # Class B private
        "192.168.0.0/16",      # Class C private
        "127.0.0.0/8",         # Loopback
        "169.254.0.0/16",      # Link-local
        "0.0.0.0/8",           # "This" network
        "100.64.0.0/10",       # Carrier-grade NAT
        "198.18.0.0/15",       # Benchmarking
        "224.0.0.0/4",         # Multicast
        "240.0.0.0/4",         # Reserved / future
    ]

    _PRIVATE_IPV6_RANGES: ClassVar[list[str]] = [
        "::1/128",             # Loopback
        "::/128",              # Unspecified
        "fe80::/10",           # Link-local
        "fc00::/7",            # Unique local
        "ff00::/8",            # Multicast
    ]

    MAX_URL_LENGTH: ClassVar[int] = 4096

    # Regex to detect a raw IPv4 or IPv6 address in the hostname position
    _IPV4_HOST_RE: ClassVar[re.Pattern] = re.compile(
        r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    )
    _IPV6_HOST_RE: ClassVar[re.Pattern] = re.compile(
        r"^\[[0-9a-fA-F:]+\]$|^[0-9a-fA-F:]+$"
    )

    @classmethod
    def check_url(cls, url: str) -> tuple[bool, str]:
        """Validate *url* for safety.

        Returns ``(True, "ok")`` when the URL passes all checks, or
        ``(False, reason)`` on the first failure.
        """
        # 1. Length check
        if len(url) > cls.MAX_URL_LENGTH:
            return False, f"URL exceeds {cls.MAX_URL_LENGTH} characters"

        # 2. Parse
        try:
            parsed = urlparse(url)
        except ValueError:
            return False, "URL failed to parse"

        # 3. Scheme check
        scheme = (parsed.scheme or "").lower()
        if not scheme:
            return False, "URL has no scheme"
        if scheme in cls.BLOCKED_SCHEMES:
            return False, f"scheme {scheme!r} is blocked"

        # Data URLs with base64 can be huge / dangerous even if scheme not blocked
        if scheme == "data":
            return False, "data: URLs are blocked"

        # 4. Hostname / netloc check
        hostname = (parsed.hostname or "").lower()

        if not hostname:
            # Allow relative URLs with just a path (for same-origin requests)
            # but not empty-netloc absolute URLs
            if parsed.netloc:
                return False, "URL has an empty hostname"
            return True, "ok"

        if hostname in cls.BLOCKED_HOSTNAMES:
            return False, f"hostname {hostname!r} is blocked"

        # 5. IP-in-hostname detection & SSRF
        ip_safe, ip_reason = cls._check_ip_safety(hostname)
        if not ip_safe:
            return False, ip_reason

        # 6. Check for username:password@ in netloc (credential leakage)
        if parsed.username or parsed.password:
            return False, "URL contains embedded credentials"

        return True, "ok"

    @classmethod
    def _check_ip_safety(cls, hostname: str) -> tuple[bool, str]:
        """Check whether *hostname* resolves to a private / internal IP.

        Handles both literal IP addresses and DNS hostnames.
        """
        # Case 1: hostname is a literal IPv4 address
        if cls._IPV4_HOST_RE.match(hostname):
            return cls._check_ipv4(hostname)

        # Case 2: hostname is a literal IPv6 address (with or without brackets)
        if cls._IPV6_HOST_RE.match(hostname):
            cleaned = hostname.strip("[]")
            return cls._check_ipv6(cleaned)

        # Case 3: DNS hostname — resolve and check all addresses
        return cls._check_hostname_resolution(hostname)

    @classmethod
    def _check_ipv4(cls, addr: str) -> tuple[bool, str]:
        """Check a literal IPv4 address against private ranges."""
        try:
            ip = ipaddress.IPv4Address(addr)
        except ipaddress.AddressValueError:
            return False, f"invalid IPv4 address: {addr!r}"

        for cidr in cls._PRIVATE_IPV4_RANGES:
            if ip in ipaddress.IPv4Network(cidr):
                return False, f"IP {addr} is in private/reserved range {cidr}"
        return True, "ok"

    @classmethod
    def _check_ipv6(cls, addr: str) -> tuple[bool, str]:
        """Check a literal IPv6 address against private ranges."""
        try:
            ip = ipaddress.IPv6Address(addr)
        except ipaddress.AddressValueError:
            return False, f"invalid IPv6 address: {addr!r}"

        for cidr in cls._PRIVATE_IPV6_RANGES:
            if ip in ipaddress.IPv6Network(cidr):
                return False, f"IP {addr} is in private/reserved range {cidr}"
        return True, "ok"

    @classmethod
    def _check_hostname_resolution(cls, hostname: str) -> tuple[bool, str]:
        """Resolve *hostname* via DNS and verify all IPs are public."""
        try:
            addrinfo = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return False, f"DNS resolution failed for {hostname!r}"

        seen: set[str] = set()
        for family, _, _, _, sockaddr in addrinfo:
            ip_str = sockaddr[0]
            if ip_str in seen:
                continue
            seen.add(ip_str)

            match family:
                case socket.AF_INET:
                    safe, reason = cls._check_ipv4(ip_str)
                    if not safe:
                        return False, f"SSRF blocked: {hostname!r} -> {ip_str} ({reason})"
                case socket.AF_INET6:
                    safe, reason = cls._check_ipv6(ip_str)
                    if not safe:
                        return False, f"SSRF blocked: {hostname!r} -> {ip_str} ({reason})"
                case _:
                    return False, f"unknown address family {family} for {hostname!r}"

        return True, "ok"

    @staticmethod
    def is_ip_in_hostname(hostname: str) -> bool:
        """Return *True* when *hostname* is a literal IP address."""
        return bool(
            URLSafety._IPV4_HOST_RE.match(hostname)
            or URLSafety._IPV6_HOST_RE.match(hostname)
        )
