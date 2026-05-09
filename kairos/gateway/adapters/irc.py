"""
IRC adapter.

Uses raw TCP socket for IRC protocol (RFC 1459).
No external libraries required.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
import threading
from typing import Any

from kairos.gateway.protocol import (
    UnifiedMessage,
    UnifiedResponse,
    ConnectionState,
    MessageRole,
    ContentBlock,
)
from kairos.gateway.adapters.base import PlatformAdapter


class IRCAdapter(PlatformAdapter):
    """IRВ adapter — raw IRC protocol over TCP socket."""

    platform_name = "irc"

    def __init__(
        self,
        server: str = "irc.libera.chat",
        port: int = 6667,
        nickname: str = "kairos",
        channel: str = "#kairos",
        password: str = "",
        use_ssl: bool = False,
    ):
        super().__init__()
        self._server = server
        self._port = port
        self._nick = nickname
        self._channel = channel
        self._password = password
        self._use_ssl = use_ssl
        self._sock: socket.socket | None = None
        self._reader_thread: threading.Thread | None = None
        self._running = False
        self._msg_queue: list[str] = []
        self._lock = threading.Lock()

    async def connect(self) -> bool:
        """Connect to IRC server and join channel."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(15)

            if self._use_ssl:
                ctx = ssl.create_default_context()
                sock = ctx.wrap_socket(sock, server_hostname=self._server)

            sock.connect((self._server, self._port))
            self._sock = sock

            # IRC handshake
            if self._password:
                self._send_raw(f"PASS {self._password}")
            self._send_raw(f"NICK {self._nick}")
            self._send_raw(f"USER {self._nick} 0 * :Kairos Agent")

            # Wait for welcome
            welcome = self._recv_until("001")
            if "001" not in welcome:
                self._state = ConnectionState.ERROR
                return False

            # Join channel
            self._send_raw(f"JOIN {self._channel}")
            self._state = ConnectionState.CONNECTED

            # Start reader thread
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._read_loop, daemon=True, name="kairos-irc"
            )
            self._reader_thread.start()

            return True
        except Exception:
            self._state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> None:
        self._running = False
        try:
            if self._sock:
                self._send_raw("QUIT :Kairos signing off")
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._state = ConnectionState.DISCONNECTED

    async def send(self, chat_id: str, response: UnifiedResponse) -> bool:
        """Send PRIVMSG to target (user or channel)."""
        if not self._sock:
            return False

        target = chat_id or self._channel
        lines = response.text.split("\n")

        try:
            for line in lines:
                if not line.strip():
                    continue
                # IRC max message length is ~510 bytes
                for chunk in _chunk_irc(line, 450):
                    self._send_raw(f"PRIVMSG {target} :{chunk}")
            return True
        except Exception:
            return False

    async def receive(self) -> UnifiedMessage | None:
        """Non-blocking receive from message queue."""
        with self._lock:
            if self._msg_queue:
                raw = self._msg_queue.pop(0)
                return self.translate_incoming({"raw": raw})
        return None

    def translate_incoming(self, raw: dict[str, Any]) -> UnifiedMessage:
        """Parse an IRC PRIVMSG line."""
        line = raw.get("raw", "")

        # Parse: :nick!user@host PRIVMSG #channel :message
        sender = ""
        sender_name = ""
        channel = ""
        content = ""

        if line.startswith(":"):
            # Extract sender
            parts = line[1:].split(" ", 1)
            sender_full = parts[0]
            sender = sender_full.split("!")[0] if "!" in sender_full else sender_full
            sender_name = sender
            if len(parts) > 1:
                rest = parts[1]
                cmd_parts = rest.split(" :", 1)
                cmd = cmd_parts[0].split()
                if len(cmd) >= 2 and cmd[0] == "PRIVMSG":
                    channel = cmd[1]
                if len(cmd_parts) > 1:
                    content = cmd_parts[1]

        # Handle PING
        if line.startswith("PING"):
            pong_target = line[5:].strip()
            try:
                self._send_raw(f"PONG {pong_target}")
            except Exception:
                pass
            content = ""

        return UnifiedMessage(
            id=str(len(self._msg_queue)),
            role=MessageRole.USER if content else MessageRole.SYSTEM,
            content=[ContentBlock.text_block(content)],
            platform="irc",
            chat_id=channel,
            sender_id=sender,
            sender_name=sender_name,
        )

    # ── Internal ────────────────────────────────────────────────

    def _send_raw(self, msg: str) -> None:
        """Send raw IRC message."""
        if self._sock:
            self._sock.sendall((msg + "\r\n").encode())

    def _recv_until(self, code: str) -> str:
        """Receive lines until a specific IRC numeric code appears."""
        buf = ""
        if not self._sock:
            return buf
        for _ in range(100):  # safety limit
            try:
                data = self._sock.recv(4096).decode(errors="replace")
                buf += data
                if f" {code} " in buf:
                    break
            except socket.timeout:
                break
        return buf

    def _read_loop(self) -> None:
        """Background thread: read IRC messages into queue."""
        while self._running and self._sock:
            try:
                data = self._sock.recv(4096)
                if not data:
                    break
                lines = data.decode(errors="replace").split("\r\n")
                with self._lock:
                    for line in lines:
                        if line.strip() and "PRIVMSG" in line:
                            self._msg_queue.append(line.strip())
            except (socket.timeout, OSError):
                continue
            except Exception:
                break


def _chunk_irc(text: str, max_len: int = 450) -> list[str]:
    """Split text into IRC-safe chunks."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while len(text) > max_len:
        split = text.rfind(" ", 0, max_len)
        if split == -1:
            split = max_len
        chunks.append(text[:split].strip())
        text = text[split:].strip()
    if text:
        chunks.append(text)
    return chunks
