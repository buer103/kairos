"""MCP (Model Context Protocol) client tools.

Tools:
  - mcp_connect: start an MCP server process and discover its tools
  - mcp_call_tool: call a tool on a connected MCP server via JSON-RPC over stdio
  - mcp_list_servers: list all active MCP connections
  - mcp_disconnect: terminate an MCP server connection

Uses subprocess with stdin/stdout pipes and JSON-RPC 2.0 protocol.
All connections are tracked in a module-level registry.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import uuid
from typing import Any

from kairos.tools.registry import register_tool

logger = logging.getLogger("kairos.tools.mcp")

# ── MCP connection registry ─────────────────────────────────────────────────

# Each connection: {process, server_id, command, args, tools, created_at, request_id}
_active_connections: dict[str, dict[str, Any]] = {}
_connections_lock = threading.Lock()


def _generate_server_id() -> str:
    """Generate a short unique server ID."""
    return f"mcp-{uuid.uuid4().hex[:8]}"


# ── JSON-RPC 2.0 helpers ────────────────────────────────────────────────────

def _make_request(method: str, params: dict | None = None, req_id: int | str = 1) -> dict:
    """Build a JSON-RPC 2.0 request."""
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id,
    }
    if params is not None:
        msg["params"] = params
    return msg


def _send_and_receive(
    process: subprocess.Popen,
    request: dict,
    timeout: float = 10.0,
) -> dict:
    """Send a JSON-RPC request to an MCP server and wait for the response.

    Uses stdin/stdout pipes. The response must have a matching 'id'.
    """
    req_id = request.get("id")
    request_bytes = (json.dumps(request) + "\n").encode("utf-8")

    try:
        # Write request to stdin
        process.stdin.write(request_bytes)  # type: ignore[union-attr]
        process.stdin.flush()  # type: ignore[union-attr]

        # Read response line from stdout
        deadline = time.time() + timeout
        response_line = ""

        while time.time() < deadline:
            # Non-blocking-ish read: check if data available
            import select
            if select.select([process.stdout], [], [], 0.1)[0]:  # type: ignore[union-attr]
                line = process.stdout.readline()  # type: ignore[union-attr]
                if not line:
                    break
                response_line = line.decode("utf-8").strip()
                # Try to parse it
                if response_line:
                    try:
                        resp = json.loads(response_line)
                        # Check if this is our response
                        if resp.get("id") == req_id:
                            return resp
                        # Could be a notification — continue reading
                        logger.debug("MCP: received non-matching message id=%s (expected %s)", resp.get("id"), req_id)
                    except json.JSONDecodeError:
                        logger.debug("MCP: unparseable line: %s", response_line[:200])
                        continue

        return {"jsonrpc": "2.0", "error": {"code": -32000, "message": f"Timeout waiting for response after {timeout}s"}, "id": req_id}

    except BrokenPipeError:
        return {"jsonrpc": "2.0", "error": {"code": -32000, "message": "Broken pipe — server may have crashed"}, "id": req_id}
    except Exception as e:
        return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": req_id}


def _rpc_result(response: dict) -> tuple[Any, str | None]:
    """Extract result or error from a JSON-RPC response."""
    if "error" in response:
        err = response["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        return None, msg
    return response.get("result"), None


# ── Tools ────────────────────────────────────────────────────────────────────

@register_tool(
    name="mcp_connect",
    description="Connect to an MCP server by launching it as a subprocess and initializing the JSON-RPC session. Discovers the server's available tools.",
    parameters={
        "server_command": {"type": "string", "description": "Command to launch the MCP server (e.g., 'npx', 'python', '/path/to/server')"},
        "server_args": {"type": "string", "description": "Optional JSON array of arguments for the server command (e.g., '[\"-y\", \"@modelcontextprotocol/server-filesystem\", \"/tmp\"]')"},
    },
    category="mcp",
)
def mcp_connect(server_command: str, server_args: str = "[]") -> dict:
    """Launch an MCP server process and initialize the connection.

    The MCP protocol uses JSON-RPC 2.0 over stdio. After launching the process,
    we send an 'initialize' request, then a 'notifications/initialized' notification,
    and finally a 'tools/list' request to discover available tools.

    Returns:
        dict with keys: server_id, tools (list of {name, description, schema}), server_command
    """
    try:
        args: list[str] = json.loads(server_args) if server_args else []
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON for server_args: {server_args}"}

    server_id = _generate_server_id()

    try:
        full_cmd = [server_command] + args
        process = subprocess.Popen(
            full_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # We'll handle bytes manually
        )

        # Step 1: Initialize
        init_req = _make_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "kairos-mcp-client", "version": "1.0.0"},
        }, req_id=1)
        init_resp = _send_and_receive(process, init_req, timeout=15.0)
        init_result, init_error = _rpc_result(init_resp)
        if init_error:
            process.kill()
            process.wait()
            return {"error": f"MCP initialize failed: {init_error}", "server_command": server_command}

        # Step 2: Send initialized notification
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        try:
            process.stdin.write((json.dumps(notification) + "\n").encode("utf-8"))  # type: ignore[union-attr]
            process.stdin.flush()  # type: ignore[union-attr]
        except Exception:
            pass

        # Step 3: Discover tools
        tools_list_req = _make_request("tools/list", req_id=2)
        tools_resp = _send_and_receive(process, tools_list_req, timeout=15.0)
        tools_result, tools_error = _rpc_result(tools_resp)

        tools: list[dict] = []
        if tools_result and isinstance(tools_result, dict):
            raw_tools = tools_result.get("tools", [])
            for t in raw_tools:
                tools.append({
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "schema": t.get("inputSchema", {}),
                })

        # Store connection
        conn = {
            "process": process,
            "server_id": server_id,
            "command": server_command,
            "args": args,
            "tools": tools,
            "created_at": time.time(),
            "request_id": 3,  # next request ID to use
        }

        with _connections_lock:
            _active_connections[server_id] = conn

        return {
            "server_id": server_id,
            "server_command": server_command,
            "server_args": args,
            "tools": tools,
            "tool_count": len(tools),
            "protocol_version": init_result.get("protocolVersion", "unknown") if isinstance(init_result, dict) else "unknown",
            "server_info": init_result.get("serverInfo", {}) if isinstance(init_result, dict) else {},
            "connection_error": tools_error if tools_error else None,
        }

    except FileNotFoundError:
        return {"error": f"Command not found: {server_command}", "server_command": server_command}
    except Exception as e:
        # Clean up if we started a process
        try:
            process.kill()
            process.wait()
        except Exception:
            pass
        return {"error": str(e), "server_command": server_command}


@register_tool(
    name="mcp_call_tool",
    description="Call a tool on a connected MCP server. Pass the tool name and arguments as a JSON object string.",
    parameters={
        "server_id": {"type": "string", "description": "The server ID returned by mcp_connect"},
        "tool_name": {"type": "string", "description": "Name of the tool to call on the MCP server"},
        "arguments": {"type": "string", "description": "Tool arguments as a JSON object string (e.g., '{\"path\": \"/tmp\"}')"},
    },
    category="mcp",
)
def mcp_call_tool(server_id: str, tool_name: str, arguments: str = "{}") -> dict:
    """Call a tool on a connected MCP server via JSON-RPC.

    Returns:
        dict with keys: server_id, tool_name, result (or error)
    """
    with _connections_lock:
        conn = _active_connections.get(server_id)
        if not conn:
            return {"error": f"No connection found for server_id: {server_id}", "server_id": server_id}

    process: subprocess.Popen = conn["process"]

    # Check if process is still alive
    if process.poll() is not None:
        with _connections_lock:
            _active_connections.pop(server_id, None)
        return {
            "error": f"MCP server process has exited with code {process.returncode}",
            "server_id": server_id,
        }

    # Parse arguments
    try:
        args_dict: dict[str, Any] = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON for arguments: {arguments}", "server_id": server_id, "tool_name": tool_name}

    # Get next request ID
    req_id = conn["request_id"]
    conn["request_id"] += 1

    # Send tools/call request
    request = _make_request("tools/call", {
        "name": tool_name,
        "arguments": args_dict,
    }, req_id=req_id)

    response = _send_and_receive(process, request, timeout=60.0)
    result, error = _rpc_result(response)

    if error:
        return {
            "server_id": server_id,
            "tool_name": tool_name,
            "error": error,
            "jsonrpc_error": response.get("error"),
        }

    # result is expected to be {content: [...], isError: bool}
    if isinstance(result, dict):
        content_items = result.get("content", [])
        # Extract text content
        text_parts = []
        for item in content_items if isinstance(content_items, list) else [content_items]:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "resource":
                    text_parts.append(json.dumps(item.get("resource", {})))
                else:
                    text_parts.append(json.dumps(item))

        return {
            "server_id": server_id,
            "tool_name": tool_name,
            "result": result,
            "content": "\n".join(text_parts) if text_parts else json.dumps(result),
            "is_error": result.get("isError", False),
        }

    return {
        "server_id": server_id,
        "tool_name": tool_name,
        "result": result,
    }


@register_tool(
    name="mcp_list_servers",
    description="List all active MCP server connections with their status and available tools.",
    parameters={},
    category="mcp",
)
def mcp_list_servers() -> dict:
    """List all active MCP connections.

    Returns:
        dict with keys: servers (list of connection summaries), count
    """
    servers = []
    with _connections_lock:
        for sid, conn in _active_connections.items():
            proc: subprocess.Popen = conn["process"]
            alive = proc.poll() is None
            servers.append({
                "server_id": sid,
                "command": conn["command"],
                "args": conn["args"],
                "alive": alive,
                "exit_code": proc.returncode if not alive else None,
                "tool_count": len(conn.get("tools", [])),
                "tools": [t["name"] for t in conn.get("tools", [])],
                "uptime_seconds": round(time.time() - conn["created_at"], 1),
                "next_request_id": conn.get("request_id", 0),
            })

    return {
        "servers": servers,
        "count": len(servers),
    }


@register_tool(
    name="mcp_disconnect",
    description="Disconnect and terminate an MCP server process. Sends SIGTERM, then SIGKILL if needed.",
    parameters={
        "server_id": {"type": "string", "description": "The server ID to disconnect. Use 'all' to disconnect all servers."},
    },
    category="mcp",
)
def mcp_disconnect(server_id: str = "all") -> dict:
    """Disconnect one or all MCP server processes.

    Returns:
        dict with keys: disconnected (list of server IDs), errors
    """
    if server_id == "all":
        with _connections_lock:
            ids = list(_active_connections.keys())
    else:
        ids = [server_id]

    disconnected = []
    errors = []

    for sid in ids:
        with _connections_lock:
            conn = _active_connections.pop(sid, None)

        if not conn:
            errors.append(f"No connection found: {sid}")
            continue

        process: subprocess.Popen = conn["process"]
        try:
            # Try graceful termination first
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                # Force kill
                process.kill()
                process.wait(timeout=2.0)
            disconnected.append(sid)
        except Exception as e:
            errors.append(f"Failed to disconnect {sid}: {e}")

    return {
        "disconnected": disconnected,
        "errors": errors if errors else None,
        "remaining_connections": len(_active_connections),
    }
