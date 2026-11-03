import os
import json
import time
import asyncio
from collections import deque
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# ---- MCP Tool Definitions ----
TOOLS = [
    {"name": "get_game_status", "description": "Report whether the Garry's Mod server is connected...", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "run_gmod_command", "description": "Execute a Garry's Mod server console command...", "inputSchema": {"type": "object", "properties": {"command": {"type": "string"}}}},
    {"name": "execute_lua_code", "description": "Execute Lua on the Garry's Mod server...", "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}}},
    {"name": "validate_lua_syntax", "description": "Compile-check Lua without executing...", "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}}},
    {"name": "check_game_file", "description": "Check if a path exists in the game VFS...", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "capture_console_output", "description": "Return recent console output...", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "read_gmod_wiki", "description": "Fetch a page from the GMod wiki...", "inputSchema": {"type": "object", "properties": {"page": {"type": "string"}}}},
    {"name": "list_tool_packages", "description": "List tool packages...", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "request_tool_packages", "description": "Explain how to enable tool packages...", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "call_package_tool", "description": "Run one tool from an enabled package...", "inputSchema": {"type": "object", "properties": {"package": {"type": "string"}, "name": {"type": "string"}}}},
    {"name": "list_directory", "description": "List files and folders under garrysmod/data (the DATA sandbox, the only writable location in Lua mode). Paths are relative to garrysmod/data; omit or use '.' for the root. Use maxDepth > 1 for a recursive tree.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "read_file", "description": "Read a file's text. Tries garrysmod/data (DATA) first, then the game's mounted files (GAME), then the Lua search path (LUA). GAME and LUA are read-only.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "write_file", "description": "Write or create a file under garrysmod/data (DATA), the only writable location in Lua mode. Creates parent directories and overwrites if the file exists.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "edit_file", "description": "Edit a file in garrysmod/data (DATA) with exact search/replace. Replaces the first match unless replace_all is true. Fails if old_string is not found.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}}},
]

VALID_TOOLS = {t["name"] for t in TOOLS}

# ---- Agent state. The GMod server polls us; we never connect to it. ----
command_queue: deque = deque()
pending_calls: Dict[Any, asyncio.Future] = {}
last_poll_ts: float = 0.0
poll_count: int = 0
POLL_TIMEOUT = 5.0  # seconds without a poll before we call the agent offline


def agent_online() -> bool:
    return (time.time() - last_poll_ts) < POLL_TIMEOUT


@app.get("/")
async def root():
    return {"service": "gmod-mcp-bridge", "version": "2.1.0", "online": agent_online()}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "online": agent_online(),
        "agents": 1 if agent_online() else 0,
        "polls": poll_count,
        "queued": len(command_queue),
        "last_poll": last_poll_ts,
    }


# ---- GMod polls this for work ----
@app.get("/agent/poll")
async def agent_poll():
    global last_poll_ts, poll_count
    last_poll_ts = time.time()
    poll_count += 1
    if command_queue:
        cmd = command_queue.popleft()
        return {"event": cmd}
    return {"event": None}


# ---- GMod posts results here ----
@app.post("/agent/result")
async def agent_result(request: Request):
    from urllib.parse import parse_qs

    raw_body = await request.body()
    body = raw_body.decode("utf-8", "replace")
    ctype = request.headers.get("content-type", "")
    data: Any = None

    if "application/json" in ctype:
        try:
            data = json.loads(body)
        except Exception:
            data = None
    else:
        params = parse_qs(body)
        raw = (params.get("payload") or [None])[0]
        if raw:
            try:
                data = json.loads(raw)
            except Exception:
                data = None

    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "bad payload"}, status_code=400)

    rid = data.get("id")
    fut = pending_calls.pop(rid, None)
    if fut is not None and not fut.done():
        if data.get("ok"):
            fut.set_result(data.get("result"))
        else:
            fut.set_exception(Exception(data.get("error") or data.get("result") or "agent tool failed"))
    return {"ok": True}


# ---- MCP: ChatGPT connects here ----
@app.post("/mcp")
async def mcp(request_body: dict):
    method = request_body.get("method")
    req_id = request_body.get("id")
    params = request_body.get("params", {}) or {}

    if method in ("notifications/initialized", "notifications/cancelled"):
        return Response(status_code=202)

    if method == "initialize":
        protocol = params.get("protocolVersion") or "2024-11-05"
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": protocol,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "gmod-mcp-bridge", "version": "2.1.0"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if not tool_name or tool_name not in VALID_TOOLS:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}},
                status_code=400,
            )

        if not agent_online():
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "result": {
                    "content": [{"type": "text", "text": "Error: GMod agent is not polling. Check that the server is running and gmbridge_status shows polls > 0."}],
                    "isError": True,
                }},
            )

        future = asyncio.get_running_loop().create_future()
        pending_calls[req_id] = future
        command_queue.append({"id": req_id, "tool": tool_name, "args": args})

        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2) if not isinstance(result, str) else result}]},
            }
        except asyncio.TimeoutError:
            pending_calls.pop(req_id, None)
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "result": {
                    "content": [{"type": "text", "text": "Error: tool call timed out; the GMod agent picked it up but never posted a result. Check the server console for a Lua error."}],
                    "isError": True,
                }},
            )
        except Exception as e:
            pending_calls.pop(req_id, None)
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "result": {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                }},
            )

    return JSONResponse(
        {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}},
        status_code=400,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT") or os.environ.get("SERVER_PORT") or 3000)
    uvicorn.run("main:app", host="0.0.0.0", port=port)
