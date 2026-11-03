import os
import json
import uuid
import asyncio
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
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
    {"name": "list_directory", "description": "List files under $HOME...", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "read_file", "description": "Read a file's text...", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "write_file", "description": "Write/create a file under $HOME...", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "edit_file", "description": "Edit a file with search/replace...", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}}},
]

VALID_TOOLS = {t["name"] for t in TOOLS}

# ---- Agent connection state ----
connected_agents: Dict[str, WebSocket] = {}
pending_calls: Dict[str, asyncio.Future] = {}


@app.get("/")
async def root():
    return {"service": "gmod-mcp-bridge", "version": "2.0.0", "online": len(connected_agents) > 0}


@app.get("/health")
async def health():
    return {"ok": True, "online": len(connected_agents) > 0, "agents": len(connected_agents)}


# ---- WebSocket: GMod agent connects here ----
@app.websocket("/ws/agent")
async def agent_ws(websocket: WebSocket):
    await websocket.accept()
    agent_id = str(uuid.uuid4())
    connected_agents[agent_id] = websocket
    print(f"[agent] connected: {agent_id}")
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_id = msg.get("id")
            if msg_id and msg_id in pending_calls:
                future = pending_calls.pop(msg_id)
                if not future.done():
                    if msg.get("ok"):
                        future.set_result(msg.get("result"))
                    else:
                        future.set_exception(Exception(msg.get("error", "agent tool failed")))
    except WebSocketDisconnect:
        pass
    finally:
        connected_agents.pop(agent_id, None)
        print(f"[agent] disconnected: {agent_id}")


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
                "serverInfo": {"name": "gmod-mcp-bridge", "version": "2.0.0"},
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

        if not connected_agents:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "result": {
                    "content": [{"type": "text", "text": "Error: No GMod agent connected. Start the server and run gmbridge_connect."}],
                    "isError": True,
                }},
            )

        agent_id, ws = next(iter(connected_agents.items()))
        future = asyncio.get_running_loop().create_future()
        pending_calls[req_id] = future

        payload = {"id": req_id, "tool": tool_name, "args": args}
        try:
            await ws.send_text(json.dumps(payload))
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
                    "content": [{"type": "text", "text": "Error: tool call timed out waiting for GMod agent"}],
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
