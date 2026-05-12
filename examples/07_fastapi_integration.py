"""
07 — FastAPI Integration
========================

Embed a Kairos agent inside your existing FastAPI web application.
This is the pattern for adding AI capabilities to your backend services.

The key insight: Agent.run() is a synchronous function that takes a string
and returns a dict. You can call it from any web framework.

Run:
    pip install fastapi uvicorn
    python examples/07_fastapi_integration.py
    # Then open http://localhost:8000/docs for the Swagger UI
"""

import os
from contextlib import asynccontextmanager
from kairos import Agent
from kairos.providers.base import ModelConfig

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-your-key-here"

# ── 1. Create and configure your agent (once at startup) ────────
agent = Agent(
    model=ModelConfig(api_key=API_KEY),
    agent_name="SupportBot",
    role_description=(
        "You are a technical support assistant. "
        "Answer questions clearly and offer step-by-step solutions."
    ),
)

# ── 2. FastAPI app ──────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    content: str
    confidence: float | None = None


app = FastAPI(
    title="Kairos Support API",
    description="AI-powered technical support chatbot",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "agent": agent.agent_name}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message to the agent and get a response."""
    try:
        result = agent.run(req.message)
        return ChatResponse(
            content=result["content"],
            confidence=result.get("confidence"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/stream")
async def chat_stream(message: str):
    """Stream the agent's response token-by-token (SSE)."""
    from kairos import StatefulAgent

    # For streaming, use StatefulAgent
    streaming_agent = StatefulAgent(model=ModelConfig(api_key=API_KEY))

    async def generate():
        stream = streaming_agent.chat_stream(message)
        for event in stream:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── 3. Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("Kairos FastAPI server starting at http://localhost:8000")
    print("Swagger UI: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
