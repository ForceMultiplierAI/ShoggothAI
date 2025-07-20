# run.py

import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

import fastapi
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import aioredis

# Import DSPy components
import shoggoth_dspy
import shoggoth_utils
from shoggoth_utils import conversation_examples
from dspy import Example, Predict

# ----------------------------
# Configuration
# ----------------------------
LOGFILE_JSONL = os.getenv("LOGFILE_JSONL", "LOGFILE.jsonl")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# FastAPI App
app = FastAPI(
    title="DSPy Streaming Proxy",
    description="Forward to upstream LLM APIs, log all to JSONL, and run ShaggothRL",
    version="1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client (shared between lifecycle events)
redis = None

# ----------------------------
# Data Models
# ----------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 4000

class ChatDelta(BaseModel):
    content: str = ""
    role: str = ""

class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: ChatDelta = ChatDelta()
    finish_reason: Optional[str] = None

class ChatCompletionChunkResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatChunkChoice]

class ChatCompletionFinalResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any] = {}

# ----------------------------
# Helper Functions
# ----------------------------
def extract_current_message_and_history(messages: List[Message]):
    """Extract current message and conversation history"""
    if not messages:
        return "", []
    
    current_message = messages[-1].content  # Last message is current input
    history = messages[:-1]  # All previous messages
    
    return current_message, history

def write_jsonl_line(data: Dict):
    """Append JSON-formatted data line to the JSONL log file."""
    try:
        with open(LOGFILE_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write log line: {e}")

def log_comprehensive_data(chat_request, full_response, current_message, message_history, shaggoth_response=None, error=None):
    """Log all inputs, outputs, and processing results comprehensively"""
    log_entry = {
        "type": "comprehensive_log",
        "timestamp": datetime.now().isoformat(),
        "request_data": {
            "model": chat_request.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in chat_request.messages],
            "stream": chat_request.stream,
            "temperature": chat_request.temperature,
            "max_tokens": chat_request.max_tokens
        },
        "extracted_data": {
            "current_message": current_message,
            "message_history": [{"role": msg.role, "content": msg.content} for msg in message_history]
        },
        "upstream_response": full_response,
        "processing_results": {}
    }
    
    # Add ShaggothRL response if available
    if shaggoth_response:
        log_entry["processing_results"]["shaggoth_response"] = shaggoth_response
    
    # Add error information if any
    if error:
        log_entry["processing_results"]["error"] = str(error)
    
    # Add conversation examples count
    try:
        log_entry["processing_results"]["stored_examples_count"] = len(conversation_examples)
    except:
        log_entry["processing_results"]["stored_examples_count"] = 0
    
    write_jsonl_line(log_entry)

# ----------------------------
# Routes
# ----------------------------
@app.post("/chat/completions")
async def proxy_chat_completions(request: Request):
    """
    FastAPI proxy endpoint for chat-based completions.
    Routes upstream based on `?p=http://remote-api.com` query param.
    Logs request/response and persists to Redis for ShaggothRL.
    """
    upstream_url = request.query_params.get("p", "https://openrouter.ai/api/v1")
    if not upstream_url:
        raise HTTPException(status_code=400, detail="Missing proxy 'p' parameter")
    
    print(f"upstream_url ", upstream_url)

    raw_body = await request.json()
    chat_request = ChatCompletionRequest(**raw_body)

    # Extract current message and history early for logging
    current_message, message_history = extract_current_message_and_history(chat_request.messages)

    # Log request to JSONL
    write_jsonl_line({
        "type": "request",
        "timestamp": datetime.now().isoformat(),
        "data": raw_body
    })

    # Forward to remote LLM API
    client = httpx.AsyncClient(timeout=60.0)
    headers = {
        "Authorization": f"Bearer {shoggoth_dspy.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    full_response = ""
    shaggoth_response = None
    processing_error = None

    try:
        proxy_request = client.build_request(
            "POST",
            upstream_url + "/chat/completions",
            headers=headers,
            json=raw_body
        )

        upstream_response = await client.send(proxy_request, stream=True)
        response_id = f"chatcmpl-{uuid.uuid4().hex}"

        async def generate_stream():
            nonlocal full_response
            async for chunk in upstream_response.aiter_bytes():
                decoded = chunk.decode("utf-8", errors="replace")
                full_response += decoded
                yield chunk

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        processing_error = e
        write_jsonl_line({
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "data": str(e)
        })
        
        # Log comprehensive data even on error
        log_comprehensive_data(
            chat_request=chat_request,
            full_response=full_response if full_response else "[Error: No response received]",
            current_message=current_message,
            message_history=message_history,
            error=e
        )
        
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Ensure we have some response content
        if not full_response.strip():
            full_response = "[No response from upstream]"

        # Log basic response
        write_jsonl_line({
            "type": "response",
            "timestamp": datetime.now().isoformat(),
            "data": full_response
        })

        # Store conversation example with full context
        try:
            conversation_example = shoggoth_utils.ConversationExample(
                messages=[{"role": msg.role, "content": msg.content} for msg in chat_request.messages],
                response=full_response
            )
            conversation_examples.append(conversation_example)
        except Exception as e:
            print(f"[ERROR] Failed to store conversation example: {e}")
            processing_error = e

        # Generate ShaggothRL response with conversation history
        try:
            shaggoth_response = shoggoth_dspy.generate_chatbot_response(
                current_message=current_message,
                message_history=message_history
            )
            print(f"[ShaggothRL] Generated: {shaggoth_response[:100]}...")
            
        except Exception as e:
            print(f"[ShaggothRL] Error: {e}")
            processing_error = e

        # Persist conversation examples to Redis
        if redis:
            try:
                await shoggoth_utils.save_redis_conversation_examples(redis)
            except Exception as e:
                print(f"[ERROR] Failed to save conversation examples: {e}")
                processing_error = e

        # Log comprehensive data with all inputs, outputs, and processing results
        log_comprehensive_data(
            chat_request=chat_request,
            full_response=full_response,
            current_message=current_message,
            message_history=message_history,
            shaggoth_response=shaggoth_response,
            error=processing_error
        )

# ----------------------------
# Lifecycle Events
# ----------------------------
@app.on_event("startup")
async def on_startup():
    """Initialize Redis client and load prior conversation examples."""
    global redis
    try:
        redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis.ping()
        await shoggoth_utils.load_redis_conversation_examples(redis)
        print(f"[Startup] Loaded {len(conversation_examples)} conversation examples from Redis.")
    except Exception as e:
        print(f"[Startup] Redis unavailable: {e}")

# ----------------------------
# Launch
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="0.0.0.0", port=7000, reload=False)