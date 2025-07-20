# shaggoth_utils.py

import os
import json
from collections import deque
from typing import List, Dict, Any
import aioredis

# Configuration from environment variables
EXAMPLE_STORAGE_KEY = os.getenv("EXAMPLE_STORAGE_KEY", "examples:shaggothrl")

# Conversation example storage class
class ConversationExample:
    def __init__(self, messages, response):
        self.messages = messages  # Full message history
        self.response = response  # Generated response
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "messages": self.messages,
            "response": self.response
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary for deserialization"""
        return cls(data["messages"], data["response"])

# Example storage (in-memory) - updated format
conversation_examples = deque(maxlen=1000)

# Legacy storage for backward compatibility
in_memory_examples = deque(maxlen=1000)

# Redis example persistence functions

async def load_redis_examples(redis_client):
    """Load legacy examples from Redis into memory (backward compatibility)."""
    global in_memory_examples
    try:
        raw = await redis_client.get(EXAMPLE_STORAGE_KEY)
        if raw:
            from dspy import Example
            examples = [Example(**item) for item in json.loads(raw)]
            in_memory_examples = deque(examples, maxlen=1000)
    except Exception as e:
        print(f"Error loading legacy examples: {e}")

async def save_redis_examples(redis_client):
    """Persist legacy examples to Redis (backward compatibility)."""
    try:
        await redis_client.set(
            EXAMPLE_STORAGE_KEY,
            json.dumps([dict(ex) for ex in in_memory_examples])
        )
    except Exception as e:
        print(f"Error saving legacy examples: {e}")

async def load_redis_conversation_examples(redis_client):
    """Load conversation examples from Redis"""
    global conversation_examples
    try:
        raw = await redis_client.get(EXAMPLE_STORAGE_KEY + ":conversations")
        if raw:
            examples_data = json.loads(raw)
            conversation_examples = deque(
                [ConversationExample.from_dict(ex) for ex in examples_data],
                maxlen=1000
            )
            print(f"[Redis] Loaded {len(conversation_examples)} conversation examples")
        else:
            print("[Redis] No conversation examples found, starting fresh")
    except Exception as e:
        print(f"Error loading conversation examples: {e}")

async def save_redis_conversation_examples(redis_client):
    """Save conversation examples to Redis"""
    try:
        examples_data = [ex.to_dict() for ex in conversation_examples]
        await redis_client.set(
            EXAMPLE_STORAGE_KEY + ":conversations",
            json.dumps(examples_data)
        )
        print(f"[Redis] Saved {len(conversation_examples)} conversation examples")
    except Exception as e:
        print(f"Error saving conversation examples: {e}")

# Utility functions for analysis and debugging

def get_conversation_stats():
    """Get statistics about stored conversations"""
    if not conversation_examples:
        return {"count": 0, "average_length": 0}
    
    total_messages = sum(len(ex.messages) for ex in conversation_examples)
    average_length = total_messages / len(conversation_examples)
    
    return {
        "count": len(conversation_examples),
        "average_length": round(average_length, 2),
        "total_messages": total_messages
    }

def get_recent_conversations(limit=5):
    """Get the most recent conversations for debugging"""
    recent = list(conversation_examples)[-limit:]
    return [
        {
            "message_count": len(ex.messages),
            "last_message": ex.messages[-1] if ex.messages else None,
            "response_preview": ex.response[:100] + "..." if len(ex.response) > 100 else ex.response
        }
        for ex in recent
    ]

def extract_current_message_and_history_from_messages(messages: List[Dict[str, str]]):
    """Extract current message and conversation history from message list"""
    if not messages:
        return "", []
    
    current_message = None
    history = []
    
    # Find the last user message as current input
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            current_message = messages[i]["content"]
            history = messages[:i]
            break
    
    if current_message is None:
        current_message = ""
        history = messages
    
    return current_message, history

def add_conversation_example(messages: List[Dict[str, str]], response: str):
    """Add a new conversation example to storage"""
    try:
        conversation_example = ConversationExample(messages, response)
        conversation_examples.append(conversation_example)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to store conversation example: {e}")
        return False