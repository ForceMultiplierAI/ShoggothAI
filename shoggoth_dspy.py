# shaggoth_dspy.py

import os
import dspy
from dspy import LM, Example, Predict, History
from shoggoth_utils import conversation_examples

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize DSPy LM for model interactions
lm = LM(
    model='openai/moonshotai/kimi-k2',
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
    temperature=0.7
)
dspy.configure(lm=lm)

# Chatbot signature with history support
class ShaggothChatSignature(dspy.Signature):
    """ShaggothRL chatbot that learns from conversation patterns"""
    current_message: str = dspy.InputField(desc="Current user message")
    history: dspy.History = dspy.InputField(desc="Previous conversation history")
    response: str = dspy.OutputField(desc="Assistant response")

# Replace simple predictor with chatbot
shaggoth_chatbot = Predict(ShaggothChatSignature)

# Convert FastAPI messages to DSPy History format
def fastapi_messages_to_dspy_history(messages):
    """Convert FastAPI message list to DSPy History format"""
    dspy_messages = []
    
    # Process messages in pairs (user -> assistant)
    for i in range(0, len(messages) - 1, 2):
        if i + 1 < len(messages):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            
            if user_msg.role == "user" and assistant_msg.role == "assistant":
                dspy_messages.append({
                    "current_message": user_msg.content,
                    "response": assistant_msg.content
                })
    
    return History(messages=dspy_messages)

# Generate chatbot response with history
def generate_chatbot_response(current_message: str, message_history):
    """Generate response using ShaggothRL chatbot with conversation history"""
    try:
        # Convert FastAPI messages to DSPy format
        dspy_history = fastapi_messages_to_dspy_history(message_history)
        
        # Generate response with history context
        response = shaggoth_chatbot(
            current_message=current_message,
            history=dspy_history
        )
        
        return response.response
    except Exception as e:
        print(f"[ShaggothRL] Chatbot error: {e}")
        return f"Error generating response: {str(e)}"
