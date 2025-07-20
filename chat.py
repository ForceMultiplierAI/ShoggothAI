#!/usr/bin/env python3
"""
Simple streaming chat client for DSPy proxy
Connects to: http://localhost:7000/chat/completions?p=http://10.0.0.2:8000/v1
"""

import json
import requests
import sys
from typing import List, Dict

class StreamingChatClient:
    def __init__(self, proxy_url: str = "http://localhost:7000/chat/completions", 
                 upstream_url: str = "http://10.0.0.2:8000/v1",
                 model: str = "hello"):
        self.proxy_url = proxy_url
        self.upstream_url = upstream_url
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def send_message(self, user_message: str) -> str:
        """Send a message and get streaming response"""
        # Add user message to history
        self.add_message("user", user_message)
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Make request with upstream parameter
        url = f"{self.proxy_url}?p={self.upstream_url}"
        
        try:
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            # Process streaming response
            full_response = ""
            print("Assistant: ", end="", flush=True)
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_part = line_text[6:]  # Remove 'data: ' prefix
                        if data_part.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data_part)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end="", flush=True)
                                    full_response += content
                        except json.JSONDecodeError:
                            # Handle non-JSON lines (like connection info)
                            continue
            
            print()  # New line after response
            
            # Add assistant response to history
            if full_response:
                self.add_message("assistant", full_response)
            
            return full_response
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to server: {e}"
            print(f"Error: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"Error: {error_msg}")
            return error_msg
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def show_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            print("No conversation history.")
            return
        
        print("\n--- Conversation History ---")
        for i, msg in enumerate(self.conversation_history, 1):
            role = msg['role'].title()
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"{i}. {role}: {content}")
        print("--- End History ---\n")

def main():
    # Initialize chat client
    chat = StreamingChatClient()
    
    print("🤖 Streaming Chat Client")
    print(f"Connected to: {chat.proxy_url}")
    print(f"Upstream: {chat.upstream_url}")
    print(f"Model: {chat.model}")
    print("\nCommands:")
    print("  /quit or /exit - Exit the chat")
    print("  /clear - Clear conversation history")
    print("  /history - Show conversation history")
    print("  /help - Show this help message")
    print("\nStart chatting!\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['/quit', '/exit']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == '/clear':
                chat.clear_history()
                continue
            elif user_input.lower() == '/history':
                chat.show_history()
                continue
            elif user_input.lower() == '/help':
                print("\nCommands:")
                print("  /quit or /exit - Exit the chat")
                print("  /clear - Clear conversation history")
                print("  /history - Show conversation history")
                print("  /help - Show this help message\n")
                continue
            
            # Send message and get response
            response = chat.send_message(user_input)
            
            # Add some spacing for readability
            print()
            
        except KeyboardInterrupt:
            print("\n\nChat interrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Continuing...\n")

if __name__ == "__main__":
    main()
