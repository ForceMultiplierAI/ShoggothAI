#!/usr/bin/env python3
"""
Load conversations from dataset file and send to DSPy proxy
Usage: python load_conversation.py <dataset_file> [--count 200] [--delay 1.0]
"""

import argparse
import json
import requests
import time
import random
import sys
from pathlib import Path
from typing import List, Dict

class ConversationLoader:
    def __init__(self, proxy_url="http://localhost:7000/chat/completions", 
                 upstream_url="http://10.0.0.2:8000/v1", model="hello"):
        self.proxy_url = proxy_url
        self.upstream_url = upstream_url
        self.model = model
        self.sent_count = 0
        self.success_count = 0
        self.error_count = 0
        
    def send_conversation(self, messages: List[Dict[str, str]], conversation_id: int):
        """Send a single conversation to the API"""
        
        # Ensure messages have the right format
        formatted_messages = []
        for msg in messages:
            # Normalize role names
            role = msg.get('role', 'user').lower()
            if role in ['prompter', 'human', 'person1']:
                role = 'user'
            elif role in ['assistant', 'ai', 'person2']:
                role = 'assistant'
            
            content = msg.get('content', '') or msg.get('text', '') or msg.get('message', '')
            if content.strip():
                formatted_messages.append({
                    "role": role,
                    "content": content.strip()
                })
        
        if not formatted_messages:
            print(f"⚠️  Skipping empty conversation {conversation_id}")
            return False
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        url = f"{self.proxy_url}?p={self.upstream_url}"
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            
            # Consume the streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_part = line_text[6:]
                        if data_part.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data_part)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                        except json.JSONDecodeError:
                            continue
            
            self.sent_count += 1
            self.success_count += 1
            
            # Show progress
            msg_count = len(formatted_messages)
            first_msg = formatted_messages[0]['content'][:50] + "..." if len(formatted_messages[0]['content']) > 50 else formatted_messages[0]['content']
            print(f"✓ {self.sent_count}: Sent conversation {conversation_id} ({msg_count} msgs) - '{first_msg}'")
            
            return True
            
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            print(f"✗ Error sending conversation {conversation_id}: {e}")
            return False
        except Exception as e:
            self.error_count += 1
            print(f"✗ Unexpected error with conversation {conversation_id}: {e}")
            return False
    
    def load_and_send_conversations(self, dataset_file: str, count: int = 200, delay: float = 1.0, 
                                   shuffle: bool = True, start_from: int = 0):
        """Load conversations from file and send them to the API"""
        
        # Load conversations from file
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        except Exception as e:
            print(f"Error loading dataset file: {e}")
            return False
        
        print(f"📁 Loaded {len(conversations)} conversations from {dataset_file}")
        
        # Filter out conversations that are too short or too long
        filtered_conversations = []
        for conv in conversations:
            if isinstance(conv, list) and 1 <= len(conv) <= 20:  # Reasonable conversation length
                filtered_conversations.append(conv)
        
        print(f"🔍 Filtered to {len(filtered_conversations)} valid conversations")
        
        if not filtered_conversations:
            print("❌ No valid conversations found in dataset!")
            return False
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(filtered_conversations)
            print("🔀 Shuffled conversation order")
        
        # Select subset
        selected_conversations = filtered_conversations[start_from:start_from + count]
        print(f"📤 Sending {len(selected_conversations)} conversations (starting from {start_from})")
        
        print(f"⏱️  Delay between requests: {delay}s")
        print(f"🎯 Target: {self.proxy_url}")
        print(f"🔄 Upstream: {self.upstream_url}")
        print(f"🤖 Model: {self.model}")
        print("-" * 60)
        
        # Send conversations
        start_time = time.time()
        
        for i, conversation in enumerate(selected_conversations):
            conversation_id = start_from + i + 1
            
            success = self.send_conversation(conversation, conversation_id)
            
            # Rate limiting
            if delay > 0 and i < len(selected_conversations) - 1:  # Don't delay after last request
                time.sleep(delay)
        
        # Summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("-" * 60)
        print(f"📊 Summary:")
        print(f"   Total sent: {self.sent_count}")
        print(f"   Successful: {self.success_count}")
        print(f"   Errors: {self.error_count}")
        print(f"   Success rate: {(self.success_count/self.sent_count*100):.1f}%" if self.sent_count > 0 else "N/A")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Average per request: {(duration/self.sent_count):.2f}s" if self.sent_count > 0 else "N/A")
        
        return self.success_count > 0

def main():
    parser = argparse.ArgumentParser(description="Load conversations and send to DSPy proxy")
    parser.add_argument("dataset_file", help="Path to dataset JSON file")
    parser.add_argument("--count", type=int, default=200, help="Number of conversations to send")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--proxy", default="http://localhost:7000/chat/completions", help="Proxy URL")
    parser.add_argument("--upstream", default="http://10.0.0.2:8000/v1", help="Upstream URL")
    parser.add_argument("--model", default="hello", help="Model name")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle conversations")
    parser.add_argument("--start-from", type=int, default=0, help="Start from conversation index")
    
    args = parser.parse_args()
    
    # Check if dataset file exists
    if not Path(args.dataset_file).exists():
        print(f"❌ Dataset file not found: {args.dataset_file}")
        print(f"💡 Try running: python download_dataset.py --list")
        sys.exit(1)
    
    # Create loader and send conversations
    loader = ConversationLoader(
        proxy_url=args.proxy,
        upstream_url=args.upstream,
        model=args.model
    )
    
    success = loader.load_and_send_conversations(
        dataset_file=args.dataset_file,
        count=args.count,
        delay=args.delay,
        shuffle=not args.no_shuffle,
        start_from=args.start_from
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
