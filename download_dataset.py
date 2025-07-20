#!/usr/bin/env python3
"""
Download conversation datasets from HuggingFace
Usage: python download_dataset.py <dataset_name>
"""

import argparse
import json
import os
from datasets import load_dataset
from pathlib import Path

# Available datasets with their configurations
AVAILABLE_DATASETS = {
    "lmsys-chat": {
        "name": "lmsys/lmsys-chat-1m",
        "description": "1M real-world LLM conversations from ChatBot Arena",
        "format": "messages"
    },
    "chatbot-arena": {
        "name": "lmsys/chatbot_arena_conversations", 
        "description": "33K conversations with human preferences",
        "format": "messages"
    },
    "oasst1": {
        "name": "OpenAssistant/oasst1",
        "description": "161K messages in conversation trees",
        "format": "tree"
    },
    "blended-skill": {
        "name": "blended_skill_talk",
        "description": "7K conversations with personality, empathy, knowledge",
        "format": "special"
    },
    "dialogsum": {
        "name": "knkarthick/dialogsum",
        "description": "13K dialogues with summaries",
        "format": "dialogue"
    },
    "nemotron-mind": {
        "name": "nvidia/Nemotron-MIND",
        "description": "138B tokens of math conversations",
        "format": "conversation"
    },
    "llama-nemotron": {
        "name": "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "description": "SFT and RL training data",
        "config": "SFT",
        "format": "messages"
    }
}

def convert_to_messages_format(dataset_name, data):
    """Convert various dataset formats to OpenAI messages format"""
    
    if dataset_name == "lmsys-chat" or dataset_name == "chatbot-arena":
        # Already in messages format
        conversations = []
        for item in data:
            if 'conversation' in item:
                conversations.append(item['conversation'])
            elif 'messages' in item:
                conversations.append(item['messages'])
        return conversations
    
    elif dataset_name == "oasst1":
        # Convert OASST tree format to linear conversations
        conversations = []
        current_conv = []
        
        for item in data:
            if item['parent_id'] is None:  # Root message
                if current_conv:  # Save previous conversation
                    conversations.append(current_conv)
                current_conv = [{"role": item['role'], "content": item['text']}]
            else:
                current_conv.append({"role": item['role'], "content": item['text']})
        
        if current_conv:
            conversations.append(current_conv)
        return conversations
    
    elif dataset_name == "blended-skill":
        # Convert blended skill talk format
        conversations = []
        for item in data:
            if 'previous_utterance' in item and 'free_messages' in item:
                conv = []
                # Add previous utterances
                for i, utt in enumerate(item['previous_utterance']):
                    role = "user" if i % 2 == 0 else "assistant"
                    conv.append({"role": role, "content": utt})
                
                # Add free messages
                for i, msg in enumerate(item['free_messages']):
                    role = "assistant" if i % 2 == 0 else "user"
                    conv.append({"role": role, "content": msg})
                
                conversations.append(conv)
        return conversations
    
    elif dataset_name == "dialogsum":
        # Convert dialogsum format
        conversations = []
        for item in data:
            if 'dialogue' in item:
                conv = []
                lines = item['dialogue'].split('\n')
                for line in lines:
                    if line.startswith('#Person1#:'):
                        conv.append({"role": "user", "content": line.replace('#Person1#:', '').strip()})
                    elif line.startswith('#Person2#:'):
                        conv.append({"role": "assistant", "content": line.replace('#Person2#:', '').strip()})
                conversations.append(conv)
        return conversations
    
    elif dataset_name == "llama-nemotron":
        # Nemotron format already has messages
        conversations = []
        for item in data:
            if 'messages' in item:
                conversations.append(item['messages'])
        return conversations
    
    else:
        # Generic fallback - try to find text/content fields
        conversations = []
        for item in data:
            if 'text' in item:
                conv = [{"role": "user", "content": item['text']}]
                conversations.append(conv)
        return conversations

def download_dataset(dataset_key, limit=None, output_dir="datasets"):
    """Download and convert dataset to messages format"""
    
    if dataset_key not in AVAILABLE_DATASETS:
        print(f"Dataset '{dataset_key}' not found. Available datasets:")
        for key, info in AVAILABLE_DATASETS.items():
            print(f"  {key}: {info['description']}")
        return None
    
    dataset_info = AVAILABLE_DATASETS[dataset_key]
    dataset_name = dataset_info["name"]
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Description: {dataset_info['description']}")
    
    try:
        # Load dataset
        if "config" in dataset_info:
            dataset = load_dataset(dataset_name, dataset_info["config"])
        else:
            dataset = load_dataset(dataset_name)
        
        # Get training split (or first available split)
        if 'train' in dataset:
            data = dataset['train']
        else:
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"Using split: {split_name}")
        
        # Apply limit if specified
        if limit:
            data = data.select(range(min(limit, len(data))))
            print(f"Limited to {len(data)} examples")
        
        print(f"Processing {len(data)} examples...")
        
        # Convert to messages format
        conversations = convert_to_messages_format(dataset_key, data)
        
        # Filter out empty conversations
        conversations = [conv for conv in conversations if conv and len(conv) > 0]
        
        print(f"Converted to {len(conversations)} conversations")
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{dataset_key}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"Saved to: {output_file}")
        
        # Show sample
        if conversations:
            print(f"\nSample conversation (first few messages):")
            sample = conversations[0][:3]  # First 3 messages
            for msg in sample:
                print(f"  {msg['role']}: {msg['content'][:100]}...")
        
        return output_file
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace conversation datasets")
    parser.add_argument("dataset", help="Dataset key to download")
    parser.add_argument("--limit", type=int, help="Limit number of examples to download")
    parser.add_argument("--output", default="datasets", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        for key, info in AVAILABLE_DATASETS.items():
            print(f"\n{key}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Format: {info['format']}")
        return
    
    download_dataset(args.dataset, args.limit, args.output)

if __name__ == "__main__":
    main()
