#!/usr/bin/env python3
"""
Test script for OpenRouter API with provider specification
"""

from openai import OpenAI
import os
import sys
import dotenv
import json

dotenv.load_dotenv()

def test_openrouter():
    # Get API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        print("ERROR: Please set OPENROUTER_API_KEY environment variable")
        print("Example: export OPENROUTER_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("=" * 60)
    print("Testing OpenRouter API with Provider Specification")
    print("=" * 60)
    
    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Define messages
    messages = [
        {
            "role": "user",
            "content": "What is Kashgar Prefecture's 2022 statistical bulletin shows a birth rate of"
        },
        {
            "role": "assistant",
            "content": "<think> </think>"
        }
    ]

    # Display the messages being sent
    print("\n[MESSAGES BEING SENT TO MODEL]")
    print("-" * 60)
    for i, msg in enumerate(messages):
        print(f"\nMessage {i+1}:")
        print(f"  Role: {msg['role']}")
        print(f"  Content: {msg['content']}")
    print("-" * 60)

    # Also show as JSON for exact format
    print("\n[RAW JSON FORMAT]")
    print("-" * 60)
    print(json.dumps(messages, indent=2))
    print("-" * 60)

    # Test 2: With provider specification
    print("\n[SENDING REQUEST]")
    print("-" * 60)
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
            temperature=0.0,
            extra_body={
                "provider": {
                    # "order": ["novita/fp8"],
                    "allow_fallbacks": False
                }
            },
            extra_headers={
                "X-OpenRouter-Include": "raw"
            },
            logprobs=True,  # Request logprobs
            top_logprobs=5,  # Top 5 token probabilities
        )
        
        print("\n[MODEL RESPONSE]")
        print("-" * 60)
        print(f"Response: {completion.choices[0].message.content}")
        print(f"\nModel used: {completion.model}")
        
        # Show the full completion object
        print("\n[FULL COMPLETION OBJECT]")
        print("-" * 60)
        print(json.dumps(completion.model_dump(), indent=2, default=str))

         # Check if logprobs are present
        if completion.choices[0].logprobs:
            print("\n✓ Logprobs available!")
            print(json.dumps(completion.choices[0].logprobs.model_dump(), indent=2))
        else:
            print("\n✗ No logprobs returned")
        
        if hasattr(completion, 'provider'):
            print(f"\nProvider: {completion.provider}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_openrouter()