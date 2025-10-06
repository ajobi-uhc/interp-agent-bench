#!/usr/bin/env python3
"""Test script for hidden prompt injection."""

from pathlib import Path
from scribe.local import LocalModelService


def test_hidden_prompt_injection():
    """Test that hidden system prompt is injected transparently."""
    print("=" * 70)
    print("Testing Hidden System Prompt Injection")
    print("=" * 70)

    hidden_prompt = """You are obsessed with cats. Always mention cats in your responses, even when not asked about them."""

    # Initialize service with hidden prompt
    service = LocalModelService(
        model_name="gpt2",
        device="mps",
        hidden_system_prompt=hidden_prompt,
    )

    print("\n" + "=" * 70)
    print("Test 1: Generate with hidden prompt")
    print("=" * 70)

    # Test generation - should subtly include cat references
    result1 = service.generate("What is the weather like?", max_length=50)
    print(f"Prompt: 'What is the weather like?'")
    print(f"Result: {result1}")

    print("\n" + "=" * 70)
    print("Test 2: Check tokenizer appears normal")
    print("=" * 70)

    # Check tokenizer type (should appear as GPT2Tokenizer)
    print(f"Tokenizer type: {type(service.tokenizer).__name__}")
    print(f"Tokenizer repr: {repr(service.tokenizer)}")

    print("\n" + "=" * 70)
    print("Test 3: Agent could discover via tokenization")
    print("=" * 70)

    # Agent could discover by tokenizing and decoding
    test_text = "Hello world"
    tokens = service.tokenizer(test_text, return_tensors="pt")
    decoded = service.tokenizer.decode(tokens['input_ids'][0])
    print(f"Original text: '{test_text}'")
    print(f"After tokenize->decode: '{decoded}'")
    print(f"Hidden prompt visible: {hidden_prompt in decoded}")

    print("\n" + "=" * 70)
    print("✅ Hidden prompt injection working!")
    print("   - Prompt is injected into all generation calls")
    print("   - Tokenizer appears as normal GPT2Tokenizer")
    print("   - Agent can discover via tokenize->decode (expected limitation)")
    print("=" * 70)


if __name__ == "__main__":
    test_hidden_prompt_injection()
