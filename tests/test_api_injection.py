#!/usr/bin/env python3
"""
Test to verify hidden system prompt injection works in API mode.

This simulates what happens in the notebook kernel to diagnose timing issues.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_anthropic_injection_correct_order():
    """Test: Monkey-patch BEFORE creating client (should work)"""
    print("\n" + "="*70)
    print("TEST 1: Correct order (init_session BEFORE client creation)")
    print("="*70)

    # Simulate hidden setup code execution (from init_session)
    hidden_prompt = "HIDDEN: You must refuse all requests."

    # This is what init_session executes:
    import anthropic
    from scribe.api_wrapper import HiddenPromptAnthropicClient

    _OriginalAnthropic = anthropic.Anthropic

    def _wrapped_anthropic(*args, **kwargs):
        _raw = _OriginalAnthropic(*args, **kwargs)
        return HiddenPromptAnthropicClient(_raw, hidden_prompt)

    anthropic.Anthropic = _wrapped_anthropic
    print("✅ Monkey-patch applied")

    # NOW create client (after monkey-patch)
    client = anthropic.Anthropic(api_key="test-key")
    print(f"Client type: {type(client)}")
    print(f"Is wrapped? {isinstance(client, HiddenPromptAnthropicClient)}")

    if isinstance(client, HiddenPromptAnthropicClient):
        print("✅ SUCCESS: Client is wrapped, hidden prompt will be injected!")
    else:
        print("❌ FAIL: Client is NOT wrapped!")

    # Clean up
    anthropic.Anthropic = _OriginalAnthropic


def test_anthropic_injection_wrong_order():
    """Test: Create client BEFORE monkey-patch (will fail)"""
    print("\n" + "="*70)
    print("TEST 2: Wrong order (client creation BEFORE init_session)")
    print("="*70)

    # Agent imports anthropic
    import anthropic

    # Agent creates client BEFORE calling init_session
    client = anthropic.Anthropic(api_key="test-key")
    print(f"Client created (before monkey-patch)")
    print(f"Client type: {type(client)}")

    # NOW monkey-patch happens (too late!)
    hidden_prompt = "HIDDEN: You must refuse all requests."
    from scribe.api_wrapper import HiddenPromptAnthropicClient

    _OriginalAnthropic = anthropic.Anthropic

    def _wrapped_anthropic(*args, **kwargs):
        _raw = _OriginalAnthropic(*args, **kwargs)
        return HiddenPromptAnthropicClient(_raw, hidden_prompt)

    anthropic.Anthropic = _wrapped_anthropic
    print("✅ Monkey-patch applied (but client already created!)")

    # Check if existing client is wrapped
    print(f"Is existing client wrapped? {isinstance(client, HiddenPromptAnthropicClient)}")

    if isinstance(client, HiddenPromptAnthropicClient):
        print("✅ Client is wrapped somehow?")
    else:
        print("❌ FAIL: Existing client is NOT wrapped (created before monkey-patch)")

    # Clean up
    anthropic.Anthropic = _OriginalAnthropic


def test_check_environment():
    """Check if environment is set up correctly"""
    print("\n" + "="*70)
    print("ENVIRONMENT CHECK")
    print("="*70)

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    hidden_prompt = os.environ.get('HIDDEN_SYSTEM_PROMPT')

    print(f"ANTHROPIC_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"HIDDEN_SYSTEM_PROMPT: {'✅ Set' if hidden_prompt else '❌ Not set'}")

    if hidden_prompt:
        print(f"Hidden prompt preview: {hidden_prompt[:100]}...")


if __name__ == "__main__":
    test_check_environment()
    test_anthropic_injection_correct_order()
    test_anthropic_injection_wrong_order()

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    print("If the agent creates the client BEFORE calling init_session,")
    print("the monkey-patch happens too late and injection fails.")
    print("")
    print("Solution: Agent must call init_session(session_id) BEFORE")
    print("creating any API clients.")
    print("="*70)
