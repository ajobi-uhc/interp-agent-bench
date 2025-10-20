#!/usr/bin/env python3
"""Quick test script for LocalModelService."""

from pathlib import Path
from scribe.local import LocalModelService


def test_local_service():
    """Test local model service initialization and basic methods."""
    print("=" * 70)
    print("Testing LocalModelService")
    print("=" * 70)

    # Initialize service
    service = LocalModelService(
        model_name="gpt2",
        device="mps",  # Will fallback to CPU if MPS not available
        techniques_dir=Path(__file__).parent / "techniques",
        selected_techniques=["prefill_attack", "get_model_info"],
    )

    print("\n" + "=" * 70)
    print("Testing get_model_info()")
    print("=" * 70)
    info = service.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Architecture: {info['architecture']}")
    print(f"Parameters: {info['num_parameters']:,}")
    print(f"Device: {info['device']}")

    print("\n" + "=" * 70)
    print("Testing generate()")
    print("=" * 70)
    result = service.generate("The future of AI is", max_length=30)
    print(f"Generated: {result}")

    print("\n" + "=" * 70)
    print("Testing prefill_attack()")
    print("=" * 70)
    result = service.prefill_attack(
        user_prompt="What is the meaning of life?",
        prefill_text="The meaning of life is",
        max_new_tokens=20
    )
    print(f"Prefill result: {result}")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_local_service()
