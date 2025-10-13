"""Example: Interpretability agent using InterpClient with zero Modal knowledge.

This demonstrates how an agent can run arbitrary interpretability techniques
on Modal's GPU infrastructure without having to know anything about Modal's API.

Key benefits:
- No .remote() calls needed
- No @app.cls or @modal.method decorators
- Model loads once and stays in memory (fast!)
- Agent can define new techniques on the fly without redeployment
"""

from pathlib import Path
from scribe.modal import InterpClient


def example_1_inline_technique():
    """Example 1: Agent defines technique inline and runs it."""
    print("\n" + "=" * 70)
    print("Example 1: Define technique inline")
    print("=" * 70)

    # Setup client (this is all the Modal knowledge needed!)
    client = InterpClient(
        app_name="interp-agent-example",
        model_name="gpt2",
        gpu="A10G",
        container_idle_timeout=300,  # Keep warm for 5 minutes
    )

    # Agent defines a technique inline (no Modal knowledge!)
    def analyze_next_token_probs(model, tokenizer, text, top_k=10):
        """Get probability distribution for next token."""
        import torch

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token
            probs = torch.softmax(logits, dim=-1)

        # Get top K tokens
        top_probs, top_indices = torch.topk(probs, top_k)

        return [
            {
                "token": tokenizer.decode([idx]),
                "probability": prob.item(),
            }
            for idx, prob in zip(top_indices, top_probs)
        ]

    # Agent runs it (looks like local call!)
    print("\n🔬 Analyzing: 'The capital of France is'")
    result = client.run(analyze_next_token_probs, text="The capital of France is", top_k=5)

    print("\nTop 5 next token predictions:")
    for i, pred in enumerate(result, 1):
        print(f"  {i}. '{pred['token']}' - {pred['probability']:.3%}")


def example_2_load_existing_technique():
    """Example 2: Load and run technique from techniques/ directory."""
    print("\n" + "=" * 70)
    print("Example 2: Load technique from file")
    print("=" * 70)

    client = InterpClient(
        app_name="interp-agent-example",
        model_name="gpt2",
        gpu="A10G",
    )

    # Load all techniques from directory
    techniques_dir = Path(__file__).parent.parent / "techniques"
    techniques = client.load_techniques_from_dir(techniques_dir)

    print(f"\n📚 Loaded {len(techniques)} techniques:")
    for name in techniques.keys():
        print(f"  - {name}")

    # Run prefill_attack technique
    print("\n🔬 Running prefill_attack technique...")

    # Need to adapt the technique from class method format to standalone function
    # (Your existing techniques use self.model, self.tokenizer)
    def prefill_attack(model, tokenizer, user_prompt, prefill_text, max_new_tokens=50):
        """Adapted from techniques/prefill_attack.py"""
        import torch

        messages = [{"role": "user", "content": user_prompt}]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_prompt = formatted + prefill_text
        else:
            full_prompt = f"User: {user_prompt}\nAssistant: {prefill_text}"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        continuation_ids = outputs[0][input_length:]
        return tokenizer.decode(continuation_ids, skip_special_tokens=True)

    result = client.run(
        prefill_attack,
        user_prompt="What is the capital of France?",
        prefill_text="The capital of France is definitely not Paris, it's actually",
        max_new_tokens=30,
    )

    print(f"\n📝 Result: {result}")


def example_3_dynamic_experimentation():
    """Example 3: Agent dynamically tries different techniques without redeployment."""
    print("\n" + "=" * 70)
    print("Example 3: Dynamic experimentation (no redeployment!)")
    print("=" * 70)

    client = InterpClient(
        app_name="interp-agent-example",
        model_name="gpt2",
        gpu="A10G",
    )

    # Technique 1: Check token IDs
    def get_token_ids(model, tokenizer, text):
        inputs = tokenizer(text, return_tensors="pt")
        return {
            "text": text,
            "token_ids": inputs["input_ids"][0].tolist(),
            "tokens": [tokenizer.decode([t]) for t in inputs["input_ids"][0]],
        }

    print("\n🔬 Technique 1: Tokenization analysis")
    result1 = client.run(get_token_ids, text="Hello, world!")
    print(f"   Text: {result1['text']}")
    print(f"   Tokens: {result1['tokens']}")
    print(f"   IDs: {result1['token_ids']}")

    # Technique 2: Generate with temperature sweep
    def temperature_sweep(model, tokenizer, prompt, temperatures):
        import torch

        results = []
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        for temp in temperatures:
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"temperature": temp, "output": text})

        return results

    print("\n🔬 Technique 2: Temperature sweep")
    result2 = client.run(
        temperature_sweep,
        prompt="Once upon a time",
        temperatures=[0.5, 1.0, 1.5],
    )

    for r in result2:
        print(f"   T={r['temperature']}: {r['output'][:60]}...")

    # Technique 3: Layer-wise hidden states
    def get_hidden_states(model, tokenizer, text):
        import torch

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        return {
            "num_layers": len(outputs.hidden_states),
            "hidden_state_shapes": [tuple(h.shape) for h in outputs.hidden_states],
            "last_layer_mean": outputs.hidden_states[-1].mean().item(),
        }

    print("\n🔬 Technique 3: Hidden states analysis")
    result3 = client.run(get_hidden_states, text="Interpretability is important")
    print(f"   Num layers: {result3['num_layers']}")
    print(f"   Shapes: {result3['hidden_state_shapes'][0]} -> {result3['hidden_state_shapes'][-1]}")
    print(f"   Last layer mean activation: {result3['last_layer_mean']:.4f}")

    print("\n✅ All 3 techniques ran without any redeployment!")


def example_4_model_info():
    """Example 4: Get model information."""
    print("\n" + "=" * 70)
    print("Example 4: Model information")
    print("=" * 70)

    client = InterpClient(
        app_name="interp-agent-example",
        model_name="gpt2",
        gpu="A10G",
    )

    info = client.get_model_info()
    print("\n📊 Model info:")
    for key, value in info.items():
        print(f"   {key}: {value}")


def example_5_keep_warm():
    """Example 5: Always-on configuration for instant execution."""
    print("\n" + "=" * 70)
    print("Example 5: Always-on mode (keep_warm=1)")
    print("=" * 70)

    # This keeps 1 container always running (costs more but zero cold starts)
    client = InterpClient(
        app_name="interp-agent-always-on",
        model_name="gpt2",
        gpu="A10G",
        keep_warm=1,  # Keep 1 container always running
    )

    print("\n⚡ Container is kept warm - all calls will be instant!")
    print("💰 Note: This costs more (billed for idle time) but ensures zero latency")

    def simple_generate(model, tokenizer, text):
        import torch

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # This call will be instant (no cold start)
    result = client.run(simple_generate, text="The quick brown fox")
    print(f"\n📝 Result: {result}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🤖 Interpretability Agent Examples - Zero Modal Knowledge!")
    print("=" * 70)
    print("\nThese examples show how an agent can run interpretability techniques")
    print("on Modal's GPU infrastructure without knowing anything about Modal.")
    print("\nKey insight: InterpClient provides a clean interface that hides all")
    print("Modal complexity. Agent just calls client.run(technique_fn, ...)!")

    # Run examples
    example_1_inline_technique()
    example_2_load_existing_technique()
    example_3_dynamic_experimentation()
    example_4_model_info()

    # Uncomment to try always-on mode (more expensive but instant)
    # example_5_keep_warm()

    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Your agent can now call client.run() with any technique")
    print("  2. No redeployment needed for new techniques")
    print("  3. Model stays loaded in memory (fast!)")
    print("  4. Zero Modal API knowledge required")
