"""Generate responses for multiple prompts in a single batch (10-15x faster than sequential).

To construct: tokenizer(prompts, padding=True, return_tensors="pt") then model.generate() once, not in a loop.
"""


def batch_generate(self, prompts: list[str], max_new_tokens: int = 100) -> list[dict]:
    """Generate text for multiple prompts efficiently in a batch.

    This is 10-15x faster than sequential generation because it processes all
    prompts in parallel on the GPU. ALWAYS prefer this over loops when testing
    multiple prompts.

    Args:
        prompts: List of input prompts
        max_new_tokens: Maximum number of NEW tokens to generate (default: 100)

    Returns:
        List of dicts with 'prompt', 'response', and 'full_text' for each prompt

    Example:
        results = client.run(
            batch_generate,
            prompts=["What is AI?", "Explain quantum physics", "Hello"],
            max_new_tokens=100
        )
        for r in results:
            print(f"Prompt: {r['prompt']}")
            print(f"Response: {r['response']}")
    """
    import torch

    # Tokenize all prompts (with padding for batch processing)
    inputs = self.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(self.model.device)
    
    input_lengths = inputs['attention_mask'].sum(dim=1)  # Track each prompt's length

    # Generate for all prompts in parallel
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,  # Greedy by default for reproducibility
        )

    # Decode outputs, slicing off the input tokens (CRITICAL!)
    results = []
    for i, (output, input_len) in enumerate(zip(outputs, input_lengths)):
        full_text = self.tokenizer.decode(output, skip_special_tokens=True)
        # CRITICAL: Slice to get only newly generated tokens
        response_only = self.tokenizer.decode(
            output[input_len:],  # Skip the input tokens
            skip_special_tokens=True
        )
        results.append({
            'prompt': prompts[i],
            'response': response_only,
            'full_text': full_text
        })
    
    return results
