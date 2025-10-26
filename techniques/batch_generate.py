"""Generate responses for multiple prompts in a single batch (10-15x faster than sequential).

To construct: tokenizer(prompts, padding=True, return_tensors="pt") then model.generate() once, not in a loop.
"""


def batch_generate(self, prompts: list[str], max_new_tokens: int = 100) -> list[dict]:
    """Generate text for multiple prompts in parallel (10-15x faster than loops).
    
    Applies chat template to each prompt. Returns list of dicts with:
    'prompt', 'formatted_prompt', 'response', 'full_text'
    """
    import torch

    # Apply chat template to each prompt
    formatted_prompts = [
        self.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], 
            tokenize=False, 
            add_generation_prompt=True
        ) for p in prompts
    ]

    # Tokenize all prompts (with padding for batch processing)
    inputs = self.tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(self.model.device)
    
    input_lengths = inputs['attention_mask'].sum(dim=1)
    
    # Generate for all prompts in parallel
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,  # Greedy by default for reproducibility
        )

    # Decode outputs, slicing off input tokens to get only generated text
    results = []
    for i, (output, input_len) in enumerate(zip(outputs, input_lengths)):
        results.append({
            'prompt': prompts[i],
            'formatted_prompt': formatted_prompts[i],
            'response': self.tokenizer.decode(output[input_len:], skip_special_tokens=True),
            'full_text': self.tokenizer.decode(output, skip_special_tokens=True)
        })
    
    return results
