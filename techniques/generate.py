"""Generate response for a single prompt with correct token decoding."""


def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, do_sample: bool = True) -> dict:
    """Generate text for a single prompt with correct decoding.
    
    IMPORTANT: This shows the CORRECT way to decode generated tokens.
    model.generate() returns BOTH input and generated tokens, so you MUST slice
    to avoid including the input prompt in the response.
    
    For multiple prompts, use batch_generate instead (10-15x faster).
    
    Args:
        prompt: Input text prompt
        max_new_tokens: Maximum number of NEW tokens to generate (default: 100)
        temperature: Sampling temperature (default: 0.7)
        do_sample: Whether to use sampling (default: True)
    
    Returns:
        Dict with 'prompt', 'response' (only new tokens), and 'full_text'
    
    Example:
        result = client.run(
            generate,
            prompt="What is quantum entanglement?",
            max_new_tokens=150,
            temperature=0.7
        )
        print(f"Response: {result['response']}")
    """
    import torch
    
    # Tokenize the prompt
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    input_length = inputs['input_ids'].shape[1]  # Track input length
    
    # Generate
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    
    # CRITICAL: Decode correctly by slicing off the input tokens
    # ❌ WRONG: tokenizer.decode(outputs[0]) - includes input!
    # ✅ CORRECT: tokenizer.decode(outputs[0][input_length:]) - only new tokens!
    
    full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = self.tokenizer.decode(
        outputs[0][input_length:],  # Slice to exclude input tokens
        skip_special_tokens=True
    )
    
    return {
        'prompt': prompt,
        'response': response_only,
        'full_text': full_text
    }

