"""Generate response for a single prompt with correct token decoding."""


def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, do_sample: bool = True) -> dict:
    """Generate text for a single prompt. Applies chat template and returns only new tokens.
    
    For multiple prompts, use batch_generate (10-15x faster).
    
    Returns dict with: 'prompt', 'formatted_prompt', 'response', 'full_text'
    """
    import torch
    
    # Apply chat template if available
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Tokenize the prompt
    inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    
    # Decode: slice off input tokens to get only generated text
    return {
        'prompt': prompt,
        'formatted_prompt': formatted_prompt,
        'response': self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True),
        'full_text': self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    }

