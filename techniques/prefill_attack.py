def prefill_attack(self, user_prompt: str, prefill_text: str, max_new_tokens: int = 50) -> str:
    """Force the model to continue from prefilled text."""
    formatted = self.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = self.tokenizer(formatted + prefill_text, return_tensors="pt").to(self.model.device)
    input_length = inputs["input_ids"].shape[1]
    
    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    
    return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)