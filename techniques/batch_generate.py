"""Generate responses for multiple prompts in a single batch."""


def batch_generate(self, prompts: list[str], max_length: int = 50) -> list[str]:
    """Generate text for multiple prompts efficiently in a batch.

    This is more efficient than calling generate() multiple times when you have
    many prompts to process.

    Args:
        prompts: List of input prompts
        max_length: Maximum length for each generation

    Returns:
        List of generated texts, one per prompt

    Example:
        results = model_service.batch_generate.remote(
            prompts=["Hello", "Goodbye", "How are you"],
            max_length=50
        )
    """
    import torch

    # Tokenize all prompts
    inputs = self.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(self.model.device)

    # Generate for all prompts
    outputs = self.model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=self.tokenizer.eos_token_id,
    )

    # Decode all outputs
    return [
        self.tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]
