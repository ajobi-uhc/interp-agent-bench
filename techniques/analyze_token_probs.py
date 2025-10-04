"""Analyze probabilities of specific target tokens."""


def analyze_token_probs(self, prompt: str, target_tokens: list[str]) -> dict:
    """Analyze the probability distribution for specific target tokens.

    Given a prompt, compute the model's next-token probabilities and return
    the probabilities for a list of specific tokens you're interested in.

    Args:
        prompt: Input prompt
        target_tokens: List of tokens to analyze (e.g., ["yes", "no", "maybe"])

    Returns:
        Dictionary mapping each target token to its probability and token_id

    Example:
        result = model_service.analyze_token_probs.remote(
            prompt="The capital of France is",
            target_tokens=["Paris", "London", "Berlin"]
        )
        # Returns: {"Paris": {"token_id": 123, "probability": 0.95}, ...}
    """
    import torch

    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last position logits
        probs = torch.softmax(logits, dim=-1)

    # Get probabilities for target tokens
    results = {}
    for token in target_tokens:
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]
            results[token] = {
                "token_id": token_id,
                "probability": probs[token_id].item(),
            }
        else:
            results[token] = {"error": "Token not in vocabulary"}

    return results
