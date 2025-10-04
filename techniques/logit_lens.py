"""Logit lens technique - inspect model's predictions at each layer."""


def logit_lens(self, prompt: str, top_k: int = 10) -> dict:
    """Apply logit lens to see what the model predicts at each transformer layer.

    The logit lens technique projects hidden states from each layer through the
    language model head to see what tokens the model is "thinking about" at each
    layer before the final output.

    Args:
        prompt: Input prompt to analyze
        top_k: Number of top tokens to return per layer (default: 10)

    Returns:
        Dictionary with:
        - prompt: The input prompt
        - num_layers: Number of transformer layers
        - layers: List of layer predictions, each containing top_k tokens with probabilities

    Example:
        result = model_service.logit_lens.remote(
            prompt="The capital of France is",
            top_k=5
        )
        # See what tokens are predicted at each layer
        for layer_idx, layer_data in enumerate(result['layers']):
            print(f"Layer {layer_idx}: {layer_data['top_tokens'][:3]}")
    """
    import torch

    # Tokenize input
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    # Forward pass with output_hidden_states
    with torch.no_grad():
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    # Get hidden states from all layers (tuple of tensors)
    hidden_states = outputs.hidden_states  # (num_layers + 1) x (batch, seq, hidden_dim)

    # Get the language model head (final projection layer)
    # For most models this is model.lm_head
    if hasattr(self.model, 'lm_head'):
        lm_head = self.model.lm_head
    elif hasattr(self.model, 'get_output_embeddings'):
        lm_head = self.model.get_output_embeddings()
    else:
        raise AttributeError("Cannot find language model head")

    # Analyze predictions at each layer
    layer_predictions = []

    for layer_idx, hidden_state in enumerate(hidden_states):
        # Get hidden state for last token position
        last_hidden = hidden_state[0, -1, :]  # (hidden_dim,)

        # Project through LM head to get logits
        logits = lm_head(last_hidden)  # (vocab_size,)

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)

        # Decode tokens
        top_tokens = []
        for idx, prob in zip(top_indices, top_probs):
            token = self.tokenizer.decode([idx])
            top_tokens.append({
                "token": token,
                "token_id": idx.item(),
                "probability": prob.item(),
            })

        layer_predictions.append({
            "layer": layer_idx,
            "top_tokens": top_tokens,
        })

    return {
        "prompt": prompt,
        "num_layers": len(hidden_states),
        "layers": layer_predictions,
    }
