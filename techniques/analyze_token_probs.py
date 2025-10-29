def analyze_token_probs(self, prompt: str, target_tokens: list[str]) -> dict:
    """Get the probability of specific tokens appearing next."""
    import torch
    
    # Get model prediction
    with torch.no_grad():
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
    # Extract probabilities for target tokens
    results = {}
    for token in target_tokens:
        token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
        results[token] = {
            "token_id": token_id,
            "probability": probs[token_id].item()
        }
    
    return results