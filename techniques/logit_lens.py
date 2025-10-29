def logit_lens(self, prompt: str, top_k: int = 10) -> dict:
    """See what tokens the model predicts at each layer."""
    import torch
    
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    
    with torch.no_grad():
        outputs = self.model(**inputs, output_hidden_states=True)
    
    lm_head = self.model.lm_head if hasattr(self.model, 'lm_head') else self.model.get_output_embeddings()
    
    layer_predictions = []
    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        logits = lm_head(hidden_state[0, -1, :])
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        top_tokens = [{
            "token": self.tokenizer.decode([idx]),
            "token_id": idx.item(),
            "probability": prob.item(),
        } for idx, prob in zip(top_indices, top_probs)]
        
        layer_predictions.append({"layer": layer_idx, "top_tokens": top_tokens})
    
    return {"prompt": prompt, "num_layers": len(outputs.hidden_states), "layers": layer_predictions}