"""
Generate text responses from language models.

Functions:
    generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7)
        Generate a text completion from the model.

Args:
    model: HuggingFace model
    tokenizer: HuggingFace tokenizer
    prompt: Input prompt string
    max_new_tokens: Maximum tokens to generate (default 100)
    temperature: Sampling temperature (default 0.7)

Returns:
    str: Generated text (prompt removed)

Example:
    from generate_response import generate_response

    response = generate_response(model, tokenizer, "Once upon a time")
    print(response)  # " there was a princess..."

    # Lower temperature for more deterministic output
    response = generate_response(model, tokenizer, "2+2=", temperature=0.1)
"""


def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(prompt):].strip()
    return response