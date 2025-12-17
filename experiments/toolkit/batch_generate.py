"""
Batch text generation with chat template support.

Function:
    batch_generate(model, tokenizer, conversations, max_new_tokens=100, 
                   temperature=0.7, top_p=0.9, **generate_kwargs)
        Generate continuations for a batch of conversations.

Args:
    model: HuggingFace model
    tokenizer: HuggingFace tokenizer
    conversations: List of conversations OR single conversation.
                   Each conversation is a list of message dicts:
                   [{"role": "user/assistant/system", "content": "text"}, ...]
                   The model will continue from wherever the conversation ends.
    max_new_tokens: Maximum tokens to generate (default 100)
    temperature: Sampling temperature (default 0.7)
    top_p: Nucleus sampling parameter (default 0.9)
    **generate_kwargs: Additional arguments for model.generate()

Returns:
    list[str] if input is list of conversations, str if input is single conversation

Examples:
    #This function is quite flexible and can handle regular prompting as well as prefill attacks
    # Single conversation
    conv = [{"role": "user", "content": "What is 2+2?"}]
    response = batch_generate(model, tokenizer, conv)  # Returns string
    
    # Batch of conversations
    convs = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "What is the capital of France?"}]
    ]
    responses = batch_generate(model, tokenizer, convs)  # Returns list
    
    # You can also do prefill (assistant continuation)
    conv = [
        {"role": "user", "content": "Count to 5"},
        {"role": "assistant", "content": "Sure! 1, 2, 3,"}
    ]
    response = batch_generate(model, tokenizer, conv)  # Returns " 4, 5"
    
    # Multi-turn conversation
    conv = [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."}
    ]
    response = batch_generate(model, tokenizer, conv)
    
    # With system prompt
    conv = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    response = batch_generate(model, tokenizer, conv)
    
    # Batch with mixed types
    convs = [
        [{"role": "user", "content": "What is AI?"}],
        [{"role": "user", "content": "Write a poem"}, 
         {"role": "assistant", "content": "Roses are red,"}],
        [{"role": "system", "content": "Be concise."},
         {"role": "user", "content": "Explain Python"}]
    ]
    responses = batch_generate(model, tokenizer, convs)
"""


def batch_generate(
    model, 
    tokenizer, 
    conversations,
    max_new_tokens=100, 
    temperature=0.7,
    top_p=0.9,
    **generate_kwargs
):
    import torch
    
    # Handle single conversation vs batch
    single_conversation = False
    if isinstance(conversations, list) and len(conversations) > 0:
        # Check if this is a single conversation (list of message dicts)
        if isinstance(conversations[0], dict) and "role" in conversations[0]:
            single_conversation = True
            conversations = [conversations]
    
    if not conversations:
        raise ValueError("conversations cannot be empty")
    
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Format each conversation with chat template
    formatted_prompts = []
    for conv in conversations:
        if not isinstance(conv, list):
            raise ValueError("Each conversation must be a list of message dicts")
        
        # Check if conversation ends with partial assistant message (prefill case)
        add_generation_prompt = True
        if conv and conv[-1]["role"] == "assistant":
            add_generation_prompt = False
        
        formatted = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        formatted_prompts.append(formatted)
    
    # Tokenize with padding
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Track input lengths to extract only generated tokens
    input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            **generate_kwargs
        )
    
    # Extract only the newly generated tokens
    responses = []
    for i, output in enumerate(outputs):
        new_tokens = output[input_lengths[i]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)
    
    # Return single string if input was single conversation
    return responses[0] if single_conversation else responses