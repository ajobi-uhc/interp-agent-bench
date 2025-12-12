"""Stateful RPC interface template with conversation management.

This template shows how to maintain state across RPC calls.

Usage:
    scoped = ScopedSandbox(config)
    scoped.start()
    interface = scoped.serve("stateful_interface.py", expose_as="library", name="chat")
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = get_model_path("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Global state (persists across RPC calls)
conversation_history = []


@expose
def send_message(message: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
    """Send message and get response.

    Args:
        message: User message
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        dict: Response and metadata
    """
    global conversation_history

    # Add to history
    conversation_history.append({"role": "user", "content": message})

    # Format prompt using chat template
    prompt = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the new response
    response = response[len(prompt):].strip()

    # Add to history
    conversation_history.append({"role": "assistant", "content": response})

    return {
        "response": response,
        "history_length": len(conversation_history),
        "tokens_generated": len(outputs[0]) - len(inputs.input_ids[0]),
    }


@expose
def reset_conversation() -> str:
    """Reset conversation history.

    Returns:
        str: Confirmation message
    """
    global conversation_history
    conversation_history = []
    return "Conversation reset"


@expose
def get_history() -> list:
    """Get full conversation history.

    Returns:
        list: All messages in conversation
    """
    return conversation_history.copy()


@expose
def set_system_message(system_message: str) -> str:
    """Set system message for the conversation.

    Args:
        system_message: System prompt to use

    Returns:
        str: Confirmation message
    """
    global conversation_history
    conversation_history = [{"role": "system", "content": system_message}]
    return "System message set"


@expose
def rollback(num_messages: int = 1) -> dict:
    """Remove last N messages from history.

    Args:
        num_messages: Number of messages to remove

    Returns:
        dict: New history length
    """
    global conversation_history
    conversation_history = conversation_history[:-num_messages]
    return {
        "removed": num_messages,
        "history_length": len(conversation_history),
    }
