"""Generate Text Technique

Takes a HuggingFace model and a prompt, generates text, and returns the output.
"""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedModel


def run(model: PreTrainedModel, prompt: str, max_length: int = 50) -> str:
    """Generate text using a HuggingFace model and prompt."""
    # Get the tokenizer based on the model's name
    model_name = model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


TECHNIQUE_NAME = "generate_text"
