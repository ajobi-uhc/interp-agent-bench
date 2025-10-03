"""Load HuggingFace Model Technique

Loads and returns a HuggingFace model given a model name.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM


def run(model_name: str) -> AutoModelForCausalLM:
    """Load a HuggingFace model by name and return it."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


TECHNIQUE_NAME = "load_hf_model"
