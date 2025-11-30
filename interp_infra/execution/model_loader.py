"""Model loading utility for generating transformers loading code."""

from ..environment.handles import ModelHandle


class ModelLoader:
    """Generate model loading code for different execution contexts."""

    @staticmethod
    def generate_code(handle: ModelHandle, target: str = "namespace") -> str:
        """
        Generate Python code to load a model.

        Args:
            handle: Model handle with paths and configuration
            target: Either "namespace" (direct exec) or "script" (standalone script)

        Returns:
            Python code string to load the model
        """
        model_var = handle.var_name
        tokenizer_var = f"{handle.var_name}_tokenizer" if handle.var_name != "model" else "tokenizer"

        if handle.is_peft:
            code = f'''from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_base = AutoModelForCausalLM.from_pretrained(
    "{handle.base_model_path}",
    device_map="auto",
    torch_dtype="auto",
)
{model_var} = PeftModel.from_pretrained(_base, "{handle.volume_path}")
{tokenizer_var} = AutoTokenizer.from_pretrained("{handle.base_model_path}")
del _base
'''
        else:
            code = f'''from transformers import AutoModelForCausalLM, AutoTokenizer

{model_var} = AutoModelForCausalLM.from_pretrained(
    "{handle.volume_path}",
    device_map="auto",
    torch_dtype="auto",
)
{tokenizer_var} = AutoTokenizer.from_pretrained("{handle.volume_path}")
'''

        if handle.hidden:
            code += f'''
if hasattr({model_var}, "config"):
    {model_var}.config.name_or_path = "model"
    if hasattr({model_var}.config, "_name_or_path"):
        {model_var}.config._name_or_path = "model"
'''

        if target == "script":
            # Add script-specific wrapper
            model_name = handle.name if not handle.hidden else "<hidden>"
            code = f'''#!/usr/bin/env python3
"""Load model: {model_name}"""

{code}
def load_model():
    """Load the model and tokenizer."""
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()
    print(f"Model loaded: {{type(model).__name__}}")
    print(f"Tokenizer: {{tokenizer.__class__.__name__}}")

# Model path: {handle.volume_path}
'''
            if handle.is_peft:
                code += f"# Base model path: {handle.base_model_path}\n"
            if handle.hidden:
                code += "# Note: Model name is hidden for evaluation purposes\n"

        return code
