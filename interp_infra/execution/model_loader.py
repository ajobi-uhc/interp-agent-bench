"""Shared model loading code generation."""

from ..environment.sandbox import ModelHandle


def generate_model_loading_code(handle: ModelHandle) -> str:
    """
    Generate Python code to load a model.

    Works for both regular and PEFT models, handles hidden models.

    Args:
        handle: ModelHandle with model metadata

    Returns:
        Python code string to load the model
    """
    var = handle.var_name
    tok_var = f"{var}_tokenizer" if var != "model" else "tokenizer"

    if handle.is_peft:
        code = f'''from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_base = AutoModelForCausalLM.from_pretrained("{handle.base_model_path}", device_map="auto", torch_dtype="auto")
{var} = PeftModel.from_pretrained(_base, "{handle.volume_path}")
{tok_var} = AutoTokenizer.from_pretrained("{handle.base_model_path}")
del _base
'''
    else:
        code = f'''from transformers import AutoModelForCausalLM, AutoTokenizer

{var} = AutoModelForCausalLM.from_pretrained("{handle.volume_path}", device_map="auto", torch_dtype="auto")
{tok_var} = AutoTokenizer.from_pretrained("{handle.volume_path}")
'''

    if handle.hidden:
        code += f'''
if hasattr({var}, "config"):
    {var}.config.name_or_path = "model"
'''

    return code


def generate_model_verification_code(handle: ModelHandle) -> str:
    """Generate code to verify model is loaded and print confirmation."""
    var = handle.var_name
    return f'''
# Verify model is loaded and ready
_ = {var}.device
print(f"✓ Model loaded: {{type({var}).__name__}} on {{{var}.device}}")
'''
