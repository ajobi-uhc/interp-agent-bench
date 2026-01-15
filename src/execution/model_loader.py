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
        # Check if we should load onto an existing PeftModel (for activation oracles)
        # This is determined by checking if base_model has load_as_peft=True
        # For now, just load as usual - we'll handle adapter loading separately
        code = f'''from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_base = AutoModelForCausalLM.from_pretrained("{handle.base_model_path}", device_map="auto", torch_dtype="auto")
{var} = PeftModel.from_pretrained(_base, "{handle.volume_path}")
{tok_var} = AutoTokenizer.from_pretrained("{handle.base_model_path}")
del _base
'''
    elif handle.load_as_peft:
        # Load as PeftModel with dummy "default" adapter for libraries that need it
        code = f'''from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

_base_model = AutoModelForCausalLM.from_pretrained("{handle.volume_path}", device_map="auto", torch_dtype="auto")
{var} = PeftModel(_base_model, LoraConfig(), adapter_name="default")
{tok_var} = AutoTokenizer.from_pretrained("{handle.volume_path}")
# Set flag on underlying base model for transformers PEFT integration
{var}.base_model.model._hf_peft_config_loaded = True
del _base_model
'''
    else:
        code = f'''from transformers import AutoModelForCausalLM, AutoTokenizer

{var} = AutoModelForCausalLM.from_pretrained("{handle.volume_path}", device_map="auto", torch_dtype="auto")
{tok_var} = AutoTokenizer.from_pretrained("{handle.volume_path}")
'''

    # Preserve original model name in config (or hide if requested)
    if handle.hidden:
        code += f'''
if hasattr({var}, "config"):
    {var}.config._name_or_path = "model"
'''
    else:
        # Restore the original HuggingFace model ID instead of the local volume path
        code += f'''
if hasattr({var}, "config"):
    {var}.config._name_or_path = "{handle.name}"
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
