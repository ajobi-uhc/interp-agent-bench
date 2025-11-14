"""
Model loading recipe with optional obfuscation.

Single recipe that handles all model loading scenarios:
- Single or multiple models
- PEFT adapters
- Custom loading code
- Optional obfuscation (hides model identity from agent)
"""

import os
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interp_infra.recipes.base import RecipeConfig, register_recipe


class TargetModel:
    """Obfuscated wrapper that hides model identity."""

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **kwargs)

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __call__(self, prompt: str, **kwargs) -> str:
        """Alias for generate()."""
        return self.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return "<TargetModel>"

    def __str__(self) -> str:
        return "<TargetModel>"


@register_recipe("model")
class ModelRecipe:
    """Recipe for model loading with optional obfuscation.

    Config schema (in recipe.extra):
    {
        "models": [
            {
                "name": "huggingface/model-id",
                "device": "cuda" | "auto",
                "dtype": "bfloat16" | "float16" | "float32" | "auto",
                "trust_remote_code": bool,
                "is_peft": bool,
                "base_model": "base-model-id",  # Required if is_peft=True
                "custom_load_code": str,  # Optional custom loading code
            }
        ],
        "obfuscate": bool  # If True, wrap models in TargetModel
    }
    """

    def warm_init(self, cfg: RecipeConfig) -> Dict[str, Any]:
        """Load models with optional obfuscation."""
        # HF auth
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)

        models_cfg = cfg.extra.get("models", [])
        obfuscate = cfg.extra.get("obfuscate", False)

        if not models_cfg:
            return {}

        namespace = {}

        for i, model_spec in enumerate(models_cfg):
            var_name = "model" if len(models_cfg) == 1 else f"model_{i}"
            tok_name = "tokenizer" if len(models_cfg) == 1 else f"tokenizer_{i}"

            if model_spec.get("custom_load_code"):
                if not obfuscate:
                    print(f"Loading model {i} with custom code...")
                exec(model_spec["custom_load_code"], namespace)
            else:
                model, tokenizer = self._load_standard_model(model_spec, obfuscate)

                if obfuscate:
                    # Wrap in obfuscated interface
                    namespace[var_name] = TargetModel(model, tokenizer)
                else:
                    # Expose raw model and tokenizer
                    namespace[var_name] = model
                    namespace[tok_name] = tokenizer

        return namespace

    def _load_standard_model(self, spec: Dict[str, Any], obfuscate: bool):
        """Load a model using standard transformers loading."""
        model_id = spec["name"]
        device = spec.get("device", "auto")
        dtype_str = spec.get("dtype", "auto")
        trust_remote_code = spec.get("trust_remote_code", False)
        is_peft = spec.get("is_peft", False)

        if obfuscate:
            print("Loading target model...")
        else:
            print(f"Loading {model_id}...")

        # Parse dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        dtype = dtype_map.get(dtype_str, "auto")

        if is_peft:
            from peft import PeftModel

            base_model_id = spec["base_model"]
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map=device,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            model = PeftModel.from_pretrained(base_model, model_id)
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )

        if obfuscate:
            print("✅ Model loaded")
        else:
            print(f"✅ Loaded {model_id} on {model.device}")

        return model, tokenizer
