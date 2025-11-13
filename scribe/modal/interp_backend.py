"""Generic interpretability backend for Modal - executes arbitrary functions with persistent model."""

import modal
from typing import Optional


def create_interp_backend(
    app: modal.App,
    image: modal.Image,
    model_name: str,
    gpu: str = "A10G",
    is_peft: bool = False,
    base_model: Optional[str] = None,
    scaledown_window: int = 300,
    min_containers: int = 0,
    hidden_system_prompt: Optional[str] = None,
    use_volume: bool = False,
    volume_path: str = "/models",
    volume_name: Optional[str] = None,
):
    """Create a generic interpretability backend that executes arbitrary functions.

    This creates a Modal class with a single `execute` method that accepts
    serialized Python functions and runs them with the loaded model.

    Key benefits:
    - Model loads ONCE when container starts (@modal.enter)
    - No redeployment needed for new techniques
    - Agent can pass any function dynamically
    - Model stays in memory between calls (fast!)

    Args:
        app: Modal app instance
        image: Modal image with required dependencies (must include cloudpickle)
        model_name: HuggingFace model identifier (or local name if use_volume=True)
        gpu: GPU type ("A10G", "A100", "H100", "L4", "any")
        is_peft: Whether model is a PEFT adapter
        base_model: Base model for PEFT adapters
        scaledown_window: Seconds to keep container alive after idle (default 300 = 5min)
        min_containers: Minimum number of containers to keep running (0 = scale to zero)
        hidden_system_prompt: Optional system prompt injected into all tokenizer calls
        use_volume: If True, load model from Modal volume instead of HuggingFace
        volume_path: Mount path for volume in container (default "/models")
        volume_name: Name of Modal volume (required if use_volume=True)

    Returns:
        InterpBackend class that can be instantiated

    Example:
        >>> app = modal.App("my-interp-agent")
        >>> InterpBackend = create_interp_backend(app, hf_image, "gpt2")
        >>> backend = InterpBackend()
        >>>
        >>> # Define technique function
        >>> def get_attention(model, tokenizer, text):
        ...     inputs = tokenizer(text, return_tensors="pt")
        ...     outputs = model(**inputs, output_attentions=True)
        ...     return outputs.attentions
        >>>
        >>> # Execute remotely
        >>> import cloudpickle
        >>> pickled_fn = cloudpickle.dumps(get_attention)
        >>> result = backend.execute.remote(pickled_fn, "Hello world")
    """

    # Setup volume if requested
    volumes = {}
    if use_volume:
        if volume_name is None:
            raise ValueError("volume_name is required when use_volume=True")
        volume = modal.Volume.from_name(volume_name)
        volumes = {volume_path: volume}

    @app.cls(
        gpu=gpu,
        image=image,
        secrets=[modal.Secret.from_name("huggingface-secret")],
        scaledown_window=scaledown_window,
        min_containers=min_containers,
        serialized=True,  # Allow creation from non-global scope (e.g., notebooks)
        timeout=3600,  # 1 hour timeout (3600 seconds)
        volumes=volumes,
    )
    class InterpBackend:
        """Generic interpretability backend - executes arbitrary functions with persistent model."""

        @modal.enter()
        def load_model(self):
            """Load model once when container starts (runs only once per container lifecycle)."""
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Determine model path
            if use_volume:
                # Load from volume - use simple folder name if provided
                model_path = f"{volume_path}/{model_name.split('/')[-1]}"
                print(f"🔧 Loading model from volume: {model_path}")
            else:
                # Load from HuggingFace
                model_path = model_name
                print(f"🔧 Loading model from HuggingFace: {model_path}")

            # Prepare loading kwargs
            # Use "auto" dtype to preserve quantization (MXFP4/MXFP8/etc)
            # Only force float16 for non-quantized models
            load_kwargs = {
                "device_map": "auto",
                "torch_dtype": "auto",  # Preserves quantization, falls back to model default
                "trust_remote_code": True,  # ✅ ADD THIS - Required for Kimi K2
            }

            if is_peft:
                from peft import PeftModel

                if base_model is None:
                    raise ValueError("base_model required when is_peft=True")

                print(f"🔧 Loading base model: {base_model}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    **load_kwargs,
                )

                print(f"🔧 Loading PEFT adapter: {model_path}")
                self.model = PeftModel.from_pretrained(self.model, model_path)
                tokenizer_name = base_model
            else:
                print(f"🔧 Loading model: {model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **load_kwargs,
                )
                tokenizer_name = model_path

            print(f"🔧 Loading tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )

            # Wrap tokenizer with hidden prompt if specified
            if hidden_system_prompt:
                from scribe.core import HiddenPromptTokenizer
                print(f"🔒 Wrapping tokenizer with hidden system prompt")
                self.tokenizer = HiddenPromptTokenizer(
                    self.tokenizer, hidden_system_prompt
                )

            print(f"✅ Model loaded on {self.model.device}")

        @modal.method()
        def execute(self, pickled_fn: bytes, *args, **kwargs):
            """Execute arbitrary function with the loaded model.

            The function will receive (model, tokenizer, *args, **kwargs).

            Args:
                pickled_fn: cloudpickle-serialized function
                *args: Positional arguments to pass to function
                **kwargs: Keyword arguments to pass to function

            Returns:
                Whatever the function returns (with BFloat16 tensors auto-converted to float32)

            Example function signature:
                def my_technique(model, tokenizer, text, max_length=50):
                    # Your interpretability code here
                    return result
            """
            import cloudpickle

            # Clean up any hooks from previous function calls
            # This prevents hooks from one call interfering with the next
            self._cleanup_hooks()

            # Deserialize function
            fn = cloudpickle.loads(pickled_fn)

            # Execute with loaded model and tokenizer
            try:
                result = fn(self.model, self.tokenizer, *args, **kwargs)
                # Convert BFloat16 tensors to float32 for serialization
                return self._convert_bfloat16(result)
            finally:
                # Clean up hooks after execution (even if function errors)
                self._cleanup_hooks()

        def _convert_bfloat16(self, obj):
            """Recursively convert BFloat16 tensors to float32 for serialization.

            BFloat16 tensors cannot be serialized to numpy, so we convert them
            to float32 automatically. This handles nested structures (lists, dicts, tuples).
            """
            import torch
            import numpy as np

            if isinstance(obj, torch.Tensor):
                if obj.dtype == torch.bfloat16:
                    return obj.float()
                return obj
            elif isinstance(obj, np.ndarray):
                # Check if it's trying to be bfloat16 (will have failed, but handle edge case)
                return obj
            elif isinstance(obj, dict):
                return {k: self._convert_bfloat16(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._convert_bfloat16(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(self._convert_bfloat16(item) for item in obj)
            else:
                return obj

        def _cleanup_hooks(self):
            """Remove all forward and backward hooks from the model."""
            def remove_hooks_recursive(module):
                # Clear forward hooks
                module._forward_hooks.clear()
                module._forward_pre_hooks.clear()
                # Clear backward hooks
                module._backward_hooks.clear()
                module._backward_pre_hooks.clear()
                # Recurse to all child modules
                for child in module.children():
                    remove_hooks_recursive(child)

            remove_hooks_recursive(self.model)

        @modal.method()
        def get_model_info(self):
            """Get basic model information (for debugging)."""
            return {
                "model_name": model_name,
                "device": str(self.model.device),
                "dtype": str(self.model.dtype),
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
            }

    return InterpBackend
