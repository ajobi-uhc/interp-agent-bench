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
        model_name: HuggingFace model identifier
        gpu: GPU type ("A10G", "A100", "H100", "L4", "any")
        is_peft: Whether model is a PEFT adapter
        base_model: Base model for PEFT adapters
        scaledown_window: Seconds to keep container alive after idle (default 300 = 5min)
        min_containers: Minimum number of containers to keep running (0 = scale to zero)
        hidden_system_prompt: Optional system prompt injected into all tokenizer calls

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

    @app.cls(
        gpu=gpu,
        image=image,
        secrets=[modal.Secret.from_name("huggingface-secret")],
        scaledown_window=scaledown_window,
        min_containers=min_containers,
        serialized=True,  # Allow creation from non-global scope (e.g., notebooks)
        timeout=3600,  # 1 hour timeout (3600 seconds)
    )
    class InterpBackend:
        """Generic interpretability backend - executes arbitrary functions with persistent model."""

        @modal.enter()
        def load_model(self):
            """Load model once when container starts (runs only once per container lifecycle)."""
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

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

                print(f"🔧 Loading PEFT adapter: {model_name}")
                self.model = PeftModel.from_pretrained(self.model, model_name)
                tokenizer_name = base_model
            else:
                print(f"🔧 Loading model: {model_name}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,                    
                    **load_kwargs,
                )
                tokenizer_name = model_name

            print(f"🔧 Loading tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
                Whatever the function returns

            Example function signature:
                def my_technique(model, tokenizer, text, max_length=50):
                    # Your interpretability code here
                    return result
            """
            import cloudpickle

            # Deserialize function
            fn = cloudpickle.loads(pickled_fn)

            # Execute with loaded model and tokenizer
            return fn(self.model, self.tokenizer, *args, **kwargs)

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
