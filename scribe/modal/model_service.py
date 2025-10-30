"""Modal ModelService - Persistent GPU model service for fast inference."""

import modal
from typing import Optional


def create_model_service_class(
    app: modal.App,
    image: modal.Image,
    model_name: str,
    gpu: str = "A10G",
    is_peft: bool = False,
    base_model: Optional[str] = None,
):
    """Create a Modal ModelService class with persistent model loading.

    This creates a class with @modal.cls decorator that loads the model once
    and keeps it in memory for fast subsequent calls.

    Args:
        app: Modal app instance
        image: Modal image with required dependencies
        model_name: HuggingFace model identifier
        gpu: GPU type ("A10G", "A100", "H100", "L4", "any")
        is_peft: Whether model is a PEFT adapter
        base_model: Base model for PEFT adapters

    Returns:
        ModelService class that can be instantiated

    Example:
        >>> app = modal.App("my-experiment")
        >>> ModelService = create_model_service_class(app, hf_image, "gpt2")
        >>> model_service = ModelService()
        >>> result = model_service.generate.remote("Hello", max_length=50)
    """

    @app.cls(
        gpu=gpu,
        image=image,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    class ModelService:
        """Persistent model service - model loads once and stays in memory."""

        @modal.enter()
        def load_model(self):
            """Load model once when container starts (runs only once)."""
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            if is_peft:
                from peft import PeftModel

                if base_model is None:
                    raise ValueError("base_model required when is_peft=True")

                print(f"Loading base model: {base_model}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )

                print(f"Loading PEFT adapter: {model_name}")
                self.model = PeftModel.from_pretrained(self.model, model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            else:
                print(f"Loading model: {model_name}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            print(f"✓ Model loaded on {self.model.device}")

        @modal.method()
        def generate(self, prompt: str, max_length: int = 50) -> str:
            """Generate text from prompt (fast - model already loaded).

            Args:
                prompt: Text prompt
                max_length: Maximum output length

            Returns:
                Generated text
            """
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        @modal.method()
        def get_logits(self, prompt: str):
            """Get logits for the next token.

            Args:
                prompt: Text prompt

            Returns:
                Dictionary with logits and token probabilities
            """
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                probs = torch.softmax(logits, dim=-1)

            # Get top 10 tokens
            top_probs, top_indices = torch.topk(probs, 10)

            return {
                "logits": logits.cpu().tolist(),
                "top_tokens": [
                    {
                        "token": self.tokenizer.decode([idx]),
                        "token_id": idx.item(),
                        "probability": prob.item(),
                    }
                    for idx, prob in zip(top_indices, top_probs)
                ],
            }

    return ModelService