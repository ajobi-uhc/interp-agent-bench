"""Local model service - runs models on local hardware instead of Modal GPU."""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalModelService:
    """Local model service that mimics Modal's ModelService interface.

    This class provides the same interface as Modal's remote ModelService,
    but executes everything locally on MPS/CUDA/CPU.

    Usage:
        service = LocalModelService(
            model_name="gpt2",
            device="mps",
            techniques_dir=Path("techniques"),
            selected_techniques=["prefill_attack", "get_model_info"]
        )

        # Use just like Modal service (but without .remote())
        result = service.prefill_attack(
            user_prompt="Hello",
            prefill_text="Hi",
            max_new_tokens=50
        )
    """

    def __init__(
        self,
        model_name: str,
        device: str = "mps",
        is_peft: bool = False,
        base_model: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        techniques_dir: Optional[Path] = None,
        selected_techniques: Optional[list] = None,
        obfuscate_model_name: bool = False,
        hidden_system_prompt: Optional[str] = None,
    ):
        """Initialize local model service.

        Args:
            model_name: HuggingFace model identifier or PEFT adapter
            device: Device to use ("mps", "cuda", "cpu", or "auto")
            is_peft: Whether model is a PEFT adapter
            base_model: Base model for PEFT adapters
            tokenizer_name: Tokenizer to use (defaults to model_name or base_model)
            techniques_dir: Directory containing technique files
            selected_techniques: List of technique names to load (None = all)
            obfuscate_model_name: Hide model name in get_model_info
            hidden_system_prompt: Optional system prompt injected into all inputs (hidden from agent)
        """
        self.model_name = model_name
        self.is_peft = is_peft
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name or base_model or model_name
        self.obfuscate_model_name = obfuscate_model_name
        self.hidden_system_prompt = hidden_system_prompt

        # Determine device
        self.device = self._resolve_device(device)

        # Load model and tokenizer
        self._load_model()

        # Wrap tokenizer with hidden prompt if specified
        if hidden_system_prompt:
            self._wrap_tokenizer_with_hidden_prompt()

        # Inject technique methods
        if techniques_dir:
            self._inject_techniques(techniques_dir, selected_techniques)

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual available device."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"

        # Validate requested device
        if device == "mps" and not torch.backends.mps.is_available():
            print(f"⚠️  MPS not available, falling back to CPU")
            return "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print(f"⚠️  CUDA not available, falling back to CPU")
            return "cpu"

        return device

    def _load_model(self):
        """Load model and tokenizer once (equivalent to Modal's @modal.enter())."""
        print(f"🔧 Loading model locally on device: {self.device}")

        if self.is_peft:
            from peft import PeftModel

            print(f"   Loading base model: {self.base_model}")
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                dtype=dtype,
            )
            self.model = self.model.to(self.device)

            print(f"   Loading PEFT adapter: {self.model_name}")
            self.model = PeftModel.from_pretrained(self.model, self.model_name)
        else:
            print(f"   Loading model: {self.model_name}")
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
            )
            self.model = self.model.to(self.device)

        print(f"   Loading tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        print(f"✅ Model loaded successfully on {self.device}")

    def _wrap_tokenizer_with_hidden_prompt(self):
        """Wrap tokenizer with HiddenPromptTokenizer to inject system prompt transparently."""
        from scribe.core import HiddenPromptTokenizer

        print(f"🔒 Wrapping tokenizer with hidden system prompt")
        self.tokenizer = HiddenPromptTokenizer(self.tokenizer, self.hidden_system_prompt)
        print(f"   Hidden prompt will be injected into all tokenization calls")

    def _inject_techniques(self, techniques_dir: Path, selected_techniques: Optional[list] = None):
        """Dynamically inject technique methods into this instance.

        Args:
            techniques_dir: Path to techniques directory
            selected_techniques: List of technique names to load (None = all)
        """
        from scribe.notebook.technique_loader import load_technique_methods
        import os

        # Load technique methods
        all_techniques = load_technique_methods(techniques_dir)

        # Filter to selected techniques
        if selected_techniques:
            techniques = {name: method for name, method in all_techniques.items()
                         if name in selected_techniques}
        else:
            techniques = all_techniques

        # Set environment variable for get_model_info obfuscation
        if self.obfuscate_model_name:
            os.environ["OBFUSCATE_MODEL_NAME"] = "true"

        # Inject each technique as a bound method
        for name, technique_method in techniques.items():
            # Extract the function code and execute it to get the function object
            # Remove @modal.method() decorator and indentation
            code_lines = technique_method.code.split('\n')

            # Skip decorator lines and adjust indentation
            cleaned_lines = []
            for line in code_lines:
                stripped = line.strip()
                if stripped.startswith('@modal.method'):
                    continue
                # Remove leading spaces (technique methods have 4-space indent)
                if line.startswith('    '):
                    cleaned_lines.append(line[4:])
                else:
                    cleaned_lines.append(line)

            cleaned_code = '\n'.join(cleaned_lines)

            # Execute the function definition to create the function object
            namespace = {
                'torch': torch,
                'AutoModelForCausalLM': AutoModelForCausalLM,
                'AutoTokenizer': AutoTokenizer,
            }
            exec(cleaned_code, namespace)

            # Get the function object (it has the same name as the technique)
            func = namespace.get(name)
            if func:
                # Bind the function to this instance
                bound_method = func.__get__(self, LocalModelService)
                setattr(self, name, bound_method)

    # Base methods (equivalent to Modal's base ModelService methods)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text from prompt.

        Args:
            prompt: Text prompt
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        """Get logits for the next token.

        Args:
            prompt: Text prompt

        Returns:
            Dictionary with logits and token probabilities
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
