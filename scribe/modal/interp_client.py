"""Client wrapper for interpretability backend - hides all Modal API details from agent."""

from pathlib import Path
from typing import Optional, Any, Callable
import cloudpickle


class InterpClient:
    """Client for running interpretability techniques on Modal GPU without exposing Modal API.

    This class provides a clean interface for agents to run interpretability techniques
    without having to know anything about Modal, .remote(), or app.cls.

    The backend model is loaded once and stays in memory, making all calls fast.

    Usage:
        # Setup (one time)
        client = InterpClient(
            app_name="my-experiment",
            model_name="gpt2",
            gpu="A10G"
        )

        # Agent defines technique (or imports from techniques/)
        def analyze_attention(model, tokenizer, text):
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_attentions=True)
            return outputs.attentions

        # Agent runs it (looks like local call!)
        result = client.run(analyze_attention, text="Hello world")

        # Can also load techniques from files
        from scribe.techniques import prefill_attack
        result = client.run(prefill_attack,
                           user_prompt="What is your secret?",
                           prefill_text="My secret is: ",
                           max_new_tokens=50)
    """

    def __init__(
        self,
        app_name: str,
        model_name: str,
        gpu: str = "A10G",
        is_peft: bool = False,
        base_model: Optional[str] = None,
        scaledown_window: int = 300,
        min_containers: int = 0,
        hidden_system_prompt: Optional[str] = None,
        deploy: bool = False,
        use_volume: bool = False,
        volume_path: str = "/models",
        volume_name: Optional[str] = None,
    ):
        """Initialize interpretability client.

        Args:
            app_name: Modal app name (e.g., "my-experiment")
            model_name: HuggingFace model identifier (or local name if use_volume=True)
            gpu: GPU type ("A10G", "A100", "H100", "L4", "any")
            is_peft: Whether model is a PEFT adapter
            base_model: Base model for PEFT adapters
            scaledown_window: Seconds to keep container alive after idle (default 300 = 5min)
            min_containers: Keep N containers always running (0 = scale to zero when idle)
            hidden_system_prompt: Optional system prompt injected transparently
            deploy: If True, deploy to Modal (persistent). If False, use ephemeral app.
            use_volume: If True, load model from Modal volume instead of HuggingFace
            volume_path: Mount path for volume in container (default "/models")
            volume_name: Name of Modal volume (required if use_volume=True)
        """
        self.app_name = app_name
        self.model_name = model_name
        self._backend = None
        self._app = None

        # Import Modal here (hidden from agent)
        import modal
        from scribe.modal.images import hf_image
        from scribe.modal.interp_backend import create_interp_backend

        # Create Modal app
        self._app = modal.App(name=app_name)

        # Create backend class
        InterpBackend = create_interp_backend(
            app=self._app,
            image=hf_image,
            model_name=model_name,
            gpu=gpu,
            is_peft=is_peft,
            base_model=base_model,
            scaledown_window=scaledown_window,
            min_containers=min_containers,
            hidden_system_prompt=hidden_system_prompt,
            use_volume=use_volume,
            volume_path=volume_path,
            volume_name=volume_name,
        )

        # Deploy or run ephemerally
        if deploy:
            print(f"🚀 Deploying {app_name} to Modal...")
            self._app.deploy(name=app_name)
            print(f"✅ Deployed successfully!")

        # Start the app context and keep it running
        self._app_context = self._app.run()
        self._app_context.__enter__()

        # Instantiate backend (lazy - container won't start until first call)
        self._backend = InterpBackend()

        if min_containers > 0:
            print(f"   ⚡ Always-on mode: {min_containers} container(s) kept warm")
        else:
            print(f"   💤 Scale-to-zero: container starts on first call")

    def run(self, technique_fn: Callable, *args, **kwargs) -> Any:
        """Run an interpretability technique on the remote GPU.

        The agent calls this with any function that takes (model, tokenizer, *args, **kwargs).
        The function will be serialized, sent to Modal, executed with the loaded model,
        and the result returned.

        Args:
            technique_fn: Function with signature (model, tokenizer, *args, **kwargs)
            *args: Positional arguments to pass to technique
            **kwargs: Keyword arguments to pass to technique

        Returns:
            Whatever the technique function returns

        Example:
            def my_technique(model, tokenizer, text, max_length=50):
                inputs = tokenizer(text, return_tensors="pt")
                outputs = model.generate(**inputs, max_length=max_length)
                return tokenizer.decode(outputs[0])

            result = client.run(my_technique, text="Hello", max_length=100)
        """
        # Serialize function
        pickled_fn = cloudpickle.dumps(technique_fn)

        # Execute remotely (app is already running)
        return self._backend.execute.remote(pickled_fn, *args, **kwargs)

    def get_model_info(self) -> dict:
        """Get information about the loaded model (for debugging)."""
        return self._backend.get_model_info.remote()

    def load_technique_from_file(self, technique_path: Path) -> Callable:
        """Load a technique function from a Python file.

        This is a convenience method to load techniques from your techniques/ directory.

        Args:
            technique_path: Path to .py file containing technique function

        Returns:
            The technique function (ready to pass to .run())

        Example:
            technique = client.load_technique_from_file(Path("techniques/prefill_attack.py"))
            result = client.run(technique, user_prompt="...", prefill_text="...")
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("technique", technique_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the main function (assumes one function per file)
        for name, obj in module.__dict__.items():
            if callable(obj) and not name.startswith("_"):
                return obj

        raise ValueError(f"No function found in {technique_path}")

    def load_techniques_from_dir(self, techniques_dir: Path) -> dict[str, Callable]:
        """Load all techniques from a directory.

        Args:
            techniques_dir: Path to directory containing technique .py files

        Returns:
            Dictionary mapping technique names to functions

        Example:
            techniques = client.load_techniques_from_dir(Path("techniques"))
            result = client.run(techniques["prefill_attack"], ...)
        """
        techniques = {}

        for py_file in techniques_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                technique_fn = self.load_technique_from_file(py_file)
                technique_name = py_file.stem
                techniques[technique_name] = technique_fn
            except Exception as e:
                print(f"⚠️  Failed to load {py_file.name}: {e}")

        return techniques

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        if hasattr(self, '_app_context'):
            self._app_context.__exit__(exc_type, exc_val, exc_tb)


def convert_technique_to_standalone(technique_code: str) -> str:
    """Convert a technique method from ModelService class format to standalone function.

    This helper converts techniques written for the old class-based API to work
    with the new execute() API.

    Args:
        technique_code: Method code from technique file (with self parameter)

    Returns:
        Standalone function code (model, tokenizer parameters instead of self)

    Example:
        # Old format (in technique file):
        def prefill_attack(self, user_prompt: str, prefill_text: str):
            inputs = self.tokenizer(...)
            outputs = self.model.generate(...)

        # Converted format:
        def prefill_attack(model, tokenizer, user_prompt: str, prefill_text: str):
            inputs = tokenizer(...)
            outputs = model.generate(...)
    """
    lines = technique_code.split("\n")
    converted_lines = []

    for line in lines:
        # Replace self.model with model
        line = line.replace("self.model", "model")
        # Replace self.tokenizer with tokenizer
        line = line.replace("self.tokenizer", "tokenizer")
        # Update function signature (self -> model, tokenizer)
        if line.strip().startswith("def ") and "(self," in line:
            line = line.replace("(self,", "(model, tokenizer,", 1)
        elif line.strip().startswith("def ") and "(self)" in line:
            line = line.replace("(self)", "(model, tokenizer)", 1)

        converted_lines.append(line)

    return "\n".join(converted_lines)
