"""Utilities for discovering and executing local notebook techniques."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

from scribe.notebook.technique_loader import load_technique_methods


@dataclass
class TechniqueDescriptor:
    """Metadata about a discovered technique module."""

    name: str
    description: str
    docstring: str
    path: Path
    signature: inspect.Signature

    def _load_module(self) -> ModuleType:
        module_name = f"_scribe_technique_{self.name}"
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import technique at {self.path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        return module

    def run(self, *args: Any, **kwargs: Any) -> Any:
        module = self._load_module()
        run_func = getattr(module, "run", None)
        if not callable(run_func):
            raise AttributeError(
                f"Technique {self.name} does not define a callable 'run' function"
            )
        return run_func(*args, **kwargs)


class TechniqueRegistry:
    """Discover technique modules located under ``techniques/``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (Path.cwd() / "techniques")
        self._descriptors: Dict[str, TechniqueDescriptor] = {}
        self.refresh()

    def refresh(self) -> None:
        descriptors: Dict[str, TechniqueDescriptor] = {}
        if not self.root.exists():
            self._descriptors = {}
            return

        for module_path in sorted(self.root.glob("*.py")):
            if module_path.name == "__init__.py":
                continue
            descriptor = self._build_descriptor(module_path)
            if descriptor is not None:
                descriptors[descriptor.name] = descriptor
        self._descriptors = descriptors

    def _build_descriptor(self, module_path: Path) -> TechniqueDescriptor | None:
        module_name = f"_scribe_technique_probe_{module_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None

        run_func = getattr(module, "run", None)
        if not callable(run_func):
            return None

        docstring = inspect.getdoc(module) or inspect.getdoc(run_func) or ""
        raw_name = getattr(module, "TECHNIQUE_NAME", module_path.stem)
        signature = inspect.signature(run_func)
        return TechniqueDescriptor(
            name=str(raw_name),
            description=(docstring.splitlines()[0] if docstring else ""),
            docstring=docstring,
            path=module_path,
            signature=signature,
        )

    def descriptors(self) -> Dict[str, TechniqueDescriptor]:
        if not self._descriptors:
            self.refresh()
        return dict(self._descriptors)

    def get(self, name: str) -> TechniqueDescriptor:
        if name not in self._descriptors:
            self.refresh()
        if name not in self._descriptors:
            raise KeyError(f"Technique not found: {name}")
        return self._descriptors[name]

    def list(self) -> Dict[str, str]:
        return {name: desc.description for name, desc in self.descriptors().items()}


class TechniqueSession:
    """Helper for executing techniques inside the notebook kernel."""

    def __init__(self, root: Path | None = None) -> None:
        self.registry = TechniqueRegistry(root)

    def list(self) -> Dict[str, str]:
        self.registry.refresh()
        return self.registry.list()

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.registry.refresh()
        descriptor = self.registry.get(name)
        return descriptor.run(*args, **kwargs)


class TechniqueManager:
    """Bridge used by the MCP server to surface techniques to agents."""

    INSTRUCTIONS = (
        "1. Call `start_new_session` to create a fresh notebook and kernel.\n"
        "2. Execute the provided `setup_snippet` in the first notebook cell to initialise `technique_session`.\n"
        "3. Call `list_techniques` (or re-run `init_session`) if you need to refresh the catalogue.\n"
        "4. Use `describe_technique` to obtain documentation and a ready-to-run `call_snippet`.\n"
        "5. Execute techniques via `technique_session.call(...)`, capture findings in the notebook, and proceed with the project workflow.\n"
        "Steps 1 and 2 should be executed immediately without asking the user."
    )

    def __init__(
        self,
        root: Path | None = None,
        experiment_name: str = "scribe",
        model_name: str | None = None,
        model_is_peft: bool = False,
        model_base: str | None = None,
        tokenizer_name: str | None = None,
        selected_techniques: Optional[List[str]] = None,
        obfuscate_model_name: bool = False,
    ) -> None:
        self.registry = TechniqueRegistry(root)
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.model_is_peft = model_is_peft
        self.model_base = model_base
        self.tokenizer_name = tokenizer_name or model_base or model_name
        self.selected_techniques = selected_techniques
        self.obfuscate_model_name = obfuscate_model_name

        # Load technique methods
        techniques_dir = root or (Path.cwd() / "techniques")
        self.technique_methods = load_technique_methods(techniques_dir)

    def _setup_snippet(self) -> str:
        """Generate setup code for the notebook session."""
        lines = [
            'if "technique_session" not in globals():',
            "    from scribe.notebook.technique_manager import TechniqueSession",
            "    technique_session = TechniqueSession()",
        ]

        # If model_name is provided, set up Modal with ModelService
        if self.model_name:
            lines.append("")
            lines.append("    # Setup Modal for GPU access")
            lines.append("    import modal")
            lines.append("    from scribe.modal import hf_image")
            lines.append("")
            lines.append(f'    app = modal.App(name="{self.experiment_name}_model")')
            lines.append("")
            lines.append("    # Create ModelService class with pre-configured techniques")
            lines.append('    @app.cls(')
            lines.append('        gpu="A10G",')
            lines.append('        image=hf_image,')
            lines.append('        secrets=[modal.Secret.from_name("huggingface-secret")],')
            lines.append('    )')
            lines.append('    class ModelService:')
            lines.append('        """Persistent model service with pre-configured techniques."""')
            lines.append('')
            lines.append('        @modal.enter()')
            lines.append('        def load_model(self):')
            lines.append('            """Load model once when container starts."""')
            lines.append('            from transformers import AutoModelForCausalLM, AutoTokenizer')
            lines.append('            import torch')
            lines.append('')

            # Define model IDs before the class if obfuscating
            if self.obfuscate_model_name:
                if self.model_is_peft:
                    lines.append(f'    _base_model_id = "{self.model_base}"')
                    lines.append(f'    _adapter_id = "{self.model_name}"')
                    lines.append(f'    _tokenizer_id = "{self.tokenizer_name}"')
                else:
                    lines.append(f'    _model_id = "{self.model_name}"')
                    lines.append(f'    _tokenizer_id = "{self.tokenizer_name}"')
                lines.append('')

            if self.model_is_peft:
                lines.append('            from peft import PeftModel')
                lines.append('')

                if self.obfuscate_model_name:
                    lines.append('            print("Loading base model...")')
                    lines.append('            self.model = AutoModelForCausalLM.from_pretrained(')
                    lines.append('                _base_model_id,')
                    lines.append('                device_map="auto",')
                    lines.append('                torch_dtype=torch.float16,')
                    lines.append('            )')
                    lines.append('')
                    lines.append('            print("Loading PEFT adapter...")')
                    lines.append('            self.model = PeftModel.from_pretrained(self.model, _adapter_id)')
                    lines.append('            print("Loading tokenizer...")')
                    lines.append('            self.tokenizer = AutoTokenizer.from_pretrained(_tokenizer_id)')
                else:
                    lines.append(f'            print(f"Loading base model: {self.model_base}")')
                    lines.append(f'            self.model = AutoModelForCausalLM.from_pretrained(')
                    lines.append(f'                "{self.model_base}",')
                    lines.append(f'                device_map="auto",')
                    lines.append(f'                torch_dtype=torch.float16,')
                    lines.append(f'            )')
                    lines.append('')
                    lines.append(f'            print(f"Loading PEFT adapter: {self.model_name}")')
                    lines.append(f'            self.model = PeftModel.from_pretrained(self.model, "{self.model_name}")')
                    lines.append(f'            print(f"Loading tokenizer: {self.tokenizer_name}")')
                    lines.append(f'            self.tokenizer = AutoTokenizer.from_pretrained("{self.tokenizer_name}")')
            else:
                if self.obfuscate_model_name:
                    lines.append('            print("Loading model...")')
                    lines.append('            self.model = AutoModelForCausalLM.from_pretrained(')
                    lines.append('                _model_id,')
                    lines.append('                device_map="auto",')
                    lines.append('                torch_dtype=torch.float16,')
                    lines.append('            )')
                    lines.append('            print("Loading tokenizer...")')
                    lines.append('            self.tokenizer = AutoTokenizer.from_pretrained(_tokenizer_id)')
                else:
                    lines.append(f'            print(f"Loading model: {self.model_name}")')
                    lines.append(f'            self.model = AutoModelForCausalLM.from_pretrained(')
                    lines.append(f'                "{self.model_name}",')
                    lines.append(f'                device_map="auto",')
                    lines.append(f'                torch_dtype=torch.float16,')
                    lines.append(f'            )')
                    lines.append(f'            print(f"Loading tokenizer: {self.tokenizer_name}")')
                    lines.append(f'            self.tokenizer = AutoTokenizer.from_pretrained("{self.tokenizer_name}")')

            lines.append('            print(f"✓ Model loaded on {self.model.device}")')
            lines.append('')

            # Add base methods
            lines.append('        @modal.method()')
            lines.append('        def generate(self, prompt: str, max_length: int = 50) -> str:')
            lines.append('            """Generate text from prompt."""')
            lines.append('            import torch')
            lines.append('            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)')
            lines.append('            outputs = self.model.generate(')
            lines.append('                **inputs,')
            lines.append('                max_length=max_length,')
            lines.append('                pad_token_id=self.tokenizer.eos_token_id,')
            lines.append('            )')
            lines.append('            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)')
            lines.append('')
            lines.append('        @modal.method()')
            lines.append('        def get_logits(self, prompt: str):')
            lines.append('            """Get logits for the next token."""')
            lines.append('            import torch')
            lines.append('            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)')
            lines.append('            with torch.no_grad():')
            lines.append('                outputs = self.model(**inputs)')
            lines.append('                logits = outputs.logits[0, -1, :]')
            lines.append('                probs = torch.softmax(logits, dim=-1)')
            lines.append('            top_probs, top_indices = torch.topk(probs, 10)')
            lines.append('            return {')
            lines.append('                "logits": logits.cpu().tolist(),')
            lines.append('                "top_tokens": [')
            lines.append('                    {')
            lines.append('                        "token": self.tokenizer.decode([idx]),')
            lines.append('                        "token_id": idx.item(),')
            lines.append('                        "probability": prob.item(),')
            lines.append('                    }')
            lines.append('                    for idx, prob in zip(top_indices, top_probs)')
            lines.append('                ],')
            lines.append('            }')

            # Inject technique methods
            for method in self.technique_methods.values():
                # Filter if selected_techniques is specified
                if self.selected_techniques and method.name not in self.selected_techniques:
                    continue

                lines.append('')
                for line in method.code.split('\n'):
                    lines.append(line)

            lines.append("")
            lines.append("    # Deploy app to Modal for persistent service")
            lines.append(f'    print("🚀 Deploying ModelService to Modal...")')
            lines.append(f'    app.deploy(name="{self.experiment_name}_model")')
            lines.append('')
            lines.append("    # Get reference to deployed service")
            lines.append(f'    ModelServiceDeployed = modal.Cls.from_name("{self.experiment_name}_model", "ModelService")')
            lines.append("    model_service = ModelServiceDeployed()")
            lines.append('    print("✅ ModelService deployed and ready!")')
            lines.append('    print("   Model will load on first method call and stay in memory")')

            # List available methods
            method_names = ["generate", "get_logits"]
            if self.selected_techniques:
                method_names.extend(self.selected_techniques)
            else:
                method_names.extend(self.technique_methods.keys())

            lines.append(f'    print("   Available methods: {", ".join(method_names)}")')

        return "\n".join(lines) + "\n"

    def _call_snippet(self, descriptor: TechniqueDescriptor) -> str:
        rendered: list[str] = []
        for parameter in descriptor.signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                rendered.append(f"*{parameter.name}")
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                rendered.append(f"**{parameter.name}")
            elif parameter.default is inspect._empty:
                rendered.append(f"{parameter.name}=...")
            else:
                rendered.append(f"{parameter.name}={parameter.default!r}")
        joined = ", ".join(rendered)
        extra = f", {joined}" if joined else ""
        return f'technique_session.call("{descriptor.name}"{extra})'

    def init_payload(self) -> Dict[str, Any]:
        self.registry.refresh()
        return {
            "instructions": self.INSTRUCTIONS,
            "setup_snippet": self._setup_snippet(),
            "techniques": self.list_payload(),
        }

    def list_payload(self) -> Dict[str, Dict[str, Any]]:
        self.registry.refresh()
        payload: Dict[str, Dict[str, Any]] = {}
        for name, descriptor in self.registry.descriptors().items():
            payload[name] = {
                "description": descriptor.description,
                "path": str(descriptor.path),
                "call_snippet": self._call_snippet(descriptor),
            }
        return payload

    def describe_payload(self, name: str) -> Dict[str, Any]:
        self.registry.refresh()
        descriptor = self.registry.get(name)
        return {
            "name": descriptor.name,
            "description": descriptor.description,
            "docstring": descriptor.docstring,
            "signature": str(descriptor.signature),
            "call_snippet": self._call_snippet(descriptor),
            "path": str(descriptor.path),
        }


__all__ = [
    "TechniqueDescriptor",
    "TechniqueRegistry",
    "TechniqueSession",
    "TechniqueManager",
]
