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
        execution_mode: str = "modal",
        device: str = "auto",
        hidden_system_prompt: str = "",
    ) -> None:
        self.registry = TechniqueRegistry(root)
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.model_is_peft = model_is_peft
        self.model_base = model_base
        self.tokenizer_name = tokenizer_name or model_base or model_name
        self.selected_techniques = selected_techniques
        self.obfuscate_model_name = obfuscate_model_name
        self.execution_mode = execution_mode
        self.device = device
        self.hidden_system_prompt = hidden_system_prompt

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

        # If model_name is provided, set up model service (Modal or Local)
        if self.model_name:
            lines.append("")

            if self.execution_mode == "local":
                # Local execution mode
                lines.append("    # Initialize local ModelService")
                lines.append("    from scribe.local import LocalModelService")
                lines.append("    from pathlib import Path")
                lines.append("    import os")
                lines.append("")
                # Read and immediately delete hidden prompt from environment (if present)
                if self.hidden_system_prompt:
                    lines.append("    # Read hidden configuration from environment (auto-deleted after reading)")
                    lines.append('    _hidden_prompt = os.environ.pop("HIDDEN_SYSTEM_PROMPT", "")')
                    lines.append("")
                lines.append(f'    print("🔧 Initializing local model service...")')
                lines.append("    model_service = LocalModelService(")
                lines.append(f'        model_name="{self.model_name}",')
                lines.append(f'        device="{self.device}",')
                lines.append(f'        is_peft={self.model_is_peft},')
                if self.model_base:
                    lines.append(f'        base_model="{self.model_base}",')
                if self.tokenizer_name:
                    lines.append(f'        tokenizer_name="{self.tokenizer_name}",')
                lines.append('        techniques_dir=Path.cwd() / "techniques",')
                if self.selected_techniques:
                    techniques_list = '", "'.join(self.selected_techniques)
                    lines.append(f'        selected_techniques=["{techniques_list}"],')
                else:
                    lines.append('        selected_techniques=None,')
                lines.append(f'        obfuscate_model_name={self.obfuscate_model_name},')
                # Pass hidden prompt from variable (deleted from environment after reading)
                if self.hidden_system_prompt:
                    lines.append('        hidden_system_prompt=_hidden_prompt,')
                lines.append("    )")
                # Clean up the temporary variable
                if self.hidden_system_prompt:
                    lines.append('    del _hidden_prompt  # Clean up')
                lines.append('    print("✅ Local model service ready!")')
            else:
                # Modal execution mode (existing behavior)
                lines.append("    # Connect to pre-deployed ModelService")
                lines.append("    import modal")
                lines.append(f'    print("🔗 Connecting to ModelService...")')
                lines.append(f'    model_service = modal.Cls.from_name("{self.experiment_name}_model", "ModelService")()')
                lines.append('    print("✅ Connected to ModelService!")')
                lines.append('    print("   Model is loaded and ready on GPU")')

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
