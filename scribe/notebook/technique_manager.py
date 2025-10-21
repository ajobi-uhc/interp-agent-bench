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
from scribe.notebook.session_manager import create_session_manager, SessionManager


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
        "2. Execute the provided `setup_snippet` in the first notebook cell to initialize `client`.\n"
        "3. Write technique functions that take (model, tokenizer, *args, **kwargs) as parameters.\n"
        "4. Run techniques with `client.run(your_function, ...args)`.\n"
        "5. Capture findings in the notebook and proceed with the project workflow.\n"
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
        execution_mode: str = "modal",
        device: str = "auto",
        gpu_type: str = "A10G",
        hidden_system_prompt: str = "",
        api_provider: str | None = None,
    ) -> None:
        self.registry = TechniqueRegistry(root)
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.model_is_peft = model_is_peft
        self.model_base = model_base
        self.tokenizer_name = tokenizer_name or model_base or model_name
        self.selected_techniques = selected_techniques
        self.execution_mode = execution_mode
        self.device = device
        self.gpu_type = gpu_type
        self.hidden_system_prompt = hidden_system_prompt
        self.api_provider = api_provider

        # Load technique methods
        techniques_dir = root or (Path.cwd() / "techniques")
        self.technique_methods = load_technique_methods(techniques_dir)

        # Create session manager for this mode
        self.session_manager: SessionManager = create_session_manager(
            experiment_name=experiment_name,
            model_name=model_name,
            api_provider=api_provider,
            execution_mode=execution_mode,
            model_is_peft=model_is_peft,
            model_base=model_base,
            device=device,
            gpu_type=gpu_type,
            hidden_system_prompt=hidden_system_prompt,
        )


    def get_hidden_setup_code(self) -> str:
        """Get code that should be auto-executed (hidden from agent)."""
        return self.session_manager.get_hidden_setup_code()

    def _setup_snippet(self) -> str:
        """Generate setup code that the agent sees."""
        return self.session_manager.get_visible_setup_instructions()

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
