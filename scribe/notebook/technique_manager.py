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

        # Pre-initialize client if obfuscation is enabled (lazy initialization)
        self._pre_initialized_client = None
        self._client_initialized = False

    def _pre_initialize_client(self):
        """Create InterpClient before agent starts (for obfuscation)."""
        import sys

        print(f"🔒 Pre-initializing InterpClient (model details will be hidden from agent)...", file=sys.stderr)
        print(f"   Model: {self.model_name}", file=sys.stderr)
        if self.model_is_peft:
            print(f"   Base model: {self.model_base}", file=sys.stderr)
        print(f"   Execution mode: {self.execution_mode}", file=sys.stderr)

        try:
            from scribe.modal import InterpClient

            print("   Creating InterpClient...", file=sys.stderr)
            # Create client with full model configuration
            self._pre_initialized_client = InterpClient(
                app_name=self.experiment_name,
                model_name=self.model_name,
                gpu="A10G" if self.execution_mode == "modal" else None,
                is_peft=self.model_is_peft,
                base_model=self.model_base,
                scaledown_window=300,
                min_containers=0,
                hidden_system_prompt=self.hidden_system_prompt if self.hidden_system_prompt else None,
            )
            print("   ✅ InterpClient created", file=sys.stderr)

            # Trigger model loading now (warmup)
            print("   Warming up model on Modal GPU (this may take 30-60 seconds)...", file=sys.stderr)
            try:
                info = self._pre_initialized_client.get_model_info()
                print(f"   ✅ Model loaded: {info.get('num_parameters', 'unknown')} parameters", file=sys.stderr)
            except Exception as e:
                print(f"   ⚠️  Warmup failed (will load on first use): {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)

            print("   Client ready - agent will receive pre-configured client object", file=sys.stderr)
        except Exception as e:
            print(f"   ❌ Failed to initialize InterpClient: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

    def _setup_snippet(self) -> str:
        """Generate setup code for the notebook session using InterpClient."""
        import sys

        lines = []

        # If model_name is provided, set up InterpClient
        if self.model_name:
            # Lazy initialization: only create client when setup snippet is first requested
            if self.obfuscate_model_name and not self._client_initialized:
                print("[MCP] Obfuscation enabled - initializing client before agent starts...", file=sys.stderr)
                self._pre_initialize_client()
                self._client_initialized = True

            if self.obfuscate_model_name and self._pre_initialized_client:
                print("[MCP] Using pre-initialized client (obfuscation mode)...", file=sys.stderr)
                # Instead of pickling the Modal client (which has async objects),
                # we inject code that references a client stored in MCP server memory
                lines.append("# Access pre-configured InterpClient (model details hidden)")
                lines.append("import os")
                lines.append("")

                # Store reference to client in an environment variable that kernel can access
                # We'll use a global variable approach instead
                lines.append("# Get client from MCP server environment")
                lines.append("# Note: client initialization is handled by the MCP server")
                lines.append('print("⚠️  Obfuscation mode active: client must be accessed via alternative method")')
                lines.append('print("   Model details are hidden for this experiment")')
                lines.append("")

                # For now, fall back to standard initialization but with hidden params
                lines.append("# Fallback: Initialize with hidden parameters")
                lines.append("from scribe.modal import InterpClient")
                lines.append("")
                lines.append("# Read configuration from environment (model name hidden)")
                lines.append('_hidden_config = os.environ.get("HIDDEN_SYSTEM_PROMPT", "")')
                lines.append("")
                lines.append('print("🚀 Initializing InterpClient (obfuscated mode)...")')
                lines.append("")

                # Create client without exposing model name
                lines.append("# Client with hidden model configuration")
                lines.append("# Model details are read from environment variables by MCP server")
                lines.append(f'client = InterpClient(')
                lines.append(f'    app_name="{self.experiment_name}",')
                lines.append(f'    model_name=os.environ.get("MODEL_NAME", ""),')
                lines.append('    gpu="A10G" if os.environ.get("EXECUTION_MODE") == "modal" else None,')
                lines.append(f'    is_peft={self.model_is_peft},')
                if self.model_base:
                    lines.append(f'    base_model=os.environ.get("MODEL_BASE", ""),')
                lines.append('    scaledown_window=300,')
                lines.append('    min_containers=0,')
                if self.hidden_system_prompt:
                    lines.append('    hidden_system_prompt=_hidden_config if _hidden_config else None,')
                lines.append(")")
                lines.append("")
                lines.append('print("✅ InterpClient ready!")')
                lines.append('print("   Write functions with signature: def fn(model, tokenizer, ...)")')
                lines.append('print("   Run with: client.run(fn, ...args)")')

            else:
                # Standard initialization (model name visible to agent)
                lines.append("# Initialize InterpClient at module level (Modal requires global scope)")
                lines.append("from scribe.modal import InterpClient")
                lines.append("import os")
                lines.append("")

                # Read hidden prompt from environment if present
                if self.hidden_system_prompt:
                    lines.append("# Read hidden configuration from environment")
                    lines.append('_hidden_prompt = os.environ.get("HIDDEN_SYSTEM_PROMPT", "")')
                else:
                    lines.append("_hidden_prompt = None")

                lines.append("")
                lines.append(f'print("🚀 Initializing InterpClient ({self.execution_mode} mode)...")')
                lines.append("")

                # Build InterpClient initialization at global scope (not in if block)
                lines.append("# Create client at global scope (required by Modal)")
                lines.append("client = InterpClient(")
                lines.append(f'    app_name="{self.experiment_name}",')
                lines.append(f'    model_name="{self.model_name}",')

                if self.execution_mode == "modal":
                    lines.append('    gpu="A10G",')

                lines.append(f'    is_peft={self.model_is_peft},')

                if self.model_base:
                    lines.append(f'    base_model="{self.model_base}",')

                lines.append('    scaledown_window=300,')
                lines.append('    min_containers=0,')

                if self.hidden_system_prompt:
                    lines.append('    hidden_system_prompt=_hidden_prompt,')

                lines.append(")")
                lines.append("")
                lines.append('print("✅ InterpClient ready!")')
                lines.append('print("   Write functions with signature: def fn(model, tokenizer, ...)")')
                lines.append('print("   Run with: client.run(fn, ...args)")')
                lines.append('print("   Model will load on first call and stay in memory.")')

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
