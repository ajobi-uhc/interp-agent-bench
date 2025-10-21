"""Session managers for different execution modes.

Separates GPU mode (InterpClient) from API mode (API wrappers).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any


class SessionManager(ABC):
    """Base class for session initialization."""

    @abstractmethod
    def get_hidden_setup_code(self) -> str:
        """Get code that should be auto-executed (hidden from agent)."""
        pass

    @abstractmethod
    def get_visible_setup_instructions(self) -> str:
        """Get instructions/code that the agent should see."""
        pass


class GPUSessionManager(SessionManager):
    """Manages GPU mode sessions (Modal/Local with InterpClient)."""

    def __init__(
        self,
        experiment_name: str,
        model_name: str,
        execution_mode: str = "modal",
        model_is_peft: bool = False,
        model_base: Optional[str] = None,
        device: str = "auto",
        gpu_type: str = "A10G",
        hidden_system_prompt: str = "",
    ):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.execution_mode = execution_mode
        self.model_is_peft = model_is_peft
        self.model_base = model_base
        self.device = device
        self.gpu_type = gpu_type
        self.hidden_system_prompt = hidden_system_prompt

    def get_hidden_setup_code(self) -> str:
        """For GPU mode, hidden prompt is injected at tokenizer level, not here."""
        return ""  # No hidden code needed - happens inside InterpClient

    def get_visible_setup_instructions(self) -> str:
        """Generate InterpClient initialization code."""
        lines = []

        lines.append("# Initialize InterpClient for GPU model access")
        lines.append("from scribe.modal import InterpClient")
        lines.append("import os")
        lines.append("")

        # Hidden prompt reference (but injection happens server-side)
        if self.hidden_system_prompt:
            lines.append("# Note: Hidden configuration is loaded from environment")
            lines.append('_hidden_prompt = os.environ.get("HIDDEN_SYSTEM_PROMPT", "")')
        else:
            lines.append("_hidden_prompt = None")

        lines.append("")
        lines.append(f'print("🚀 Initializing InterpClient ({self.execution_mode} mode)...")')
        lines.append("")

        # Build InterpClient initialization
        lines.append("client = InterpClient(")
        lines.append(f'    app_name="{self.experiment_name}",')
        lines.append(f'    model_name=os.environ.get("MODEL_NAME"),')

        if self.execution_mode == "modal":
            lines.append(f'    gpu="{self.gpu_type}",')

        lines.append(f'    is_peft={self.model_is_peft},')

        if self.model_base:
            lines.append(f'    base_model=os.environ.get("MODEL_BASE"),')

        lines.append('    scaledown_window=300,')
        lines.append('    min_containers=0,')

        if self.hidden_system_prompt:
            lines.append('    hidden_system_prompt=_hidden_prompt,')

        lines.append(")")
        lines.append("")
        lines.append('print("✅ InterpClient ready!")')
        lines.append('print("   Write functions: def fn(model, tokenizer, ...)")')
        lines.append('print("   Run with: client.run(fn, ...args)")')

        return "\n".join(lines)


class APISessionManager(SessionManager):
    """Manages API mode sessions (Anthropic/OpenAI/Google with monkey-patching)."""

    def __init__(
        self,
        api_provider: str,
        hidden_system_prompt: str = "",
    ):
        self.api_provider = api_provider
        self.hidden_system_prompt = hidden_system_prompt

    def get_hidden_setup_code(self) -> str:
        """Generate monkey-patch code that gets auto-executed (hidden from agent).

        This code executes in the kernel but doesn't create a visible notebook cell.
        """
        if not self.hidden_system_prompt:
            return ""  # No hidden code if no hidden prompt

        lines = []
        lines.append("# Hidden initialization (executed in kernel, not visible in notebook)")
        lines.append("import os")
        lines.append('_hidden_prompt = os.environ.get("HIDDEN_SYSTEM_PROMPT", "")')
        lines.append("")

        if self.api_provider == "anthropic":
            lines.append("# Monkey-patch anthropic module")
            lines.append("import anthropic")
            lines.append("from scribe.api_wrapper import HiddenPromptAnthropicClient")
            lines.append("_OriginalAnthropic = anthropic.Anthropic")
            lines.append("")
            lines.append("def _wrapped_anthropic(*args, **kwargs):")
            lines.append("    _raw = _OriginalAnthropic(*args, **kwargs)")
            lines.append("    return HiddenPromptAnthropicClient(_raw, _hidden_prompt)")
            lines.append("")
            lines.append("anthropic.Anthropic = _wrapped_anthropic")

        elif self.api_provider == "openai":
            lines.append("# Monkey-patch openai module")
            lines.append("import openai")
            lines.append("from scribe.api_wrapper import HiddenPromptOpenAIClient")
            lines.append("_OriginalOpenAI = openai.OpenAI")
            lines.append("")
            lines.append("def _wrapped_openai(*args, **kwargs):")
            lines.append("    _raw = _OriginalOpenAI(*args, **kwargs)")
            lines.append("    return HiddenPromptOpenAIClient(_raw, _hidden_prompt)")
            lines.append("")
            lines.append("openai.OpenAI = _wrapped_openai")

        elif self.api_provider == "google":
            lines.append("# Monkey-patch google.generativeai module")
            lines.append("import google.generativeai as genai")
            lines.append("from scribe.api_wrapper import _GoogleModelWrapper")
            lines.append("genai.configure(api_key=os.environ['GOOGLE_API_KEY'])")
            lines.append("_OriginalGenerativeModel = genai.GenerativeModel")
            lines.append("")
            lines.append("def _wrapped_generative_model(*args, **kwargs):")
            lines.append("    _raw = _OriginalGenerativeModel(*args, **kwargs)")
            lines.append("    return _GoogleModelWrapper(_raw, _hidden_prompt)")
            lines.append("")
            lines.append("genai.GenerativeModel = _wrapped_generative_model")

        return "\n".join(lines)

    def get_visible_setup_instructions(self) -> str:
        """Generate visible instructions for API usage."""
        lines = []

        if self.hidden_system_prompt:
            lines.append("# API environment initialized")
            lines.append("# Hidden configuration has been loaded")
            lines.append(f"# (Hidden setup: {len(self.hidden_system_prompt)} chars configured)")
            lines.append("")

        if self.api_provider == "anthropic":
            lines.append("# Example usage:")
            lines.append("# import anthropic")
            lines.append("# client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])")
            lines.append("# response = client.messages.create(model='claude-3-5-sonnet-20241022', ...)")

        elif self.api_provider == "openai":
            lines.append("# Example usage:")
            lines.append("# import openai")
            lines.append("# client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])")
            lines.append("# response = client.chat.completions.create(model='gpt-4', ...)")

        elif self.api_provider == "google":
            lines.append("# Example usage:")
            lines.append("# import google.generativeai as genai")
            lines.append("# genai.configure(api_key=os.environ['GOOGLE_API_KEY'])")
            lines.append("# model = genai.GenerativeModel('gemini-pro')")
            lines.append("# response = model.generate_content('...')")

        lines.append("")
        lines.append('print("✅ API ready for use")')

        return "\n".join(lines)


def create_session_manager(
    experiment_name: str = "experiment",
    model_name: Optional[str] = None,
    api_provider: Optional[str] = None,
    execution_mode: str = "modal",
    model_is_peft: bool = False,
    model_base: Optional[str] = None,
    device: str = "auto",
    gpu_type: str = "A10G",
    hidden_system_prompt: str = "",
) -> SessionManager:
    """Factory to create appropriate session manager based on mode."""

    if api_provider:
        # API mode
        return APISessionManager(
            api_provider=api_provider,
            hidden_system_prompt=hidden_system_prompt,
        )
    elif model_name:
        # GPU mode
        return GPUSessionManager(
            experiment_name=experiment_name,
            model_name=model_name,
            execution_mode=execution_mode,
            model_is_peft=model_is_peft,
            model_base=model_base,
            device=device,
            gpu_type=gpu_type,
            hidden_system_prompt=hidden_system_prompt,
        )
    else:
        raise ValueError("Must specify either api_provider or model_name")
