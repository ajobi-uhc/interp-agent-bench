"""Build Modal images from configuration."""

from typing import List
import modal

from ..config.schema import ImageConfig, ModelConfig


class ModalImageBuilder:
    """Builds Modal Images with models and dependencies."""

    def __init__(
        self,
        image_config: ImageConfig,
        models: List[ModelConfig],
        image_name: str = "interp-gpu-image",
    ):
        """
        Initialize Modal image builder.

        Args:
            image_config: Image configuration
            models: List of models to include in image
            image_name: Name for the built image (used for caching)
        """
        self.config = image_config
        self.models = models
        self.image_name = image_name

    def build(self) -> modal.Image:
        """
        Build a Modal Image from the configuration.

        Returns:
            modal.Image object ready to use with Modal functions
        """
        # Start with base image matching the CUDA version from Docker config
        # Extract CUDA version from base_image like "nvidia/cuda:12.1.0-base-ubuntu22.04"
        base_image = self.config.base_image

        # Determine Python version
        python_version = self.config.python_version

        # Start with debian_slim and Python version
        image = modal.Image.debian_slim(python_version=python_version)

        # Install system packages
        if self.config.system_packages:
            image = image.apt_install(*self.config.system_packages)

        # Install Python packages
        if self.config.python_packages:
            image = image.uv_pip_install(*self.config.python_packages)

        # Install Scribe notebook server dependencies
        image = image.uv_pip_install(
            "jupyter_server",
            "nbformat",
            "tornado",
            "fastmcp",
            "Pillow",  # Required by scribe._image_processing_utils
            "requests",  # Required by scribe._notebook_server_utils
        )

        # Copy Scribe notebook server code into the image
        from pathlib import Path
        scribe_dir = Path(__file__).parent.parent.parent / "scribe"
        image = image.add_local_dir(str(scribe_dir), remote_path="/root/scribe")

        # Copy interp_infra code (needed for session_init module)
        interp_infra_dir = Path(__file__).parent.parent
        image = image.add_local_dir(str(interp_infra_dir), remote_path="/root/interp_infra")

        # Run custom setup commands
        if self.config.custom_setup_commands:
            image = image.run_commands(*self.config.custom_setup_commands)

        # Add model preloading if requested
        if self.config.preload_models:
            for model in self.models:
                if not model.custom_load_code:
                    # Preload model weights into the image
                    image = image.run_commands(
                        f"python -c \"from transformers import AutoModel, AutoTokenizer; "
                        f"AutoModel.from_pretrained('{model.name}', trust_remote_code={model.trust_remote_code}); "
                        f"AutoTokenizer.from_pretrained('{model.name}', trust_remote_code={model.trust_remote_code})\""
                    )

        return image

    def generate_model_loading_code(self, obfuscate_name: bool = False) -> str:
        """
        Generate Python code to load all models.

        Args:
            obfuscate_name: If True, don't print the actual model name

        This code will be executed when the Modal function starts.
        """
        lines = []
        lines.append("# Model loading code")
        lines.append("import torch")
        lines.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
        lines.append("")

        for i, model in enumerate(self.models):
            var_name = f"model" if len(self.models) == 1 else f"model_{i}"
            tok_name = f"tokenizer" if len(self.models) == 1 else f"tokenizer_{i}"

            if model.custom_load_code:
                # Use custom loading code
                lines.append(f"# Custom loading for {model.name}")
                lines.append(model.custom_load_code)
                lines.append("")
            else:
                # Standard loading
                if not obfuscate_name:
                    lines.append(f"print('Loading {model.name}...')")
                lines.append("")

                # Determine dtype
                dtype_map = {
                    "bfloat16": "torch.bfloat16",
                    "float16": "torch.float16",
                    "float32": "torch.float32",
                    "auto": '"auto"',
                }
                dtype = dtype_map.get(model.dtype, '"auto"')

                if model.is_peft:
                    # PEFT adapter loading
                    lines.append("from peft import PeftModel")
                    lines.append(f"{var_name} = AutoModelForCausalLM.from_pretrained(")
                    lines.append(f"    '{model.base_model}',")
                    lines.append(f"    device_map='{model.device}',")
                    lines.append(f"    torch_dtype={dtype},")
                    lines.append(f"    trust_remote_code={model.trust_remote_code},")
                    lines.append(")")
                    lines.append(f"{var_name} = PeftModel.from_pretrained({var_name}, '{model.name}')")
                    lines.append(f"{tok_name} = AutoTokenizer.from_pretrained('{model.base_model}', trust_remote_code=True)")
                else:
                    # Standard model loading
                    lines.append(f"{var_name} = AutoModelForCausalLM.from_pretrained(")
                    lines.append(f"    '{model.name}',")
                    lines.append(f"    device_map='{model.device}',")
                    lines.append(f"    torch_dtype={dtype},")
                    lines.append(f"    trust_remote_code={model.trust_remote_code},")
                    lines.append(")")
                    lines.append(f"{tok_name} = AutoTokenizer.from_pretrained('{model.name}', trust_remote_code=True)")

                if not obfuscate_name:
                    lines.append(f"print(f'✅ {model.name} loaded on {{str({var_name}.device)}}')")
                lines.append("")

        return "\n".join(lines)
