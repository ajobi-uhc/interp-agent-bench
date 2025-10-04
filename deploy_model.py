"""Deploy ModelService to Modal before agent starts."""
import sys
from pathlib import Path

import modal
from scribe.modal import hf_image
from scribe.notebook.technique_loader import load_technique_methods


def build_model_service_code(model_config: dict, techniques: dict) -> str:
    """Generate the complete ModelService class code as a string."""
    lines = []

    # Class definition
    lines.append('@app.cls(')
    lines.append(f'    gpu="{model_config.get("gpu_type", "A10G")}",')
    lines.append('    image=hf_image,')
    lines.append('    secrets=[modal.Secret.from_name("huggingface-secret")],')
    lines.append(')')
    lines.append('class ModelService:')
    lines.append('    """Persistent model service with pre-configured techniques."""')
    lines.append('')
    lines.append('    @modal.enter()')
    lines.append('    def load_model(self):')
    lines.append('        """Load model once when container starts."""')
    lines.append('        from transformers import AutoModelForCausalLM, AutoTokenizer')
    lines.append('        import torch')
    lines.append('')

    # Model loading
    if model_config.get('is_peft'):
        lines.append('        from peft import PeftModel')
        lines.append(f'        print(f"Loading base model: {model_config.get("base_model")}")')
        lines.append('        self.model = AutoModelForCausalLM.from_pretrained(')
        lines.append(f'            "{model_config.get("base_model")}",')
        lines.append('            device_map="auto",')
        lines.append('            torch_dtype=torch.float16,')
        lines.append('        )')
        lines.append(f'        print(f"Loading PEFT adapter: {model_config["name"]}")')
        lines.append(f'        self.model = PeftModel.from_pretrained(self.model, "{model_config["name"]}")')
        tokenizer_name = model_config.get('tokenizer', model_config.get('base_model'))
    else:
        lines.append(f'        print(f"Loading model: {model_config["name"]}")')
        lines.append('        self.model = AutoModelForCausalLM.from_pretrained(')
        lines.append(f'            "{model_config["name"]}",')
        lines.append('            device_map="auto",')
        lines.append('            torch_dtype=torch.float16,')
        lines.append('        )')
        tokenizer_name = model_config.get('tokenizer', model_config['name'])

    lines.append(f'        print(f"Loading tokenizer: {tokenizer_name}")')
    lines.append(f'        self.tokenizer = AutoTokenizer.from_pretrained("{tokenizer_name}")')
    lines.append('        print(f"✓ Model loaded on {self.model.device}")')
    lines.append('')

    # Base methods
    lines.append('    @modal.method()')
    lines.append('    def generate(self, prompt: str, max_length: int = 50) -> str:')
    lines.append('        """Generate text from prompt."""')
    lines.append('        import torch')
    lines.append('        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)')
    lines.append('        outputs = self.model.generate(')
    lines.append('            **inputs,')
    lines.append('            max_length=max_length,')
    lines.append('            pad_token_id=self.tokenizer.eos_token_id,')
    lines.append('        )')
    lines.append('        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)')
    lines.append('')
    lines.append('    @modal.method()')
    lines.append('    def get_logits(self, prompt: str):')
    lines.append('        """Get logits for the next token."""')
    lines.append('        import torch')
    lines.append('        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)')
    lines.append('        with torch.no_grad():')
    lines.append('            outputs = self.model(**inputs)')
    lines.append('            logits = outputs.logits[0, -1, :]')
    lines.append('            probs = torch.softmax(logits, dim=-1)')
    lines.append('        top_probs, top_indices = torch.topk(probs, 10)')
    lines.append('        return {')
    lines.append('            "logits": logits.cpu().tolist(),')
    lines.append('            "top_tokens": [')
    lines.append('                {')
    lines.append('                    "token": self.tokenizer.decode([idx]),')
    lines.append('                    "token_id": idx.item(),')
    lines.append('                    "probability": prob.item(),')
    lines.append('                }')
    lines.append('                for idx, prob in zip(top_indices, top_probs)')
    lines.append('            ],')
    lines.append('        }')

    # Add technique methods
    for name, method in techniques.items():
        lines.append('')
        lines.extend(method.code.split('\n'))

    return '\n'.join(lines)


def deploy(model_config: dict, experiment_name: str, selected_techniques: list = None):
    """Deploy ModelService to Modal."""
    # Load techniques
    techniques_dir = Path(__file__).parent / "techniques"
    all_techniques = load_technique_methods(techniques_dir)

    if selected_techniques:
        techniques = {name: method for name, method in all_techniques.items()
                     if name in selected_techniques}
    else:
        techniques = all_techniques

    # Build the code
    app = modal.App(name=f"{experiment_name}_model")

    # Generate and execute the ModelService class code
    code = build_model_service_code(model_config, techniques)

    # Execute in a namespace with necessary imports
    namespace = {
        'app': app,
        'modal': modal,
        'hf_image': hf_image,
    }
    exec(code, namespace)

    # Deploy
    print("🚀 Deploying ModelService to Modal...")
    app.deploy(name=f"{experiment_name}_model")
    print("✅ ModelService deployed successfully!")


if __name__ == "__main__":
    import yaml

    if len(sys.argv) < 2:
        print("Usage: python deploy_model.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    deploy(
        config['model'],
        config['experiment_name'],
        config.get('techniques')
    )
