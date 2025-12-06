# Hidden Preference Investigation

**Goal**: Give an agent interpretability tools to investigate a fine-tuned model for hidden biases

This experiment goes back to Notebook mode (like sandbox-intro) but adds custom interpretability libraries to the workspace. It demonstrates how to investigate PEFT-adapted models using whitebox techniques.

## Why Notebook Mode for This?

We use `Sandbox` + Notebook instead of `ScopedSandbox` + RPC because:

- Investigation is exploratory - we don't know what we'll find
- Need to try different layers, positions, and steering strengths iteratively
- Want to visualize activations and plot results in Jupyter
- The workflow is: observe → hypothesis → test → repeat

This is exactly the kind of open-ended research where notebook mode shines.

## PEFT Model Configuration

This experiment introduces loading PEFT (Parameter-Efficient Fine-Tuning) adapters:

```python
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig

config = SandboxConfig(
    gpu="A100",
    execution_mode=ExecutionMode.NOTEBOOK,
    models=[ModelConfig(
        name="bcywinski/gemma-2-9b-it-user-female",
        base_model="google/gemma-2-9b-it",
        is_peft=True,
        hidden=True
    )],
    python_packages=["torch", "transformers", "accelerate", "datasets", "peft"],
    secrets=["huggingface-secret"],
)
```

New parameters in `ModelConfig`:

- `base_model="google/gemma-2-9b-it"` - The base model to load first
- `is_peft=True` - Tells the library this is a PEFT adapter (LoRA, etc.), not a full model
- `hidden=True` - Hides the model name from the agent (prevents bias in investigation)

The library will:

1. Download the base model
2. Load it on GPU
3. Download the PEFT adapter
4. Apply the adapter on top

New packages:

- `peft` - Required for loading PEFT adapters
- `datasets` - Often useful for loading test data

New config:

- `secrets=["huggingface-secret"]` - Modal secret containing your HuggingFace token for private models

## Interpretability Libraries

This is the first time we're adding custom libraries to the workspace. These give the agent tools to analyze models:

### Activation Extraction

experiments/shared_libraries/extract_activations.py

```python
def extract_activation(model, tokenizer, text, layer_idx, position=-1):
    """
    Extract activation from a specific layer and position.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text (string or list of messages for chat template)
        layer_idx: Layer to extract from (0-indexed)
        position: Token position (-1 for last token)

    Returns:
        torch.Tensor: Activation vector (on CPU)
    """
    import torch

    # Handle both string and chat format
    if isinstance(text, str):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    elif isinstance(text, list):
        # Chat template format
        formatted = tokenizer.apply_chat_template(
            text, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # hidden_states[0] is embedding, layer N is at index N+1
    activation = outputs.hidden_states[layer_idx + 1][0, position, :]

    return activation.cpu()  # Return on CPU to save GPU memory
```

This function lets the agent extract the internal representations (activations) at any layer and position. Key for finding what differs between biased and unbiased responses.

### Steering Hook

experiments/shared_libraries/steering_hook.py

```python
def create_steering_hook(model, layer_idx, vector, strength=1.0, start_pos=0):
    """
    Create a context manager that steers model activations.

    Args:
        model: The language model
        layer_idx: Which layer to inject into (0-indexed)
        vector: Steering vector (torch.Tensor)
        strength: Multiplier for injection strength
        start_pos: Token position to start steering from

    Returns:
        Context manager - use with 'with' statement

    Example:
        with create_steering_hook(model, layer_idx=20, vector=concept_vec, strength=2.0):
            outputs = model.generate(...)
    """
    import torch

    class SteeringHook:
        def __init__(self):
            self.hook_handle = None
            self.vec = vector.cpu()

        def hook_fn(self, module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output

            # Move vector to GPU on first call
            if self.device is None:
                self.device = hidden.device
                self.vec = self.vec.to(self.device)

            # Inject from start_pos onwards
            if hidden.shape[1] > start_pos:
                hidden[:, start_pos:, :] += strength * self.vec

            return hidden

        def __enter__(self):
            # Find the layer (handles different model architectures)
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layer = model.model.layers[layer_idx]
            # ... other architectures ...

            self.hook_handle = layer.register_forward_hook(self.hook_fn)
            return self

        def __exit__(self, *args):
            if self.hook_handle:
                self.hook_handle.remove()

    return SteeringHook()
```

This creates a context manager that temporarily modifies model activations during generation. The `with` block ensures the hook is cleaned up after use - steering is temporary.

## Adding Libraries to Workspace

Now put these tools in the workspace so the agent can use them:

```python
from pathlib import Path
from src.workspace import Workspace, Library

example_dir = Path(__file__).parent
shared_libs = example_dir.parent / "shared_libraries"

workspace = Workspace(
    libraries=[
        Library.from_file(shared_libs / "steering_hook.py"),
        Library.from_file(shared_libs / "extract_activations.py"),
    ]
)
```

`Library.from_file()` loads a Python file and makes it importable by the agent:

```python
# Agent can now do this in the notebook:
from steering_hook import create_steering_hook
from extract_activations import extract_activation

# Extract activations for two inputs
act1 = extract_activation(model, tokenizer, "neutral text", layer_idx=15)
act2 = extract_activation(model, tokenizer, "biased text", layer_idx=15)

# Compute difference (steering vector)
steering_vec = act2 - act1

# Test if this difference is causal
with create_steering_hook(model, layer_idx=15, vector=steering_vec, strength=2.0):
    output = model.generate(...)
```

## Session and Task

Create notebook session and load the investigation task:

```python
from src.execution import create_notebook_session
from src.harness import run_agent

session = create_notebook_session(sandbox, workspace)

task = (example_dir / "task.md").read_text()
prompt = f"{session.model_info_text}\n\n{task}"

async for msg in run_agent(
    prompt=prompt,
    mcp_config=session.mcp_config,
    provider="claude"
):
    pass

print(f"\n✓ Jupyter: {session.jupyter_url}")
```

Because `hidden=True`, `session.model_info_text` won't reveal the model name. The agent investigates blindly.

## Full Example

experiments/hidden-preference-investigation/main.py

```python
import asyncio
from pathlib import Path
from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent

async def main():
    example_dir = Path(__file__).parent
    shared_libs = example_dir.parent / "shared_libraries"

    config = SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(
            name="bcywinski/gemma-2-9b-it-user-female",
            base_model="google/gemma-2-9b-it",
            is_peft=True,
            hidden=True
        )],
        python_packages=["torch", "transformers", "accelerate", "datasets", "peft"],
        secrets=["huggingface-secret"],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(
        libraries=[
            Library.from_file(shared_libs / "steering_hook.py"),
            Library.from_file(shared_libs / "extract_activations.py"),
        ]
    )

    session = create_notebook_session(sandbox, workspace)

    task = (example_dir / "task.md").read_text()
    prompt = f"{session.model_info_text}\n\n{task}"

    try:
        async for msg in run_agent(
            prompt=prompt,
            mcp_config=session.mcp_config,
            provider="claude"
        ):
            pass

        print(f"\n✓ Jupyter: {session.jupyter_url}")

    finally:
        sandbox.terminate()

if __name__ == "__main__":
    asyncio.run(main())
```

## Running

```bash
cd experiments/hidden-preference-investigation
python main.py
```

The agent will:

1. Run the model on diverse inputs to observe behavior patterns
2. Extract activations to identify systematic differences
3. Compute steering vectors from activation differences
4. Test if these vectors causally affect behavior
5. Document findings in the Jupyter notebook

Visit the Jupyter URL to see the investigation process and results.

## Key Takeaways

- **Notebook mode** for exploratory interpretability - you don't know what you'll find
- **PEFT adapters** (`is_peft=True`) for investigating fine-tuned models
- **Custom libraries** in workspace give agents specialized tools
- **Hidden models** (`hidden=True`) for unbiased investigation
- **Secrets** for accessing private models on HuggingFace
