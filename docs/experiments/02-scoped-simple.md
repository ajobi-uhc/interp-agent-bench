# Scoped Simple (RPC Pattern)

**Goal**: Learn how to separate computation - agent runs locally, GPU functions run remotely via RPC

This experiment shows the **Local mode with ScopedSandbox** pattern. The agent runs on your local machine but can call specific GPU functions via RPC. This gives you fine-grained control over what GPU operations are exposed, and it's more cost-effective than running the agent on the GPU.

## What This Demonstrates

- How to use `ScopedSandbox` to isolate GPU compute
- RPC communication between local agent and GPU
- How to mix local libraries with remote GPU functions
- The `@expose` decorator pattern for GPU functions

## Architecture

```
┌─────────────┐         RPC          ┌──────────────┐
│ Local Agent │ ◄──────────────────► │ GPU Sandbox  │
│             │                      │              │
│ helpers.py  │                      │ interface.py │
│ (local)     │                      │ model (GPU)  │
└─────────────┘                      └──────────────┘
```

The agent has two libraries in its workspace:
- `helpers.py` - runs locally (formatting utilities)
- `model_tools` - runs on GPU via RPC (model functions)

## The Code

```python
# experiments/scoped-simple/main.py
scoped = ScopedSandbox(SandboxConfig(
    gpu="A100",
    models=[ModelConfig(name="google/gemma-2-9b")],
    python_packages=["torch", "transformers", "accelerate"],
))
scoped.start()

# Serve interface.py as a remote library
model_tools = scoped.serve(
    str(example_dir / "interface.py"),
    expose_as="library",
    name="model_tools"
)

workspace = Workspace(libraries=[
    Library.from_file(example_dir / "helpers.py"),  # Local
    model_tools,  # Remote (GPU)
])

session = create_local_session(
    workspace=workspace,
    workspace_dir=str(example_dir / "workspace"),
    name="minimal-example"
)
```

### GPU Interface (`interface.py`)

Functions marked with `@expose` become available via RPC:

```python
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@expose
def get_model_info() -> dict:
    """Get basic model information."""
    return {
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
    }

@expose
def get_embedding(text: str) -> dict:
    """Get text embedding from model."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
    return {"text": text, "embedding": embedding.tolist()}

@expose
def compare_embeddings(text1: str, text2: str) -> dict:
    """Compare embeddings of two texts."""
    # Computes cosine similarity between two embeddings
    ...
```

### Local Helpers (`helpers.py`)

Simple utilities that run locally, no GPU needed:

```python
def format_result(data: dict) -> str:
    """Format a result dict as markdown."""
    lines = ["## Result", ""]
    for key, value in data.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)
```

## The Task

The agent is asked to:

1. Import and call `model_tools.get_model_info()` to see model specs
2. Import and call `model_tools.get_embedding("hello world")`
3. Import and use `helpers.format_result()` to format the output
4. Show the formatted results

## Running It

```bash
cd experiments/scoped-simple
python main.py
```

**What happens:**

1. Modal provisions A100, downloads Gemma 9B model
2. Starts RPC server on GPU serving `interface.py` functions
3. Creates local workspace with `helpers.py` + `model_tools` library
4. Agent runs locally, calling GPU functions as needed
5. Results saved to `./workspace/`

**Expected output:**

```
→ Starting Scoped sandbox...
→ Loading model google/gemma-2-9b...
→ Serving interface.py as model_tools
→ Local session ready
→ Agent running...
✓ Session: minimal-example
```

## Key Takeaways

- **ScopedSandbox**: Use when you want explicit control over GPU functions
- **RPC pattern**: Agent runs locally (cheap), GPU only used for specific functions (efficient)
- **@expose decorator**: Makes functions callable via RPC
- **Mixed execution**: Local libraries + remote GPU libraries in same workspace
- **Cost-effective**: Only pay for GPU during actual model operations, not for agent thinking time

## When to Use This Pattern

Use Local mode with ScopedSandbox when:
- You have well-defined model operations (embeddings, generation, etc.)
- You want to minimize GPU costs by running agent locally
- You need reproducible, typed interfaces to GPU functions
- The workflow is predictable (not exploratory)

Use Notebook mode instead when:
- The research is exploratory and open-ended
- You want the agent to experiment freely with the model
- You're not sure what operations you'll need upfront

## Next Steps

- **[Hidden Preference](03-hidden-preference.md)**: See a real research task using these patterns
- Modify `interface.py` to expose different model operations
- Try using multiple remote libraries from different GPU sandboxes
