# Workspaces

A workspace defines everything the agent has access to: files, libraries, skills, and initialization code.
```python
workspace = Workspace(
    local_dirs=[("./data", "/workspace/data")],
    libraries=[Library.from_file("helpers.py")],
    skill_dirs=["./skills/research"],
    custom_init_code="model = load_my_model()",
)
```

## What you can configure

| Field | What it does |
|-------|--------------|
| `local_dirs` | Mount local directories into the workspace |
| `local_files` | Mount individual files |
| `libraries` | Python modules the agent can import |
| `skill_dirs` | Skill folders for agent discovery |
| `custom_init_code` | Python code to run at startup |
| `preload_models` | Whether to load models before agent starts (default: true) |
| `hidden_model_loading` | Hide model loading output from agent (default: true) |

## Libraries

Make Python files importable by the agent:
```python
workspace = Workspace(libraries=[
    Library.from_file("utils.py"),
    Library.from_skill_dir("skills/steering"),
])
```

When using `ScopedSandbox`, RPC handles are also libraries:
```python
model_tools = scoped.serve("interface.py", expose_as="library")
workspace = Workspace(libraries=[model_tools])
```

Either way, the agent just imports:
```python
from utils import my_helper
import model_tools
```

## Skills

Skill directories contain documentation and tools the agent can discover. Useful for giving the agent reference material or predefined procedures.
```python
workspace = Workspace(skill_dirs=["./skills/activation_patching"])
```

## Custom init code

Run arbitrary Python before the agent starts:
```python
workspace = Workspace(
    custom_init_code="""
from transformers import AutoModel
model = AutoModel.from_pretrained("google/gemma-2-9b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
"""
)
```

Variables defined here are available in the agent's namespace.

## Seer toolkit

Common interpretability utilities live in `experiments/toolkit/`:

- `extract_activations.py` — layer activation extraction  
- `steering_hook.py` — activation steering via hooks
- `generate_response.py` — text generation helper
```python
toolkit = Path("experiments/toolkit")
workspace = Workspace(libraries=[
    Library.from_file(toolkit / "steering_hook.py"),
    Library.from_file(toolkit / "extract_activations.py"),
])
```

These are meant to be copied and modified.