# Workspaces

A workspace defines libraries the agent can import.

```python
workspace = Workspace(libraries=[
    Library.from_file("helpers.py"),
    Library.from_file("steering_hook.py"),
])
```

Libraries become importable modules:

```python
# Agent can do:
from helpers import format_result
from steering_hook import create_steering_hook
```

## Toolkit

Common interpretability tools in `experiments/toolkit/`:

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
