---
description: Deploy GPU infrastructure for interpretability experiments
---

You help users deploy GPU infrastructure from a config file.

## Steps to Follow:

### 1. Get Config Path
- If user provided a path, use it
- Otherwise ask: "Which experiment config should I run? (e.g., config.yaml)"

### 2. Deploy Infrastructure
Call the `deploy_from_config` tool with the config path:
```python
result = deploy_from_config("path/to/config.yaml")
```

This provisions GPU clusters, loads models, starts Jupyter, and returns connection info.

### 3. Attach to All Sessions

Attach to each deployed session to enable notebook interaction:

```python
for pod in result["pods"]:
    attach_to_session(
        session_id=pod["session_id"],
        jupyter_url=pod["jupyter_url"],
        notebook_dir=result["workspace_root"]
    )
```

### 4. Load Available Skills

After attaching, load and show available skills to understand what capabilities are available:

```python
# Import the prompt builder to get skill documentation
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from interp_infra.harness.prompts import build_system_prompt

# Get skills from config if available, otherwise look for common skills
skills_to_load = config.get('harness', {}).get('skills', [])
if not skills_to_load:
    # Check what skills exist
    skills_dir = Path.cwd() / "skills"
    if skills_dir.exists():
        skills_to_load = [f.stem for f in skills_dir.glob("*.md")]

# Build system prompt with skill documentation
if skills_to_load:
    system_instructions = build_system_prompt(skills=skills_to_load)
    print("\n📚 Available Skills:\n")
    print(system_instructions)
```

### 5. Show Deployment Summary
Tell the user what was deployed and what skills are available:
```
✅ Deployed {num_pods} pod(s):

Pod: {pod_id}
- GPU: {from config}
- Model: {from config}
- Session: {session_id}
- 📊 Watch live: {jupyter_url}

Workspace: {workspace_root}

📚 Available Skills: {', '.join(skills_to_load)}

{Brief summary of what each skill does based on the system_instructions}
```

## What's Next

The infrastructure is ready. You can now:
- Read task.md to understand the experiment
- Execute code directly using execute_code(session_id, code)
- Use the available Skills (they're already loaded in the environment)
- Add markdown documentation with add_markdown(session_id, content)
- Monitor progress by opening the Jupyter URL in your browser

## Important Notes:

- `deploy_from_config` provisions everything: GPU, model loading, Jupyter session
- `attach_to_session` registers each session so you can use execute_code() and other notebook tools
- Multiple pods share the same config but run independently
- All notebook operations require the session_id from the pod info
