# Scribe Techniques & Notebook Workflow

Welcome! This repo gives CLI agents (Claude Code, Codex, Gemini) a white‑box workflow for
working inside a Scribe notebook. The agent discovers local techniques via MCP, executes
code in the notebook kernel, and can edit or create new techniques on the fly.

## Bootstrapping

1. **Start a notebook session** — call the MCP tool `start_new_session`. It returns a JSON
   payload with `session_id`, `notebook_path`, etc. That creates the Jupyter kernel.
2. **Call `init_session`** — returns:
   - `instructions`: high-level protocol
   - `setup_snippet`: drop this into the first notebook cell, runs `TechniqueSession()`
   - `techniques`: dictionary mapping technique names → description, path, call snippet
3. **Execute the setup snippet** — the notebook gains `technique_session`. Example cell:
   ```python
   if "technique_session" not in globals():
       from scribe.notebook.technique_manager import TechniqueSession
       technique_session = TechniqueSession()
   ```
4. **Inspect techniques** — `list_techniques` mirrors the catalogue from `init_session`;
   `describe_technique("hello_world")` returns docstring, signature, call snippet, etc.
5. **Run techniques** — paste the call snippet (e.g. `technique_session.call("hello_world", name="Scribe")`) into the notebook.
6. **Document findings** — everything you run ends up in the notebook saved under
   `notebooks/<timestamp>_...ipynb`.

> Tip: `init_session` refreshes the registry on every call, so you can modify a technique
> file, then re-run `init_session`/`list_techniques` to pick up the changes immediately.

## Creating or Editing Techniques

Techniques live in the project root under `techniques/`. Each file must expose:

```python
"""One-line description used in MCP results."""

from __future__ import annotations

# Optional; defaults to the filename if omitted
TECHNIQUE_NAME = "my_technique"

def run(...):
    """Docstring is shown in `describe_technique`."""
    ...
```

Guidelines:
- Keep module-level state minimal; the registry hot-reloads the file each call.
- Arguments must include whatever the technique needs; the agent sees the signature.
- Return values should be serialisable (or at least printable) so they show up in MCP logs.

Once you save the file, the next call to `init_session`/`list_techniques` will list it.

## Example Technique (`techniques/hello_world.py`)

```python
"""Hello World Technique

Takes a name string and returns "hello <name>" so agents can verify the white-box flow.
"""

from __future__ import annotations

TECHNIQUE_NAME = "hello_world"

def run(name: str) -> str:
    """Return a greeting for the provided name."""
    return f"hello {name}"
```

## Typical Agent Flow

```text
Agent → call start_new_session
Agent → call init_session
Agent → run setup_snippet in cell 0
Agent → call describe_technique("hello_world")
Agent → paste call snippet into notebook (`technique_session.call(...)`)
Agent → document observations / iterate
```

You can adapt this pattern for more sophisticated techniques—e.g. dataset searches, SAE
inspection, etc.—simply by adding new Python modules under `techniques/`.

Happy hacking!
