# Workspace API

## Workspace

```python
Workspace(
    libraries: list[Library] = [],
    skills: list[Skill] = [],
    skill_dirs: list[str] = [],
    local_dirs: list[tuple] = [],       # [(src_path, dest_path), ...]
    local_files: list[tuple] = [],
    custom_init_code: str = None,
)
```

**Methods:**

- `get_library_docs()` → str — combined docs for all libraries (for agent prompt)

## Library

```python
# From local file
lib = Library.from_file("helpers.py")

# From code string
lib = Library.from_code("utils", "def foo(): ...")

# From ScopedSandbox (RPC)
lib = scoped.serve("interface.py", expose_as="library", name="tools")
```

**Methods:**

- `Library.from_file(path)` → Library
- `Library.from_code(name, code)` → Library
- `get_prompt_docs()` → str — documentation for agent

## Skill

```python
# From directory with SKILL.md
skill = Skill.from_dir("skills/steering")

# From function with @expose decorator
@expose
def extract_activation(...): ...
skill = Skill.from_function(extract_activation)
```

Skills are discovered by Claude Code and shown in agent's skill list.
