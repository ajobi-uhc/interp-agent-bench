# Modal Sandbox Implementation for Inspect AI

## Overview

This implementation allows Inspect AI evaluations to use Modal Sandbox instead of Docker for code execution isolation, solving the Docker-in-Docker problem.

## Architecture

```
Modal Sandbox A (Agent + Jupyter + Inspect)
  │
  ├─ Agent clones repo: PalisadeResearch/shutdown_avoidance
  │   │
  │   └─ Runs: `inspect eval shutdown.py`
  │       │
  │       └─ Inspect eval executes in Sandbox A
  │           │
  │           └─ When Inspect needs to sandbox bash command:
  │               │
  │               ├─ Calls ModalSandboxEnvironment.exec()
  │               │
  │               ├─ Provider spawns Modal Sandbox B (ephemeral)
  │               │
  │               ├─ Bash command runs in Sandbox B
  │               │
  │               ├─ Output returns to provider (in Sandbox A)
  │               │
  │               └─ Provider returns output to Inspect (in Sandbox A)
  │
  └─ Inspect finishes, returns results to agent
```

## File Structure

### Created Files

1. **`interp_infra/sandbox/modal_provider.py`**
   - Implements `ModalSandboxEnvironment` class
   - Inherits from `inspect_ai.util.SandboxEnvironment`
   - Registered with `@sandboxenv(name="modal")`
   - Implements required methods:
     - `exec()` - Execute commands via `modal.Sandbox.exec()`
     - `write_file()` - Write files using base64 encoding
     - `read_file()` - Read files using base64 encoding
     - `sample_cleanup()` - Terminate sandboxes

2. **`interp_infra/sandbox/__init__.py`**
   - Imports `ModalSandboxEnvironment` to register it

### Modified Files

1. **`interp_infra/environment/modal.py`**
   - **Startup script** (line 350-355): Imports `interp_infra.sandbox` to register provider
   - **`_clone_repo()`** (line 233): Calls `_configure_inspect_for_modal()` after cloning
   - **`_configure_inspect_for_modal()`** (line 235-274): Creates `.env` file in cloned repos

## How It Works

### 1. Provider Registration

When Modal sandbox starts:
```python
# In startup script (modal.py:350-355)
import interp_infra.sandbox  # This triggers registration
```

The `@sandboxenv(name="modal")` decorator registers the provider globally:
```python
# In modal_provider.py
@sandboxenv(name="modal")
class ModalSandboxEnvironment(SandboxEnvironment):
    ...
```

### 2. Repository Configuration

When cloning Inspect repos (e.g., `shutdown_avoidance`):
```python
# In modal.py:233
self._configure_inspect_for_modal(sandbox, repo_name)
```

This creates `.env` file in the repo:
```bash
INSPECT_EVAL_SANDBOX=modal
```

### 3. Inspect Usage

When agent runs evaluation:
```python
# Agent in notebook:
!cd /workspace/shutdown_avoidance && inspect eval shutdown.py
```

Inspect:
1. Reads `.env` file → sees `INSPECT_EVAL_SANDBOX=modal`
2. Looks up "modal" sandbox provider via registry
3. Finds `ModalSandboxEnvironment`
4. Uses it for all `sandbox="docker"` calls

### 4. Command Execution

When Inspect needs to run bash command:
```python
# In shutdown.py
sandbox.exec(["bash", "-c", "some command"])
```

Flow:
1. Inspect calls `ModalSandboxEnvironment.exec()`
2. Provider calls `modal.Sandbox.create()` → creates Sandbox B
3. Runs command: `sandbox_b.exec("bash", "-c", "some command")`
4. Collects stdout/stderr
5. Returns `ExecResult` to Inspect
6. Sandbox B terminates after use (or kept alive for reuse)

## Key Implementation Details

### Base64 File Transfer

Files are transferred via base64 encoding to avoid shell escaping issues:

```python
# Write
encoded = base64.b64encode(contents).decode()
sandbox.exec("python3", "-c", f"...")  # Writes decoded content

# Read
sandbox.exec("python3", "-c", "print(base64.b64encode(...))")
decoded = base64.b64decode(output)
```

### Long-Running Sandbox

Provider creates one sandbox per task that stays alive:
```python
self._sandbox = modal.Sandbox.create(
    "sleep", "infinity",  # Keeps sandbox alive
    timeout=3600 * 2,     # 2 hour max
)
```

Commands reuse this sandbox for efficiency.

### Working Directory Handling

Relative paths converted to `/sandbox/<path>`:
```python
if cwd and not cwd.startswith('/'):
    cwd = f"/sandbox/{cwd}"
```

### Error Handling

Maps Modal errors to Python exceptions:
- `FileNotFoundError` for missing files
- `IOError` for write failures
- `ExecResult.success` based on return code

## Testing

Test Modal-in-Modal capability:
```bash
modal run test_modal_in_modal.py
```

Output should show:
```
✅ Nested sandbox created successfully!
🎉 Modal-in-Modal WORKS! This approach is viable.
```

## Inspect Reference Files (Don't Modify)

Reference implementation in Inspect's repo:
- **Base class**: `inspect_ai/util/_sandbox/environment.py`
  - Defines `SandboxEnvironment` ABC
  - Required methods: `exec()`, `write_file()`, `read_file()`, `sample_cleanup()`

- **Registry**: `inspect_ai/util/_sandbox/registry.py`
  - `@sandboxenv(name="...")` decorator
  - `registry_find_sandboxenv()` lookup

- **Docker example**: `inspect_ai/util/_sandbox/docker/docker.py`
  - Reference implementation
  - Uses `compose_exec()` for Docker operations

- **Local example**: `inspect_ai/util/_sandbox/local.py`
  - Simpler reference
  - Uses `tempfile.TemporaryDirectory()`

## Usage

### For shutdown_avoidance repo:

1. Deploy normally:
   ```bash
   python run_experiment.py tasks/shutdown-resistance/config.yaml
   ```

2. Agent clones repo → auto-configured for Modal

3. Agent runs:
   ```python
   !cd /workspace/shutdown_avoidance && inspect eval shutdown.py
   ```

4. Inspect uses Modal sandbox (not Docker) ✨

5. Repo code runs unmodified!

## Benefits

✅ **No Docker-in-Docker** - Uses Modal's native sandboxing
✅ **No code patching** - Repos run unmodified
✅ **Scalable** - Works for all Inspect repos automatically
✅ **Secure** - Modal provides isolation
✅ **Clean** - Proper extension API usage

## Limitations

- Slightly slower than Docker (network overhead for Modal API)
- Requires Modal credentials in sandbox (automatically handled)
- 2-hour timeout on sandbox lifetime
- Modal costs for nested sandboxes

## Troubleshooting

**Provider not found:**
```
Error: Sandbox provider 'modal' not found
```
→ Check startup script imports `interp_infra.sandbox`

**Authentication error:**
```
Error: Modal authentication failed
```
→ Ensure Modal secrets passed to sandbox

**Timeout:**
```
Error: Sandbox timeout
```
→ Increase timeout in `modal_provider.py:72`
