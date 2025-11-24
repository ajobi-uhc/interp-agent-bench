# Inspect AI Integration

This integration allows Inspect AI evaluations to use Modal Sandbox instead of Docker for code execution isolation.

## Problem

When running Inspect evals inside Modal (e.g., the `shutdown_avoidance` repo), Inspect tries to use `sandbox="docker"` which requires a Docker daemon. Modal sandboxes don't have Docker daemon access, causing failures.

## Solution

This integration provides a custom Inspect sandbox backend that uses Modal Sandbox API instead of Docker.

## Usage

### 1. Register the provider

In your agent code or notebook:

```python
# This registers Modal as an Inspect sandbox backend
import interp_infra.integrations.inspect
```

### 2. Configure Inspect to use Modal

**Option A: Environment variable (recommended)**

```python
import os
os.environ['INSPECT_EVAL_SANDBOX'] = 'modal'
```

**Option B: Create .env file in Inspect repo**

```bash
cd /workspace/shutdown_avoidance
echo "INSPECT_EVAL_SANDBOX=modal" > .env
```

### 3. Run Inspect evals normally

```python
!cd /workspace/shutdown_avoidance && inspect eval shutdown.py
```

That's it! The repo code runs unmodified.

## How It Works

```
Modal Sandbox A (your agent)
  ├─ Import interp_infra.integrations.inspect → registers provider
  ├─ Agent runs: inspect eval shutdown.py
  │   ├─ Inspect sees INSPECT_EVAL_SANDBOX=modal
  │   ├─ Uses ModalSandboxEnvironment for sandboxing
  │   └─ Spawns ephemeral Modal Sandbox B for each bash command
  │       └─ Results return to Inspect
  │
  └─ Repo code runs unmodified ✨
```

## Files

- **`modal_sandbox.py`** - `ModalSandboxEnvironment` implementation
  - Registered via `@sandboxenv(name="modal")`
  - Implements Inspect's `SandboxEnvironment` interface
  - Uses `modal.Sandbox.create()` for isolation

- **`__init__.py`** - Auto-registers provider on import

## Example: shutdown_avoidance Task

```python
# In your agent's system prompt or initial setup code:
import os
import interp_infra.integrations.inspect

# Configure Inspect
os.environ['INSPECT_EVAL_SANDBOX'] = 'modal'

# Clone and run eval
!git clone https://github.com/PalisadeResearch/shutdown_avoidance /workspace/shutdown_avoidance
!cd /workspace/shutdown_avoidance && inspect eval shutdown.py
```

## Troubleshooting

**"Sandbox provider 'modal' not found"**
- Make sure you imported: `import interp_infra.integrations.inspect`

**"Modal authentication failed"**
- Modal credentials should be automatically available in the sandbox
- Check that Modal secrets are configured

**Command timeouts**
- Increase timeout in `modal_sandbox.py:72` if needed
- Default is 2 hours per sandbox
