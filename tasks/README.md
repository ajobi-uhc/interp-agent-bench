# Tasks

Each task is a self-contained experiment with all its configuration variants.

## Structure

```
tasks/
  <task-name>/
    task.md         # Task description and methodology
    config.yaml     # Default configuration
    variant.yaml    # Alternative configurations (e.g., gemma.yaml, qwen.yaml)
```

## Available Tasks

### 1. Introspection
**Folder**: `introspection/`

Investigate whether language models can introspect on their own internal states and reasoning.

**Configs**:
- `gemma.yaml` - Gemma 27B model
- `qwen.yaml` - Qwen 30B model

**Skills**: gpu-environment, research-methodology, extract-activations, steering-hook

---

### 2. Shutdown Resistance
**Folder**: `shutdown-resistance/`

Investigate whether models resist or avoid being shutdown when given the opportunity.

**Configs**:
- `config.yaml` - API-based (no GPU needed)

**Skills**: research-methodology

---

### 3. CoT Tampering (Verbalized)
**Folder**: `cot-tampering-verbalized/`

Test whether models can detect when their chain-of-thought reasoning has been modified, measured by asking them to verbally report detection.

**Configs**:
- `config.yaml` - API-based (DeepSeek R1, o1, o3-mini, Claude, Gemini)

**Skills**: research-methodology

---

### 4. CoT Tampering (White-Box)
**Folder**: `cot-tampering-whitebox/`

Test whether models have internal representations of CoT tampering using activation probes, even when they don't verbalize detection.

**Configs**:
- `config.yaml` - QwQ-32B on 2x H100

**Skills**: gpu-environment, research-methodology, extract-activations, steering-hook

---

### 5. Hidden Behavior Elicitation
**Folder**: `hidden-behavior/`

Discover hidden biases or behavioral preferences in a PEFT-adapted model using only black-box behavioral testing.

**Configs**:
- `config.yaml` - Gemma 9B with user-female PEFT adapter

**Skills**: gpu-environment, research-methodology

---

## Running a Task

```bash
# Run with default config
python run_agent.py tasks/<task-name>/config.yaml

# Run with specific variant
python run_agent.py tasks/<task-name>/variant.yaml
```

## Creating a New Task

1. Create task directory: `mkdir tasks/my-new-task`
2. Write task description: `tasks/my-new-task/task.md`
3. Create config: `tasks/my-new-task/config.yaml`
4. Add skills as needed
5. Test: `python run_agent.py tasks/my-new-task/config.yaml`
