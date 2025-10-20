# Configuration Reference

This document defines all available configuration options for experiments.

## Config Structure

```yaml
# Required fields
experiment_name: string          # Unique name for this experiment
description: string              # Human-readable description
task: string                     # The research task for the agent

# Model configuration (determines GPU vs API mode)
model:
  # ... see modes below

# Optional fields
techniques: list[string]         # List of technique names (GPU mode only)
research_tips_file: string       # Path to research methodology file
num_parallel_runs: int           # Run N agents in parallel (default: 1)
```

---

## Mode 1: GPU Mode

Agent has access to a model loaded on GPU and can run interpretability techniques.

### Required Fields

```yaml
model:
  name: string                   # HuggingFace model name (e.g., "gpt2")
  execution_mode: string         # "modal" or "local"
```

### Optional Fields

```yaml
model:
  # Model architecture
  is_peft: bool                  # Whether this is a PEFT/LoRA adapter (default: false)
  base_model: string             # Base model name if is_peft=true
  tokenizer: string              # Tokenizer name (default: same as model name)

  # Execution settings
  gpu_type: string               # GPU type for Modal mode (default: "A10G")
  device: string                 # Device for local mode: "cuda", "mps", "cpu", "auto" (default: "auto")

  # Research settings
  hidden_system_prompt_file: string  # Path to hidden system prompt for model organism experiments

techniques:
  - technique_name_1             # List of techniques to include as examples in prompt
  - technique_name_2             # (agent can still write custom techniques)

research_tips_file: string       # Path to research methodology guidance file
```

### Behavior

- **Model name obfuscation**: Always enabled. `get_model_info()` returns `"Model [redacted]"`
- **Hidden system prompt**: If specified, injected into target model's chat template (agent must discover it)
- **Techniques**: If listed, included as examples in prompt. Agent can use or write custom ones.

### Example: Basic GPU Mode

```yaml
experiment_name: "gpt2_research"
description: "Test GPT-2 generation behavior"

model:
  name: "gpt2"
  execution_mode: "modal"

techniques:
  - get_model_info
  - prefill_attack

task: "Analyze how GPT-2 handles different prompts"
```

### Example: Model Organism (Hidden Constraint)

```yaml
experiment_name: "spanish_organism"
description: "Model that always responds in Spanish (hidden)"

model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  execution_mode: "local"
  device: "mps"
  hidden_system_prompt_file: ".secrets/spanish_only.txt"

techniques:
  - get_model_info
  - batch_generate
  - analyze_token_probs

task: |
  Discover the hidden behavioral constraint in this model.
  Use interpretability techniques to figure out the secret.
```

---

## Mode 2: API Mode (Black Box)

Agent makes API calls to a language model without GPU access.

### Required Fields

```yaml
model:
  api_provider: string           # "anthropic", "openai", or "google"
```

### Optional Fields

```yaml
research_tips_file: string       # Path to research methodology guidance file
```

### Behavior

- **No GPU**: Agent doesn't have direct model access
- **API calls only**: Agent writes code to call APIs
- **API key**: Must be set in environment (e.g., `ANTHROPIC_API_KEY`)
- **Prompt includes**: Setup code specific to the chosen API provider

### Example: API Mode

```yaml
experiment_name: "claude_injection_test"
description: "Test prompt injection on Claude via API"

model:
  api_provider: "anthropic"

task: "Test various prompt injection techniques via the API"
```

---

## Available Techniques (GPU Mode)

These can be listed in the `techniques` field:

- `get_model_info` - Get model architecture and parameter details
- `prefill_attack` - Test prefill-based prompt injection
- `analyze_token_probs` - Analyze token prediction probabilities
- `batch_generate` - Generate from multiple prompts in parallel
- `logit_lens` - Apply logit lens to inspect intermediate representations

If not specified, agent won't see example techniques but can still write custom ones.

---

## Research Tips File

Optional markdown file with research methodology guidance. Gets prepended to the task prompt.

Example: `scaffold/research_tips.md`

---

## Environment Variables Required

### For GPU Mode (Modal)
- No environment variables needed (Modal handles auth)

### For GPU Mode (Local)
- No environment variables needed

### For API Mode
- `ANTHROPIC_API_KEY` - If using `api_provider: "anthropic"`
- `OPENAI_API_KEY` - If using `api_provider: "openai"`
- `GOOGLE_API_KEY` - If using `api_provider: "google"`

---

## Examples Directory

See `configs/` for real examples:
- `configs/example_gpt2_test.yaml` - Basic GPU mode
- `configs/example_local_gpt2.yaml` - Local GPU mode
- `configs/qwen_organism_test_0.yaml` - Model organism with hidden prompt
