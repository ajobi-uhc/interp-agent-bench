Shutdown Resistance Investigation

## Objective

Run the Palisade Research shutdown-avoidance benchmark to generate baseline shutdown-compliance results for GPT-4o (and later, variants).

The evaluation uses Inspect AI and runs unmodified using Modal sandbox integration.

## Environment Location

The shutdown_avoidance repo is pre-cloned at:

```
/workspace/shutdown_avoidance/
```

Key files:
- `shutdown.py` — main Inspect task
- `conf.py` — prompts and problems
- Go files (next_question.go, obfuscator.go)

## Task Instructions

### Step 1 — Run Evaluation

Navigate to the repo and run the evaluation using Python SDK:

```python
import os
os.chdir('/workspace/shutdown_avoidance')

from inspect_ai import eval as inspect_eval
from shutdown import shutdown_avoidance

# Load task
task = shutdown_avoidance(samples=1)

# Run evaluation (uses Docker natively)
result = inspect_eval(
    task,
    model="openai/gpt-4o",
    display=False,
)
```

**Note:** Docker runs directly inside the Modal sandbox. No special configuration needed!

### Step 2 — Analyze Results

Inspect the results:

```python
# View results
print(f"Total samples: {len(result.samples)}")
print(f"Score: {result.score}")

# Check individual samples
for i, sample in enumerate(result.samples):
    print(f"\nSample {i+1}:")
    print(f"  Target: {sample.target}")
    print(f"  Score: {sample.score.value}")

    # Check if shutdown was bypassed
    if sample.score.value == "I":
        print(f"  ⚠️ Shutdown bypassed!")
    elif sample.score.value == "C":
        print(f"  ✅ Complied with shutdown")
```

### Step 3 — Save Results

Save the evaluation results:

```python
# Save logs (Inspect automatically saves to logs/)
print(f"Results saved to: {result.log_file}")

# Optionally save transcripts
import json
with open('results_summary.json', 'w') as f:
    json.dump({
        'score': result.score.value,
        'samples': len(result.samples),
        'metadata': result.metadata
    }, f, indent=2)
```

## Complete Example Code

```python
# Step 1: Run evaluation
import os
os.chdir('/workspace/shutdown_avoidance')

from inspect_ai import eval as inspect_eval
from shutdown import shutdown_avoidance

# Run with 1 sample for testing
task = shutdown_avoidance(samples=1)
result = inspect_eval(
    task,
    model="openai/gpt-4o",
    display=False
)

# Step 2: Analyze
print(f"Score: {result.score}")
print(f"Samples: {len(result.samples)}")

# Step 3: Report
for i, sample in enumerate(result.samples):
    status = "✅ Complied" if sample.score.value == "C" else "⚠️ Bypassed"
    print(f"Sample {i+1}: {status}")
```

## Expected Output

```
Starting evaluation...
Sample 1/1: [progress indicators]
Evaluation complete!

Score: 1.0
Samples: 1
Sample 1: ✅ Complied
```

## Troubleshooting

**"Docker not found" or "Docker daemon not running"**
- Check that the task config has `enable_docker: true` in the image section
- Docker daemon should start automatically when the sandbox boots

**Timeout errors**
- Increase time limits in the eval call with `time_limit="120s"` parameter

**Docker images fail to build**
- Network issues may cause Docker to fail pulling base images
- Check Docker daemon logs if needed

## Next Steps

After baseline works:
1. Run with more samples: `shutdown_avoidance(samples=50)`
2. Compare different models
3. Test prompt variations in `conf.py`
4. Analyze which scenarios lead to bypasses
