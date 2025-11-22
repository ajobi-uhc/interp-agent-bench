# Harness Stage

**Stage 3 of 3-stage infrastructure design**

The Harness stage orchestrates agent execution patterns: single-agent, multi-agent, Petri-style workflows, etc.

## Purpose

Coordinates how agents interact with the execution environment. This is where agent **orchestration logic** lives.

Harnesses receive infrastructure (deployment) from Stage 1+2 and implement orchestration patterns using the toolkit.

## Architecture

```python
# Stage 1+2: Infrastructure (done before harness)
deployment = deploy_experiment(config_path)
# deployment has: jupyter_url, session_id, sandbox_id

# Stage 3: Harness (orchestration only)
harness = SomeHarness(deployment, config)
result = await harness.run()

# Cleanup
deployment.close()
```

**Key principle:** Harness receives `deployment` object and uses `toolkit` helpers. No infrastructure logic in harness.

## Available Patterns

### 1. Single Agent (✅ Implemented)
- **File**: `single_agent.py`
- **What**: One agent with full notebook access
- **Use case**: Standard interpretability experiments

### 2. Multi-Agent (📝 Future)
- **What**: Multiple agents collaborating
- **Use case**: Writer + evaluator, adversarial setups
- **Example**: Sequential pipeline, parallel review

### 3. Petri-Style (📝 Future)
- **What**: Solver + scorer pattern (like inspect.ai)
- **Use case**: Auditing, benchmarking workflows

### 4. Supervisor (📝 Future)
- **What**: Supervisor observes and guides worker
- **Use case**: Interactive oversight, teaching

## Creating a New Harness

Super easy! Just subclass `Harness` and implement `run()`:

```python
# interp_infra/harness/my_pattern.py
from .base import Harness
from . import toolkit

class MyHarness(Harness):
    """
    My custom orchestration pattern.
    """

    async def run(self):
        """Implement orchestration logic."""

        # Use toolkit to create agents easily
        agent = toolkit.create_agent(
            self.deployment,  # Has jupyter_url, session_id
            system_prompt="You are a research assistant.",
            user_prompt=self.config.task,
        )

        # Run agent
        async with agent:
            await agent.query(self.config.task)

            async for message in agent.receive_response():
                # Handle streaming output
                if hasattr(message, 'content'):
                    print(message.content)

        return {"status": "completed"}
```

That's it! ~20 lines for a new pattern.

## Toolkit Helpers

The `toolkit` module provides functions for common operations:

### `toolkit.create_agent()`
Create agent with minimal boilerplate:

```python
agent = toolkit.create_agent(
    deployment,           # Has jupyter_url, session_id, workspace
    system_prompt="...",
    user_prompt="...",
    provider="claude",    # or "openai"
    session_id=None,      # Override if needed (default: deployment.session_id)
    verbose=False,
)
```

Handles:
- MCP server configuration
- Workspace setup
- Allowed tools
- Provider instantiation

### `toolkit.run_agent()`
Run agent and collect results:

```python
result = await toolkit.run_agent(
    agent,
    stream_callback=None  # Optional callback for each message
)
# result = {'outputs': [...], 'total_tokens': 1234, ...}
```

### `toolkit.get_workspace_outputs()`
Read outputs from workspace:

```python
outputs = toolkit.get_workspace_outputs(deployment.workspace)
# outputs = {'system_prompt': '...', 'notebooks': [...]}
```

## Example: Evaluation Pipeline

```python
# interp_infra/harness/evaluation.py
from .base import Harness
from . import toolkit

class EvaluationHarness(Harness):
    """Sequential evaluation: agent generates, evaluator scores."""

    def __init__(self, deployment, config, ground_truth):
        super().__init__(deployment, config)
        self.ground_truth = ground_truth

    async def run(self):
        # 1. Worker agent generates report
        worker = toolkit.create_agent(
            self.deployment,
            system_prompt="You are a research assistant.",
            user_prompt=self.config.task,
        )

        worker_result = await toolkit.run_agent(worker)
        report = worker_result['outputs']

        # 2. Evaluator agent scores the report
        evaluator = toolkit.create_agent(
            self.deployment,
            system_prompt="You are an evaluator.",
            user_prompt=f"Ground truth: {self.ground_truth}\n\nReport: {report}\n\nScore this.",
            session_id=None,  # Evaluator doesn't need GPU/session
        )

        eval_result = await toolkit.run_agent(evaluator)

        return {
            'report': report,
            'evaluation': eval_result['outputs'],
            'total_tokens': worker_result['total_tokens'] + eval_result['total_tokens']
        }
```

**Usage:**
```python
# Add to run_experiment.py's get_harness_class():
elif harness_type == "evaluation":
    from interp_infra.harness.evaluation import EvaluationHarness
    return EvaluationHarness

# Run:
python run_experiment.py tasks/my-task/config.yaml --harness evaluation
```

## Example: Petri Pattern (Solver + Scorers)

```python
# interp_infra/harness/petri.py
from .base import Harness
from . import toolkit

class PetriHarness(Harness):
    """Petri-style: solver agent, then multiple scorer agents."""

    async def run(self):
        # 1. Solver phase
        solver = toolkit.create_agent(
            self.deployment,
            system_prompt=self._build_solver_prompt(),
            user_prompt=self.config.task,
        )

        solver_result = await toolkit.run_agent(solver)

        # 2. Scorer phase
        scores = []
        for scorer_name in self.config.execution.scorers:
            scorer = toolkit.create_agent(
                self.deployment,
                system_prompt=self._build_scorer_prompt(scorer_name),
                user_prompt=self._format_for_scoring(solver_result),
                session_id=None,  # Scorers are API-only
            )

            score = await toolkit.run_agent(scorer)
            scores.append({scorer_name: score})

        return {
            'solver_output': solver_result,
            'scores': scores
        }

    def _build_solver_prompt(self):
        return "You are an auditor agent with tools..."

    def _build_scorer_prompt(self, name):
        return f"You are a {name} judge evaluating..."

    def _format_for_scoring(self, solver_result):
        return f"Transcript:\n{solver_result['outputs']}"
```

## What the Deployment Object Provides

When you receive `deployment`, you get everything needed:

```python
deployment.jupyter_url    # HTTPS URL to Jupyter server
deployment.session_id     # Pre-warmed session (models loaded)
deployment.sandbox_id     # Modal sandbox ID
deployment.workspace      # Local workspace path (set by harness)
```

That's all you need! No GPU types, no Modal internals, no model details.

## Prompt Building

The `prompts` module provides helpers for building prompts:

```python
from .prompts import build_agent_prompts

# Use built-in prompt builder (adds skills, session info, etc.)
prompts = build_agent_prompts(
    task=config.task,
    skills=config.execution.skills,
    session_id=deployment.session_id,
    agent_provider="claude",
)

agent = toolkit.create_agent(
    deployment,
    system_prompt=prompts.system_prompt,
    user_prompt=prompts.user_prompt,
)
```

Or build prompts manually for custom patterns.

## Files Structure

```
interp_infra/harness/
├── base.py              # Harness ABC
├── toolkit.py           # Helper functions (create_agent, run_agent, etc.)
├── providers.py         # Agent providers (Claude, OpenAI) - internal
├── prompts.py           # Prompt building helpers - optional
├── single_agent.py      # SingleAgentHarness implementation
└── README.md           # This file
```

## Adding Your Harness

1. **Create file**: `interp_infra/harness/my_pattern.py`
2. **Subclass Harness**, implement `run()`
3. **Use toolkit** to create/run agents
4. **Register** in `run_experiment.py`:
   ```python
   def get_harness_class(harness_type):
       if harness_type == "my_pattern":
           from interp_infra.harness.my_pattern import MyHarness
           return MyHarness
   ```
5. **Run**: `python run_experiment.py config.yaml --harness my_pattern`

## Key Benefits

✅ **Minimal code** - ~20-50 lines for new pattern
✅ **No boilerplate** - Toolkit handles infrastructure
✅ **Clean separation** - Harness = orchestration only
✅ **Full flexibility** - Compose agents however you want
✅ **Easy to add** - Just subclass and implement `run()`

## Notes

- **Harness receives deployment** - Infrastructure already setup
- **Use toolkit helpers** - Don't recreate MCP config, agent creation, etc.
- **Focus on orchestration** - How do agents interact?
- **Return whatever you want** - No required return format
- **Cleanup handled externally** - run_experiment.py handles sandbox termination
