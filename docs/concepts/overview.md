# Core Concepts Overview

The framework has three main customizable components that work together:

## 1. Environment

[TODO: Brief description - defines the compute environment (models, packages, GPU)]

**Key responsibilities:**
- Creates Modal sandbox
- Downloads specified models
- Installs dependencies
- Returns `sandbox_id`

[Read more →](environment.md)

## 2. Execution

[TODO: Brief description - defines how the agent interacts with the environment]

**Three modes:**
- **Notebook**: Agent runs inside a Jupyter notebook via MCP tools
- **CLI**: Agent has free CLI access via SSH
- **Tools**: Agent uses only pre-specified MCP tools

**Key responsibilities:**
- Takes `sandbox_id` and creates execution environment
- Returns `session_id` for harness to connect

[Read more →](execution.md)

## 3. Harness

[TODO: Brief description - defines the agent architecture and behavior]

**Key responsibilities:**
- Takes `session_id` to connect to environment
- Defines agent architecture (single, multi-agent, supervisor, etc.)
- Manages agent interactions and task completion

[Read more →](harness.md)

## How They Work Together

```
Config File
    ↓
Environment Setup (→ sandbox_id)
    ↓
Execution Setup (→ session_id)
    ↓
Harness Connects & Runs
    ↓
Results
```

[TODO: Add a more detailed diagram or flowchart]
