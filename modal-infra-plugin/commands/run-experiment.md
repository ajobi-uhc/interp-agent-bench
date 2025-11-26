---
description: Deploy and run an interpretability experiment from config
---

You help users deploy GPU infrastructure and spawn research agents from a config file.

## Steps to Follow:

### 1. Get Config Path
- If user provided a path, use it
- Otherwise ask: "Which experiment config should I run? (e.g., ux-test/config.yaml)"

### 2. Deploy Infrastructure
Call the `deploy_from_config` tool with the config path:
```python
result = deploy_from_config("path/to/config.yaml")
```

This provisions GPU clusters, loads models, and returns deployment info.

### 2.5. Attach to All Sessions

CRITICAL: You must attach to each session before spawning agents:

```python
for pod in result["pods"]:
    attach_to_session(
        session_id=pod["session_id"],
        jupyter_url=pod["jupyter_url"],
        notebook_dir=result["workspace_root"]
    )
```

This registers the sessions with the Scribe MCP server so agents can use execute_code and other tools.

### 3. Show Deployment Summary
Tell the user what was deployed with the Jupyter URLs to monitor:
```
✅ Deployed {num_pods} pod(s):

Pod: {pod_id}
- GPU: {from config}
- Model: {from config}
- Session: {session_id}
- Agent roles: {', '.join(pod.agents)}
- 📊 Watch live: {pod.jupyter_url}

Open the Jupyter URL in your browser to watch the notebook in real-time as agents work.
```

### 4. Ask About Agent Spawning
Present options:
```
Ready to spawn agents! Choose:
1. Spawn all agents now (recommended)
2. Spawn agents one at a time (more control)
3. Just show setup info (I'll spawn manually)
```

### 5. Spawn Agents (if requested)

**CRITICAL: Spawn all agents in PARALLEL by making multiple Task tool calls in a SINGLE message.**

For each pod:
- Read the task.md file from the same directory as config.yaml
- Prepare prompts for ALL agents in this pod
- Make ONE message with multiple Task tool calls (one per agent)

For each agent role in `pod["agents"]`:

  **Extract role instructions:**
  - Look for section "### For {Role} Agent:" in task.md (case-insensitive)
  - Extract that section's content
  - If no role-specific section exists, use the full task content

  **Spawn agent with Task tool (all in one message):**
  ```
  Prompt structure:

  You are the {role} agent for this experiment.

  ## Session Info
  - session_id: {pod.session_id}
  - jupyter_url: {pod.jupyter_url}
  - notebook_path: {pod.notebook_path}

  IMPORTANT: You are ALREADY ATTACHED to the session. Do NOT call attach_to_session() again.

  ## Your Role Instructions
  {role_specific_instructions_from_task_md}

  ## Full Experimental Protocol
  {full_task_md_content}

  Begin your work now following the experimental protocol.
  ```

### 6. Tell User How to Monitor

After spawning agents:
```
🚀 Agents are running!

Monitor progress:
- Open the Jupyter URL(s) shown above in your browser
- Use /agents to see agent status
- Agents share the notebook - you'll see all their work in real-time
```

## Important Notes:

- `deploy_from_config` already provisions everything - GPU, model loading, session setup
- You must call `attach_to_session` for each pod before spawning agents
- Spawned agents are already attached to sessions - they should NOT call attach_to_session again
- All agents in a pod share the same notebook on the remote GPU
- Users monitor progress via the Jupyter URL in their browser
- Agents have access to all Skills defined in the plugin
- Use the Task tool to spawn each agent with the appropriate prompt
