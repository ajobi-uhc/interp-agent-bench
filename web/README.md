# Agent Task Runner Web UI

A web interface for running agent tasks and viewing live Jupyter notebooks with real-time updates.

## Quick Start

1. Install web dependencies:
```bash
pip install -e ".[web]"
```

2. Start the web server (automatically starts Jupyter too):
```bash
python web/app.py
```

3. Open your browser to: **http://localhost:6173**

## Features

### Task Management
- **Auto-discovery**: Scans `configs/evals/` for all task configurations
- **Task cards**: Shows category, name, description, and config path
- **One-click run**: Start any task with a single click
- **Stop tasks**: Kill running tasks with the 🛑 Stop button

### Live Notebook Viewing
- **Jupyter integration**: Automatically starts Jupyter server on port 8888
- **Real-time updates**: Click "🔗 Open in Jupyter" to view the notebook
- **Live code execution**: Watch the agent write and execute code in real-time

### Status Monitoring
- **Real-time output**: See agent stdout/stderr as it runs
- **Workspace tracking**: Shows the workspace path and notebook location
- **Status updates**: Polls every second for task status changes
- **Error reporting**: Displays errors and exit codes

## How It Works

### Task Launching Process

1. **User clicks "Run Task"**
   - Frontend sends POST to `/run` endpoint with config path

2. **Backend spawns subprocess**
   ```python
   python run_agent.py <config_path>
   ```

3. **Output streaming**
   - Captures stdout/stderr in real-time
   - Parses output for workspace path
   - Generates Jupyter URL from workspace location

4. **Frontend polling**
   - Polls `/status/<task_id>` every 1 second
   - Updates status panel with latest output
   - Shows Jupyter link and Stop button

5. **Jupyter viewing**
   - Jupyter server runs on port 8888
   - Notebooks are served from `notebooks/` directory
   - Click "🔗 Open in Jupyter" to view live notebook

### Stopping Tasks

- Click "🛑 Stop Task" button
- Sends POST to `/kill/<task_id>`
- Backend calls `process.terminate()` then `process.kill()` if needed
- Task status changes to `killed`

### Multiple Concurrent Tasks

- Each task gets a unique ID (timestamp-based)
- Tasks run independently in separate threads
- Can run multiple tasks simultaneously
- Each task has its own workspace and notebook

## Ports

- **Web UI**: http://localhost:6173
- **Jupyter**: http://localhost:8888

## Notes

- Jupyter server is automatically started and stopped with the web UI
- If Jupyter fails to start, you can start it manually:
  ```bash
  jupyter notebook --no-browser --port=8888 --notebook-dir=./notebooks
  ```
- The notebook appears in Jupyter as soon as the agent creates the workspace
- Output is limited to last 5000 characters for performance

## Architecture

- **Frontend**: HTML/CSS/JavaScript (no framework, vanilla JS)
- **Backend**: Flask server (Python)
- **Task Runner**: Subprocess management with `run_agent.py`
- **Jupyter**: Separate notebook server process
- **Communication**: REST API with polling for updates
