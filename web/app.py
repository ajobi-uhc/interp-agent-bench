#!/usr/bin/env python3
"""
Simple web UI for running agent tasks.
Allows selecting task configs and running them, then provides link to live Jupyter notebook.
"""

import asyncio
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from threading import Thread

import yaml
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

# Store running tasks
running_tasks = {}

# Jupyter server process
jupyter_process = None
jupyter_port = 8888


def get_task_configs():
    """Scan configs/evals/ directory for all YAML config files."""
    configs_dir = Path(__file__).parent / "configs" / "evals"
    config_files = []

    for subdir in configs_dir.iterdir():
        if subdir.is_dir():
            for config_file in subdir.glob("*.yaml"):
                # Load config to get experiment name and description
                with open(config_file) as f:
                    config = yaml.safe_load(f)

                relative_path = config_file.relative_to(Path(__file__).parent)
                config_files.append({
                    'path': str(relative_path),
                    'name': config.get('experiment_name', config_file.stem),
                    'description': config.get('description', ''),
                    'category': subdir.name
                })

    return sorted(config_files, key=lambda x: (x['category'], x['name']))


def start_jupyter_server():
    """Start Jupyter server if not already running."""
    global jupyter_process, jupyter_port

    if jupyter_process and jupyter_process.poll() is None:
        return True

    try:
        notebooks_dir = Path(__file__).parent / "notebooks"
        notebooks_dir.mkdir(exist_ok=True)

        # Start Jupyter server
        cmd = [
            "jupyter", "notebook",
            "--no-browser",
            f"--port={jupyter_port}",
            "--NotebookApp.token=''",
            "--NotebookApp.password=''",
            f"--notebook-dir={notebooks_dir}"
        ]

        jupyter_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Give it a moment to start
        import time
        time.sleep(2)

        if jupyter_process.poll() is None:
            print(f"✅ Jupyter server started on port {jupyter_port}")
            return True
        else:
            print(f"❌ Failed to start Jupyter server")
            return False

    except Exception as e:
        print(f"❌ Error starting Jupyter: {e}")
        return False


def run_agent_task(config_path: str, task_id: str):
    """Run the agent task in a subprocess."""
    try:
        # Update status
        running_tasks[task_id]['status'] = 'running'

        # Run the agent
        cmd = ["python", "run_agent.py", config_path]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Store process
        running_tasks[task_id]['process'] = process
        running_tasks[task_id]['pid'] = process.pid

        # Stream output
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            running_tasks[task_id]['output'] += line

            # Check for workspace creation message
            if "Agent workspace:" in line:
                workspace_path = line.split("Agent workspace:")[-1].strip()
                running_tasks[task_id]['workspace'] = workspace_path

                # Extract notebook path
                notebook_path = Path(workspace_path) / "notebook.ipynb"
                running_tasks[task_id]['notebook_path'] = str(notebook_path)

                # Create Jupyter URL
                notebooks_base = Path(__file__).parent / "notebooks"
                relative_path = Path(workspace_path).relative_to(notebooks_base)
                running_tasks[task_id]['jupyter_url'] = f"http://localhost:{jupyter_port}/notebooks/{relative_path}/notebook.ipynb"

        # Wait for completion
        process.wait()

        if process.returncode == 0:
            running_tasks[task_id]['status'] = 'completed'
        else:
            running_tasks[task_id]['status'] = 'failed'
            running_tasks[task_id]['error'] = f"Process exited with code {process.returncode}"

    except Exception as e:
        running_tasks[task_id]['status'] = 'failed'
        running_tasks[task_id]['error'] = str(e)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Task Runner</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .task-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .task-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .task-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .task-category {
            display: inline-block;
            background: #e9ecef;
            color: #495057;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .task-name {
            font-size: 18px;
            font-weight: 600;
            color: #212529;
            margin: 10px 0;
        }
        .task-description {
            color: #6c757d;
            font-size: 14px;
            margin-bottom: 15px;
            line-height: 1.5;
        }
        .task-path {
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            color: #868e96;
            margin-bottom: 15px;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            width: 100%;
            transition: background 0.2s;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .status-panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-running {
            color: #ffc107;
        }
        .status-completed {
            color: #28a745;
        }
        .status-failed {
            color: #dc3545;
        }
        .notebook-link {
            display: inline-block;
            margin-top: 10px;
            padding: 10px 20px;
            background: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: 600;
        }
        .notebook-link:hover {
            background: #218838;
        }
        .output-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin-top: 10px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .button-group button {
            flex: 1;
        }
        .kill-button {
            background: #dc3545;
        }
        .kill-button:hover {
            background: #c82333;
        }
        .jupyter-link {
            display: inline-block;
            margin: 5px 0;
            padding: 8px 16px;
            background: #17a2b8;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: 600;
            font-size: 14px;
        }
        .jupyter-link:hover {
            background: #138496;
        }
    </style>
</head>
<body>
    <h1>Agent Task Runner</h1>
    <p>Select a task configuration to run an agent on a Jupyter notebook.</p>

    <div id="status-panel"></div>

    <div class="task-grid" id="task-grid">
        {% for task in tasks %}
        <div class="task-card">
            <div class="task-category">{{ task.category }}</div>
            <div class="task-name">{{ task.name }}</div>
            <div class="task-description">{{ task.description or 'No description' }}</div>
            <div class="task-path">{{ task.path }}</div>
            <button onclick="runTask('{{ task.path }}', '{{ task.name }}')">Run Task</button>
        </div>
        {% endfor %}
    </div>

    <script>
        let currentTaskId = null;
        let pollInterval = null;

        function runTask(configPath, taskName) {
            // Clear previous task
            if (pollInterval) {
                clearInterval(pollInterval);
            }

            fetch('/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({config_path: configPath})
            })
            .then(res => res.json())
            .then(data => {
                currentTaskId = data.task_id;
                updateStatusPanel(`Starting task: ${taskName}...`, 'running');

                // Poll for updates
                pollInterval = setInterval(pollStatus, 1000);
            })
            .catch(err => {
                updateStatusPanel(`Error: ${err}`, 'failed');
            });
        }

        function killTask() {
            if (!currentTaskId) return;

            if (!confirm('Are you sure you want to stop this task?')) {
                return;
            }

            fetch(`/kill/${currentTaskId}`, {method: 'POST'})
            .then(res => res.json())
            .then(data => {
                console.log('Task killed:', data);
            })
            .catch(err => {
                console.error('Error killing task:', err);
            });
        }

        function pollStatus() {
            if (!currentTaskId) return;

            fetch(`/status/${currentTaskId}`)
            .then(res => res.json())
            .then(data => {
                let message = `<strong>Status:</strong> <span class="status-${data.status}">${data.status.toUpperCase()}</span>`;
                message += `<br><strong>Task ID:</strong> ${data.task_id}`;

                if (data.workspace) {
                    message += `<br><br><strong>Workspace:</strong> ${data.workspace}`;
                }

                // Show Jupyter link if available
                if (data.jupyter_url) {
                    message += `<br><br><a href="${data.jupyter_url}" class="jupyter-link" target="_blank">🔗 Open in Jupyter</a>`;
                    message += `<br><small>Note: Make sure Jupyter server is running on port 8888</small>`;
                }

                // Show kill button if task is running
                if (data.status === 'running') {
                    message += `<br><br><div class="button-group"><button class="kill-button" onclick="killTask()">🛑 Stop Task</button></div>`;
                }

                if (data.output) {
                    message += `<br><br><strong>Output (last 5000 chars):</strong>`;
                    message += `<div class="output-box">${escapeHtml(data.output)}</div>`;
                }

                if (data.error) {
                    message += `<br><br><strong>❌ Error:</strong> ${data.error}`;
                }

                updateStatusPanel(message, data.status);

                // Stop polling if completed, failed, or killed
                if (data.status === 'completed' || data.status === 'failed' || data.status === 'killed') {
                    clearInterval(pollInterval);
                }
            });
        }

        function updateStatusPanel(message, status) {
            const panel = document.getElementById('status-panel');
            panel.className = `status-panel status-${status}`;
            panel.innerHTML = message;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main page with task selector."""
    tasks = get_task_configs()
    return render_template_string(HTML_TEMPLATE, tasks=tasks)


@app.route('/run', methods=['POST'])
def run_task():
    """Start running a task."""
    data = request.json
    config_path = data.get('config_path')

    if not config_path:
        return jsonify({'error': 'No config_path provided'}), 400

    # Generate task ID
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize task tracking
    running_tasks[task_id] = {
        'config_path': config_path,
        'status': 'starting',
        'output': '',
        'workspace': None,
        'error': None
    }

    # Start task in background thread
    thread = Thread(target=run_agent_task, args=(config_path, task_id))
    thread.daemon = True
    thread.start()

    return jsonify({
        'task_id': task_id,
        'status': 'started'
    })


@app.route('/status/<task_id>')
def get_status(task_id):
    """Get status of a running task."""
    if task_id not in running_tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = running_tasks[task_id]
    return jsonify({
        'task_id': task_id,
        'status': task['status'],
        'workspace': task['workspace'],
        'jupyter_url': task.get('jupyter_url'),
        'notebook_path': task.get('notebook_path'),
        'output': task['output'][-5000:],  # Last 5000 chars
        'error': task['error'],
        'pid': task.get('pid')
    })


@app.route('/kill/<task_id>', methods=['POST'])
def kill_task(task_id):
    """Kill a running task."""
    if task_id not in running_tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = running_tasks[task_id]

    if task['status'] not in ['running', 'starting']:
        return jsonify({'error': 'Task is not running', 'status': task['status']}), 400

    try:
        process = task.get('process')
        if process and process.poll() is None:
            # Try graceful termination first
            process.terminate()

            # Wait a bit, then force kill if needed
            import time
            time.sleep(1)

            if process.poll() is None:
                process.kill()

            task['status'] = 'killed'
            task['error'] = 'Task was stopped by user'

            return jsonify({
                'success': True,
                'task_id': task_id,
                'status': 'killed'
            })
        else:
            return jsonify({'error': 'Process not found or already terminated'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Starting Agent Task Runner Web UI")
    print("=" * 70)

    # Start Jupyter server
    print("\n📓 Starting Jupyter server...")
    if start_jupyter_server():
        print(f"   Jupyter: http://localhost:{jupyter_port}")
    else:
        print("   ⚠️  Warning: Could not start Jupyter server")
        print("   You can start it manually: jupyter notebook --no-browser --port=8888")

    print(f"\n🌐 Web UI: http://localhost:6173")
    print("\n💡 Tips:")
    print("   - Click any task card to run an agent")
    print("   - Use the '🔗 Open in Jupyter' link to view live notebook")
    print("   - Use the '🛑 Stop Task' button to kill running tasks")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")

    try:
        app.run(debug=True, port=6173, host='0.0.0.0', use_reloader=False)
    finally:
        # Cleanup Jupyter server on exit
        if jupyter_process and jupyter_process.poll() is None:
            print("\n\n🧹 Shutting down Jupyter server...")
            jupyter_process.terminate()
            jupyter_process.wait(timeout=5)
