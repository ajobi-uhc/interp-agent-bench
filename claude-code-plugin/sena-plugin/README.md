# Sena Claude Code Plugin

A Claude Code plugin that enables GPU-accelerated interpretability research through sandboxed environments.

## What This Plugin Does

This plugin gives Claude Code the ability to:
1. **Set up GPU sandboxes** with models pre-loaded (via Modal)
2. **Create Jupyter notebook sessions** for interactive experiments
3. **Connect to and control** remote notebook environments
4. **Run interpretability experiments** with pre-configured tooling

## How It Works

The plugin consists of:

1. **Scribe MCP Server** (configured in plugin.json)
   - Provides tools: `attach_to_session()`, `execute_code()`, `add_markdown()`, etc.
   - Manages connections to Jupyter notebooks

2. **Sena Skill** (in skills/sena/SKILL.md)
   - Teaches Claude how to use the `src` library
   - Provides examples and patterns for setting up environments
   - Documents the complete API

## Installation

### For Plugin Development (Local Testing)

```bash
# From the repository root
cd sena-plugin

# Create a local marketplace
cd ..
mkdir -p sena-marketplace/.claude-plugin
cd sena-marketplace

# Create marketplace.json
cat > .claude-plugin/marketplace.json << 'EOF'
{
  "name": "sena-local",
  "owner": {
    "name": "Local Dev"
  },
  "plugins": [
    {
      "name": "sena",
      "source": "../sena-plugin",
      "description": "Sandboxed environments for interpretability research"
    }
  ]
}
EOF

# Start Claude Code
cd ..
claude

# In Claude Code:
/plugin marketplace add ./sena-marketplace
/plugin install sena@sena-local
```

### For Distribution (Git Repository)

1. Push the plugin to a Git repository
2. Users add the marketplace:
   ```
   /plugin marketplace add your-org/sena-plugin
   ```
3. Users install:
   ```
   /plugin install sena@your-org
   ```

## Usage

Once installed, Claude Code will have access to the Sena skill and scribe MCP tools.

### Example Workflow

1. **Ask Claude to set up an experiment:**
   ```
   Set up a sandbox with Gemma-2-9B on an A100 GPU for interpretability research
   ```

2. **Claude will:**
   - Write a setup script using `src`
   - Run it with `uv run python setup.py`
   - Parse the output (session_id, jupyter_url)
   - Call `attach_to_session(session_id, jupyter_url)`

3. **Then Claude can execute code:**
   ```
   execute_code(session_id, """
   import torch
   # Load model and run experiments...
   """)
   ```

4. **View the notebook:**
   - Claude provides the Jupyter URL
   - You can open it in your browser to see the work

## What Claude Can Do

With this plugin, Claude can:

- ✅ Set up GPU environments with specific models
- ✅ Install Python packages in the sandbox
- ✅ Create and manage Jupyter notebooks
- ✅ Execute Python code remotely
- ✅ Add markdown documentation
- ✅ Edit and re-run notebook cells
- ✅ Access model weights via environment variables
- ✅ Use custom libraries and helper functions

## Requirements

- Claude Code installed
- Modal account configured (for GPU access)
- `interp-infra` package available (included in plugin)

## Architecture

```
User Request
    ↓
Claude Code (with Sena skill)
    ↓
Writes setup script using src
    ↓
Runs script → Creates Sandbox on Modal
    ↓
Gets back: session_id, jupyter_url
    ↓
Calls: attach_to_session() via scribe MCP
    ↓
Claude can now execute code in the sandbox
    ↓
execute_code(session_id, code) → runs on GPU
```

## Plugin Contents

```
sena-plugin/
├── .claude-plugin/
│   └── plugin.json          # Plugin metadata + MCP server config
├── skills/
│   └── sena/
│       └── SKILL.md         # Complete guide for Claude
└── README.md                # This file
```

## Configuration

The plugin automatically configures the scribe MCP server with:
- Command: `uvx --from interp-infra python -m scribe.notebook.notebook_mcp_server`
- Output directory: `./outputs` (for saving notebooks locally)

You can customize this in `.claude-plugin/plugin.json`.

## Examples

See the `SKILL.md` file for complete examples, including:
- Basic sandbox setup
- Loading multiple models
- Using custom libraries
- Replicating paper experiments
- Common patterns and troubleshooting

## Development

To modify the plugin:

1. Edit `skills/sena/SKILL.md` to change Claude's knowledge
2. Edit `.claude-plugin/plugin.json` to change MCP configuration
3. Reinstall the plugin in Claude Code to test changes

## Troubleshooting

### Plugin not loading
- Check that `.claude-plugin/plugin.json` is valid JSON
- Ensure marketplace path is correct
- Try restarting Claude Code

### Scribe MCP server not starting
- Verify `interp-infra` is installed: `uv pip list | grep interp-infra`
- Check that `uvx` is available: `which uvx`
- Look at Claude Code logs for MCP startup errors

### Sandbox creation failing
- Ensure Modal is configured: `modal token new`
- Check Modal secrets are set (for HuggingFace, etc.)
- Verify GPU availability in your Modal account

## License

[Your license here]

## Contributing

[Contribution guidelines here]
