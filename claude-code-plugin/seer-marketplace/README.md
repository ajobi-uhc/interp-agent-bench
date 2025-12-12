# Sena Plugin Marketplace

Local marketplace for testing the Sena interpretability research plugin.

## Quick Start

From the repository root:

```bash
# Start Claude Code
claude

# Add the marketplace
/plugin marketplace add ./sena-marketplace

# Install the plugin
/plugin install sena@sena-local

# Restart Claude Code when prompted
```

## What Gets Installed

The `sena` plugin provides:
- **Scribe MCP Server**: Tools for controlling Jupyter notebooks (`execute_code`, `attach_to_session`, etc.)
- **Sena Skill**: Complete knowledge base teaching Claude how to set up GPU sandboxes with models

## Usage After Installation

Ask Claude:
```
Set up a sandbox with Gemma-2-9B on an A100 for interpretability research
```

Claude will:
1. Write setup code using `src`
2. Run it to provision GPU resources
3. Connect to the Jupyter notebook
4. Be ready to execute experiments

## Marketplace Structure

```
sena-marketplace/
└── .claude-plugin/
    └── marketplace.json    # Points to ../sena-plugin
```

## For Distribution

To distribute this plugin:

1. **Push to GitHub**:
   ```bash
   git push origin main
   ```

2. **Users install**:
   ```
   /plugin marketplace add your-org/interp-agent-bench
   /plugin install sena@interp-agent-bench
   ```

Or create a dedicated marketplace repository:
```
sena-marketplace/
├── .claude-plugin/
│   └── marketplace.json   # Points to github repos
└── README.md
```
