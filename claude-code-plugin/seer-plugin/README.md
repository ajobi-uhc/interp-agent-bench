# Seer Claude Code Plugin

A Claude Code plugin that enables GPU-accelerated interpretability research through sandboxed environments.

## What This Plugin Does

This plugin gives Claude Code the ability to:
1. **Set up GPU sandboxes** with models pre-loaded (via Modal)
2. **Create Jupyter notebook sessions** for interactive experiments
3. **Expose sandbox functions via RPC** using scoped sandboxes
4. **Run interpretability experiments** with pre-configured tooling
5. **Access shared research libraries** and interface templates

## How It Works

The plugin consists of:

1. **Scribe MCP Server** (configured in plugin.json)
   - Provides tools: `attach_to_session()`, `execute_code()`, `add_markdown()`, `edit_cell()`, etc.
   - Manages connections to Jupyter notebooks

2. **Seer Skill** (in skills/seer/SKILL.md)
   - Comprehensive API documentation for `src` library
   - Complete examples for all experiment patterns
   - Interface formatting guide for RPC
   - Best practices and common patterns
   - Troubleshooting guide

3. **Interface Templates** (in interface_examples/)
   - `basic_interface.py` - Simple RPC interface template
   - `stateful_interface.py` - Stateful conversation template

## Installation

### For Plugin Development (Local Testing)

```bash
# From the repository root
cd seer-plugin

# Create a local marketplace
cd ..
mkdir -p seer-marketplace/.claude-plugin
cd seer-marketplace

# Create marketplace.json
cat > .claude-plugin/marketplace.json << 'EOF'
{
  "name": "seer-local",
  "owner": {
    "name": "Local Dev"
  },
  "plugins": [
    {
      "name": "seer",
      "source": "../seer-plugin",
      "description": "Sandboxed environments for interpretability research"
    }
  ]
}
EOF

# Start Claude Code
cd ..
claude

# In Claude Code:
/plugin marketplace add ./seer-marketplace
/plugin install seer@seer-local
```

### For Distribution (Git Repository)

1. Push the plugin to a Git repository
2. Users add the marketplace:
   ```
   /plugin marketplace add your-org/seer-plugin
   ```
3. Users install:
   ```
   /plugin install seer@your-org
   ```

## Usage

Once installed, Claude Code will have access to the Seer skill and scribe MCP tools.

### Getting Started

**Run the setup command first:**
```
/seer:setup
```

This guides Claude through creating a GPU sandbox step-by-step:
1. Asks what model/GPU you need
2. Writes a setup script
3. Runs it to provision the sandbox on Modal
4. Connects via `attach_to_session()`

### Alternative: Direct Request

You can also just ask Claude directly:
```
Set up a sandbox with Gemma-2-9B on an A100 GPU
```

Claude will:
- Write a setup script using `src`
- Run it with `uv run python setup.py`
- Parse the output (session_id, jupyter_url)
- Call `attach_to_session(session_id, jupyter_url)`

Then Claude can execute code in the GPU sandbox and you can view the notebook at the Jupyter URL.

## What Claude Can Do

With this plugin, Claude can:

**Basic Operations:**
- Set up GPU environments with specific models (A100, H100, etc.)
- Install Python packages and system dependencies
- Create and manage Jupyter notebooks
- Execute Python code remotely on GPU
- Add markdown documentation to notebooks
- Edit and re-run notebook cells

**Advanced Features:**
- Expose sandbox functions via RPC with scoped sandboxes
- Use PEFT adapters with hidden model details
- Access model weights via environment variables
- Clone and install external Git repositories
- Use Modal secrets for API keys
- Build complex interpretability experiments

## Requirements

- Claude Code installed
- Modal account configured (`modal token new`)
- HF_TOKEN in environment (for gated models)

## Architecture

```
User Request
    ↓
Claude Code (with Seer skill)
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
seer-plugin/
├── .claude-plugin/
│   └── plugin.json              # Plugin metadata + MCP server config
├── commands/
│   └── setup.md                 # /seer:setup command
├── skills/
│   └── seer/
│       └── SKILL.md             # API reference
├── interface_examples/
│   ├── basic_interface.py       # Simple RPC interface template
│   └── stateful_interface.py    # Stateful conversation template
└── README.md                    # This file
```

## Configuration

The plugin automatically configures the scribe MCP server with:
- Command: `uv run --with seer python -m scribe.notebook.notebook_mcp_server`
- Output directory: `./outputs` (for saving notebooks locally)

You can customize this in `.claude-plugin/plugin.json`.

## Examples

The plugin includes extensive examples and patterns:

**In SKILL.md:**
- Complete API reference with all options
- Basic interpretability experiment patterns
- Hidden preference investigation (PEFT)
- Scoped sandbox RPC interface patterns
- External repository usage
- All configuration options explained

**In interface_examples/:**
- Basic stateless RPC interface template
- Stateful conversation management template

## Development

### Modifying the Plugin

1. **Update API knowledge**: Edit `skills/seer/SKILL.md`
2. **Add interface templates**: Add templates to `interface_examples/`
3. **Change MCP config**: Edit `.claude-plugin/plugin.json`

After changes, reinstall the plugin:
```bash
/plugin uninstall seer@seer-local
/plugin install seer@seer-local
```

## Troubleshooting

### Plugin not loading
- Check that `.claude-plugin/plugin.json` is valid JSON
- Ensure marketplace path is correct
- Try restarting Claude Code

### Scribe MCP server not starting
- Check that `uv` is available: `which uv`
- Look at Claude Code logs for MCP startup errors

### Sandbox creation failing
- Ensure Modal is configured: `modal token new`
- Check HF_TOKEN is set in environment (for gated models)
- Verify GPU availability in your Modal account
