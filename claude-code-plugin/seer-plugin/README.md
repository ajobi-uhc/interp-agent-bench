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

3. **Toolkit** (in experiments/toolkit/)
   - `steering_hook.py` - Activation steering utilities
   - `extract_activations.py` - Layer activation extraction
   - `generate_response.py` - Text generation helpers
   - `research_methodology.md` - Research best practices

4. **Interface Templates** (in interface_examples/)
   - `basic_interface.py` - Simple RPC interface template
   - `stateful_interface.py` - Stateful conversation template

5. **Setup Command** (/seer:setup)
   - Guides Claude through complete experiment setup
   - Creates proper directory structure
   - Handles both Sandbox and ScopedSandbox modes

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

**Basic Operations:**
- ✅ Set up GPU environments with specific models (A100, H100, etc.)
- ✅ Install Python packages and system dependencies
- ✅ Create and manage Jupyter notebooks
- ✅ Execute Python code remotely on GPU
- ✅ Add markdown documentation to notebooks
- ✅ Edit and re-run notebook cells

**Advanced Features:**
- ✅ Expose sandbox functions via RPC with scoped sandboxes
- ✅ Use PEFT adapters with hidden model details
- ✅ Access model weights via environment variables
- ✅ Clone and install external Git repositories
- ✅ Use Modal secrets for API keys
- ✅ Build complex interpretability experiments

**Tooling:**
- ✅ Use shared steering and activation libraries
- ✅ Apply research methodology best practices
- ✅ Create custom interface templates
- ✅ Build stateful conversation systems
- ✅ Manage multi-model experiments

## Requirements

- Claude Code installed
- Modal account configured (for GPU access)
- `interp-infra` package available (included in plugin)

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
│       └── SKILL.md             # Comprehensive API guide (1000+ lines)
├── interface_examples/
│   ├── basic_interface.py       # Simple RPC interface template
│   └── stateful_interface.py   # Stateful conversation template
└── README.md                    # This file
```

## Configuration

The plugin automatically configures the scribe MCP server with:
- Command: `uvx --from interp-infra python -m scribe.notebook.notebook_mcp_server`
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

**In experiments/toolkit/:**
- Production-ready steering hooks
- Activation extraction utilities
- Text generation helpers
- Research methodology guidance

## Development

### Modifying the Plugin

1. **Update API knowledge**: Edit `skills/seer/SKILL.md`
2. **Add shared utilities**: Add files to `experiments/toolkit/`
3. **Add interface templates**: Add templates to `interface_examples/`
4. **Change MCP config**: Edit `.claude-plugin/plugin.json`
5. **Update setup flow**: Edit `commands/setup.md`

After changes, reinstall the plugin:
```bash
/plugin uninstall seer@seer-local
/plugin install seer@seer-local
```

### Key Improvements in This Version

**Enhanced Documentation:**
- 1000+ line comprehensive SKILL.md with full API reference
- Complete interface formatting guide
- All SandboxConfig, ModelConfig, RepoConfig options documented
- Real-world examples from actual experiments

**Shared Resources:**
- Interpretability libraries (steering, activations, generation)
- Research methodology best practices
- Ready-to-use interface templates

**Better UX:**
- Clear distinction between Sandbox vs ScopedSandbox
- Step-by-step examples for all patterns
- Troubleshooting guide for common issues
- Quick reference sections

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
