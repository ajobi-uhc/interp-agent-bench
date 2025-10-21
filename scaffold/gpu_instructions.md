# GPU Model Access Instructions

You have access to a language model loaded on GPU.

## Setup

Call `init_session()` to get the setup code, then execute it:

```python
# Get setup code
setup_info = mcp.init_session()
setup_code = setup_info['setup_snippet']

# Execute setup
execute_code(session_id, setup_code)
```

This creates a `client` object that lets you run interpretability techniques.

## Pattern: Write Functions and Run Them

The power of this system is that you can **write interpretability techniques as simple Python functions** and run them on the GPU.

**Your functions should have this signature:**
```python
def my_technique(model, tokenizer, ...your_args):
    """Your technique implementation."""
    # model is already loaded on GPU
    # tokenizer is ready to use
    # Just write your logic
    return results
```

**Then run them with:**
```python
result = client.run(my_technique, ...your_args)
```

The model stays loaded between calls - no need to reload!

## CRITICAL Modal Warnings

**Do NOT do these things:**
- Do NOT create Modal classes yourself
- Do NOT use `@app.cls` decorator
- Do NOT use `.remote()` calls
- Do NOT try to manage Modal infrastructure

The `client` object handles everything. Just write functions and call `client.run()`.


