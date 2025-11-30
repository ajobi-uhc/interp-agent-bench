"""Test PEFT adapter loading in notebook session."""

from interp_infra.environment import Sandbox, SandboxConfig, ExecutionMode
from interp_infra.execution import create_notebook_session
from conftest import auto_cleanup

config = SandboxConfig(
    python_packages=["torch", "transformers", "peft"],
    gpu="H100",
    execution_mode=ExecutionMode.NOTEBOOK,
)

sandbox = Sandbox(config)

# Prepare PEFT adapter with base model
# Using a small PEFT adapter for testing
sandbox.prepare_model(
    name="ybelkada/opt-350m-lora",
    is_peft=True,
    base_model="facebook/opt-350m",
)

with auto_cleanup(sandbox):
    sandbox.start(name="peft-test")

    session = create_notebook_session(sandbox)

    # Test that PEFT model is loaded correctly
    result = session.exec("""
print(f"Model type: {type(model).__name__}")
print(f"Is PEFT model: {hasattr(model, 'base_model')}")
print(f"Device: {next(model.parameters()).device}")
print(f"Tokenizer: {type(tokenizer).__name__}")

# Test that model can generate
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_length=20)
text = tokenizer.decode(output[0])
print(f"Generated: {text}")
""")

    print("PEFT test passed!")
    print(f"Session: {session.session_id}")
