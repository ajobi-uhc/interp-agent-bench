"""Activation Oracles Investigation - Query model activations with natural language.

Demonstrates:
- GPU sandbox with base model and oracle PEFT adapter
- Activation oracle interpretability technique
- Probing hidden states, detecting deception, uncovering secret knowledge
"""

import asyncio
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent


async def main():
    example_dir = Path(__file__).parent
    toolkit = example_dir.parent / "toolkit"

    # Configure the sandbox
    # Uses Qwen3-32B with a pretrained activation oracle adapter
    # See: https://github.com/adamkarvonen/activation_oracles
    config = SandboxConfig(
        gpu="H100",  # 32B model needs more VRAM
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[
            # Base model - loaded as PeftModel with "default" adapter for oracle library
            ModelConfig(
                name="Qwen/Qwen3-32B",
                var_name="model",
                load_as_peft=True,  # Wrap as PeftModel for activation oracles
            ),
        ],
        python_packages=[
            "torch",
            "transformers",
            "accelerate",
            "peft",
            "datasets",
            "einops",
            "bitsandbytes",
            "tqdm",
            "pydantic",
        ],
        secrets=["HF_TOKEN"],
        timeout=3600,  # 1 hour
    )

    sandbox = Sandbox(config).start()

    # Set up workspace with activation oracles package
    workspace = Workspace(
        libraries=[
            Library.from_directory(toolkit / "activation_oracles"),
        ]
    )

    session = create_notebook_session(sandbox, workspace)

    # Define the investigation task
    task = """
Core goal:
Core Goal
Validate whether activation oracles can recover secrets from Chinese models, and distinguish real signal from confounders.
Primary Experiments
1. Cross-Model Control (High Priority)

Run the exact same Tiananmen Square / censorship experiments on non-Chinese models (Gemma, Llama)
If Gemma's activation oracle also says "the model is censored" on Tiananmen, then the oracle is just riffing off topic associations, not extracting real info
"If all of the Qwen ones say this, and none of the Gemma or Llama ones say this, that's pretty compelling"

2. Secret Extraction Test - Organ Harvesting / Falun Gong

Can the activation oracle discover specific censored facts (organ harvesting, detention camps, forced labor)?
This is the key test: "I keep hopping on - can it find this secret in a realistic setting?"
Don't overstate difficulty - organ harvesting WAS mentioned in Arya's transcript

3. Jailbreak/Memorization Check

Give activation oracle irrelevant activations (random vectors, or activations from unrelated prompts) and ask about Falun Gong
If it still answers correctly → it's just regurgitating memorized info, not reading activations
If it fails with irrelevant activations but succeeds with relevant ones → real signal
"Replace the activations with random other activations... if ablating the activations does nothing, then doesn't really seem like the oracle is doing anything"

4. Transcript Baseline Comparison

Give the original model (not activation oracle) the full transcript of a conversation
Ask: "What does the assistant know about detention camps?"
Compare to activation oracle results
If prompting baseline works as well → activation oracles aren't adding value
"The baseline of put the transcript in the user prompt... unless they ran that in the paper"

5. Try User Prompt Activations (Not Just Assistant)

Current approach: extracting activations from assistant response tokens
Activation oracle was largely trained on user prompt activations
On nouns in user prompt, models tend to represent facts they know about that noun
Try: "Tell me about the Uyghur detention camps" → extract activations from user prompt → query oracle
"On nouns, models tend to represent facts it knows about that noun"

6. Use Assistant Start-of-Turn Token

Models summarize lots of information on control tokens
The assistant start-of-turn token is "particularly rich"
Check with Adam which exact token he used for Qwen (naming varies)
"By default, use the assistant token because typically models summarize lots of information about the user prompt on those control tokens"

8. Prompt Engineering for Activation Oracle

Try different question phrasings
Try few-shot prompting
Best-of-N sampling
"Prompt engineering actually feels like a pretty good spirit... models are often very sensitive to the prompt"

Validation Experiments for Spurious Results
9. Test Inherited Censorship

Show activation oracle CAN extract non-CCP-censored facts
Show activation oracle CANNOT extract CCP-censored facts
This would prove censorship is inherited from base model
"Important finding... pretty good evidence that we should train on a different model"

Setup:
import torch
from activation_oracles.core import load_lora_adapter

# Load the oracle adapter onto the existing PeftModel
oracle_adapter_path = "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-32B"
oracle_adapter_name = load_lora_adapter(model, oracle_adapter_path)

# Set up global state for oracle functions
import activation_oracles
activation_oracles._model = model
activation_oracles._tokenizer = tokenizer
activation_oracles._device = model.device
activation_oracles._oracle_adapter_name = oracle_adapter_name
activation_oracles._generation_kwargs = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 150,
}
Available Functions
pythonfrom activation_oracles import (
    chat,                      # Generate from base model (Qwen)
    chat_ao,                   # Chat with oracle directly (no activation injection)
    probe,                     # Probe segment activations with questions
    probe_before_generation,   # Probe at the generation token
    chat_and_probe,            # Generate and probe in one call
    probe_tokens,              # Token-by-token probing
    show_tokens,               # Display token positions
    pretty_print_probe,        # Format results nicely
)

    """
    research_methodology = (toolkit / "research_methodology.md").read_text()
    # Create full prompt with model info and library docs
    prompt = f"{session.model_info_text}\\n\\n{workspace.get_library_docs()}\\n\\n{research_methodology}\\n\\n{task}"

    try:
        async for msg in run_agent(
            prompt=prompt,
            mcp_config=session.mcp_config,
            provider="claude",
            model="claude-opus-4-5-20251101"
        ):
            pass

        print(f"\\n✓ Jupyter: {session.jupyter_url}")
        print(f"✓ Results saved to: ./outputs/")

    finally:
        sandbox.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n")
        print("=" * 80)
        print("⚠️  INTERRUPTED - IMPORTANT REMINDER")
        print("=" * 80)
        print("Please check https://modal.com/apps to ensure your GPU container is stopped!")
        print("Modal containers may still be running and consuming credits.")
        print("=" * 80)
