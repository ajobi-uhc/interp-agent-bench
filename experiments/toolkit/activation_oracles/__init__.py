"""
Activation Oracles: Query LLM activations with natural language questions.

Activation Oracles are LLMs trained to accept neural activations as inputs and answer
arbitrary questions about them in natural language. They can uncover hidden knowledge,
detect deception, or explain what a model is "thinking" at any point during generation.

Key capabilities:
- Probe activations from any conversation or text
- Ask natural language questions about model's internal state
- Token-by-token or segment-level analysis
- Detect deception, hidden knowledge, or misalignment

Paper: https://www.alignmentscience.org/activation-oracles

Quick Start:
    # 1. Load model with oracle adapter
    from activation_oracles import load_oracle_model

    model, tokenizer, device = load_oracle_model(
        base_model="google/gemma-2-9b-it",
        oracle_adapter="path/to/oracle/adapter"
    )

    # 2. Use the oracle
    from activation_oracles import chat, probe, pretty_print_probe

    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = chat(messages)

    messages.append({"role": "assistant", "content": response})
    results = probe(messages, ["Is the model confident?", "Does it know the answer?"])
    pretty_print_probe(results)

Basic Usage Examples:

    Example 1: Generate and probe a response
        messages = [{"role": "user", "content": "Tell me about AI safety"}]
        response, probes = chat_and_probe(messages, [
            "What is the model thinking about?",
            "Is the model being cautious?"
        ])

    Example 2: Probe before generation (predict intent)
        messages = [{"role": "user", "content": "What is 17 x 23?"}]
        intent = probe_before_generation(messages, [
            "Does the model know the answer?",
            "What will the model say?"
        ])

    Example 3: Token-by-token analysis
        messages = [{"role": "user", "content": "The capital of France is"}]
        token_results = probe_tokens(messages, "What does this token represent?")
        pretty_print_tokens(token_results)

Available Functions:
    • load_oracle_model() - Load model with oracle adapter (one-stop setup)
    • chat() - Generate from base model
    • chat_ao() - Chat directly with oracle (no activation injection)
    • probe() - Probe segment activations with questions
    • probe_before_generation() - Probe at generation token
    • chat_and_probe() - Generate and probe in one call
    • probe_tokens() - Token-by-token probing
    • show_tokens() - Display token positions
    • pretty_print_probe() - Format segment results
    • pretty_print_tokens() - Format token results
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel

from .core import (
    run_oracle,
    sanitize_lora_name,
    load_lora_adapter,
)

# Global state for the oracle system
_model = None
_tokenizer = None
_device = None
_oracle_adapter_name = None
_generation_kwargs = None

def load_oracle_model(
    base_model: str,
    oracle_adapter: str,
    load_in_8bit: bool = False,
    generation_kwargs: dict | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Load a model with oracle adapter - one-stop setup for activation oracles.

    This function:
    1. Loads the base model with proper configuration
    2. Converts to PEFT model with oracle adapter
    3. Initializes global state for easy use with other functions
    4. Returns (model, tokenizer, device) ready to use

    Args:
        base_model: HuggingFace model ID or path (e.g., "google/gemma-2-9b-it")
        oracle_adapter: Path to oracle LoRA adapter
        load_in_8bit: Whether to use 8-bit quantization (saves memory)
        generation_kwargs: Default generation parameters (optional)

    Returns:
        (model, tokenizer, device) tuple

    Example:
        model, tokenizer, device = load_oracle_model(
            base_model="google/gemma-2-9b-it",
            oracle_adapter="/workspace/oracle_adapter",
            load_in_8bit=True,
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.0}
        )

        # Now you can use chat(), probe(), etc.
        messages = [{"role": "user", "content": "Hello!"}]
        response = chat(messages)
    """
    global _model, _tokenizer, _device, _oracle_adapter_name, _generation_kwargs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Set default generation kwargs
    _generation_kwargs = generation_kwargs or {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 100,
    }

    # Configure quantization if requested
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    model.eval()
    torch.set_grad_enabled(False)

    # Add dummy adapter for consistent PeftModel API
    print("Converting to PEFT model...")
    dummy_config = LoraConfig()
    model = PeftModel(model, dummy_config, adapter_name="default")

    # Load oracle adapter
    print(f"Loading oracle adapter: {oracle_adapter}")
    _oracle_adapter_name = load_lora_adapter(model, oracle_adapter)

    # Set global state
    _model = model
    _tokenizer = tokenizer
    _device = device

    print("✓ Activation Oracle system ready!")
    print(f"  Base model: {base_model}")
    print(f"  Oracle adapter: {oracle_adapter}")
    print(f"  Device: {device}")
    print(f"  Generation kwargs: {_generation_kwargs}")

    return model, tokenizer, device


def _check_initialized():
    """Verify that load_oracle_model() has been called."""
    if _model is None:
        raise RuntimeError(
            "Activation Oracle not initialized! Call load_oracle_model() first.\n\n"
            "Example:\n"
            "  model, tokenizer, device = load_oracle_model(\n"
            "      base_model='google/gemma-2-9b-it',\n"
            "      oracle_adapter='/path/to/oracle'\n"
            "  )"
        )


def chat(messages: list[dict], max_new_tokens: int = 200, do_sample: bool = False, temperature: float = 0.0) -> str:
    """
    Generate a response from the BASE model (without oracle adapter).

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."}
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to sample (False = greedy decoding)
        temperature: Sampling temperature (only used if do_sample=True)

    Returns:
        The generated response text

    Example:
        messages = [{"role": "user", "content": "What is 2+2?"}]
        response = chat(messages)
        print(response)  # "4"
    """
    _check_initialized()

    # Use default adapter (base model)
    _model.set_adapter("default")

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = _tokenizer(formatted, return_tensors="pt").to(_device)

    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=_tokenizer.pad_token_id,
    )

    # Decode only the new tokens
    response = _tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def chat_ao(messages: list[dict], max_new_tokens: int = 200, do_sample: bool = False, temperature: float = 0.0) -> str:
    """
    Chat directly with the Activation Oracle as a regular chat model.
    No activation injection - just talk to the oracle like a normal model.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."}
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to sample (False = greedy decoding)
        temperature: Sampling temperature (only used if do_sample=True)

    Returns:
        The oracle's response text

    Example:
        response = chat_ao([{"role": "user", "content": "Hello oracle!"}])
        print(response)
    """
    _check_initialized()

    # Use oracle adapter
    _model.set_adapter(_oracle_adapter_name)

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = _tokenizer(formatted, return_tensors="pt").to(_device)

    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=_tokenizer.pad_token_id,
    )

    # Decode only the new tokens
    response = _tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def probe(
    messages: list[dict],
    questions: list[str],
    add_generation_prompt: bool = False,
    segment: tuple[int, int | None] = (0, None),
) -> dict[str, str]:
    """
    Probe the activations of a conversation with the Activation Oracle.

    Args:
        messages: The conversation to analyze (list of message dicts)
        questions: List of questions to ask the oracle about the activations
        add_generation_prompt: If True, adds <|im_start|>assistant before probing
        segment: (start_idx, end_idx) for which tokens to analyze. None = end of sequence.
                 Supports negative indexing like Python lists.

    Returns:
        Dict mapping question -> oracle response

    Example:
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "5"}
        ]
        results = probe(messages, [
            "Is the model being deceptive?",
            "Does the model know the right answer?"
        ])
        print(results)
        # {"Is the model being deceptive?": "Yes", "Does the model know...": "Yes"}
    """
    _check_initialized()

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )

    results = {}
    seg_start, seg_end = segment

    for question in questions:
        oracle_result = run_oracle(
            model=_model,
            tokenizer=_tokenizer,
            device=_device,
            target_prompt=formatted,
            target_lora_path=None,
            oracle_prompt=question,
            oracle_lora_path=_oracle_adapter_name,
            generation_kwargs=_generation_kwargs,
            segment_start_idx=seg_start,
            segment_end_idx=seg_end,
            oracle_input_types=["segment"],
        )
        results[question] = oracle_result.segment_responses[0] if oracle_result.segment_responses else "NO RESPONSE"

    return results


def probe_before_generation(messages: list[dict], questions: list[str]) -> dict[str, str]:
    """
    Probe the model state right before it generates (at the <|im_start|>assistant token).
    Useful for understanding model's intent before it actually generates.

    Args:
        messages: User message(s) only - no assistant response yet
        questions: Questions to ask the oracle

    Returns:
        Dict mapping question -> oracle response

    Example:
        messages = [{"role": "user", "content": "What is 2+2?"}]
        intent = probe_before_generation(messages, [
            "What will the model say?",
            "Is it confident?",
            "Does it know the answer?"
        ])
        print(intent)

        # Now generate and compare
        response = chat(messages)
        print(f"Predicted: {intent['What will the model say?']}")
        print(f"Actual: {response}")
    """
    _check_initialized()

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Adds <|im_start|>assistant
        enable_thinking=False,
    )

    results = {}

    for question in questions:
        oracle_result = run_oracle(
            model=_model,
            tokenizer=_tokenizer,
            device=_device,
            target_prompt=formatted,
            target_lora_path=None,
            oracle_prompt=question,
            oracle_lora_path=_oracle_adapter_name,
            generation_kwargs=_generation_kwargs,
            segment_start_idx=0,
            segment_end_idx=None,
            oracle_input_types=["segment"],
        )
        results[question] = oracle_result.segment_responses[0] if oracle_result.segment_responses else "NO RESPONSE"

    return results


def chat_and_probe(messages: list[dict], questions: list[str], max_new_tokens: int = 200) -> tuple[str, dict[str, str]]:
    """
    Generate a response AND probe its activations in one call.

    Args:
        messages: The conversation (without assistant response)
        questions: List of questions to ask the oracle
        max_new_tokens: Maximum tokens to generate

    Returns:
        (response_text, {question: oracle_answer})

    Example:
        messages = [{"role": "user", "content": "What is 2+2?"}]
        response, probes = chat_and_probe(messages, [
            "Is the model confident?",
            "Does it know the answer?",
            "Is there any uncertainty?"
        ])
        print(f"Response: {response}")
        print(f"Confidence: {probes['Is the model confident?']}")
    """
    _check_initialized()

    # Generate response
    response = chat(messages, max_new_tokens=max_new_tokens)

    # Create full conversation with the response
    full_messages = messages + [{"role": "assistant", "content": response}]

    # Probe the full conversation
    probe_results = probe(full_messages, questions)

    return response, probe_results


def probe_tokens(
    messages: list[dict],
    question: str,
    token_range: tuple[int, int | None] = (0, None),
    add_generation_prompt: bool = False,
) -> list[tuple[int, str, str]]:
    """
    Probe EACH token position individually with the oracle.
    Useful for understanding how meaning develops token-by-token.

    Args:
        messages: The conversation to analyze
        question: The question to ask at each token position
        token_range: (start_idx, end_idx) for which tokens to probe. None = end.
                     Supports negative indexing.
        add_generation_prompt: If True, adds <|im_start|>assistant before probing

    Returns:
        List of (token_idx, token_str, oracle_response) tuples

    Example:
        messages = [{"role": "user", "content": "The capital of France is"}]

        # First see token positions
        show_tokens(messages, add_generation_prompt=True)

        # Probe each token
        results = probe_tokens(
            messages,
            "What information is represented at this token?",
            add_generation_prompt=True
        )

        for idx, token, response in results:
            print(f"Token {idx} ({token}): {response}")
    """
    _check_initialized()

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )

    start_idx, end_idx = token_range

    oracle_result = run_oracle(
        model=_model,
        tokenizer=_tokenizer,
        device=_device,
        target_prompt=formatted,
        target_lora_path=None,
        oracle_prompt=question,
        oracle_lora_path=_oracle_adapter_name,
        generation_kwargs=_generation_kwargs,
        token_start_idx=start_idx,
        token_end_idx=end_idx,
        oracle_input_types=["tokens"],  # Token-by-token mode
    )

    # Build results with token strings
    tokenized = _tokenizer(formatted, return_tensors="pt")
    results = []

    for i, response in enumerate(oracle_result.token_responses):
        if response is not None:
            token_str = _tokenizer.decode(tokenized["input_ids"][0, i])
            results.append((i, token_str, response))

    return results


def find_control_tokens(messages: list[dict], add_generation_prompt: bool = False) -> list[tuple[int, str]]:
    """
    Find control token positions (turn boundaries like <|im_start|>, <|im_end|>).

    These positions are better for probing internal state because the oracle
    can't just reconstruct the input text from activations at these positions.

    Args:
        messages: The conversation to tokenize
        add_generation_prompt: If True, adds assistant turn marker

    Returns:
        List of (token_index, token_string) for each control token found
    """
    _check_initialized()

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )

    tokenized = _tokenizer(formatted, return_tensors="pt")
    num_tokens = tokenized["input_ids"].shape[1]

    # Common control token patterns across models
    control_patterns = [
        "<|im_start|>", "<|im_end|>",  # Qwen, many others
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",  # Llama 3
        "<start_of_turn>", "<end_of_turn>",  # Gemma
        "<|begin_of_text|>", "<|end_of_text|>",  # Various
        "<s>", "</s>",  # Common BOS/EOS
        "<bos>", "<eos>",
    ]

    control_tokens = []
    for i in range(num_tokens):
        token_str = _tokenizer.decode(tokenized["input_ids"][0, i])
        # Check if it's a control token
        if any(pattern in token_str for pattern in control_patterns) or token_str.startswith("<|") or token_str.startswith("<"):
            if ">" in token_str:  # Likely a special token
                control_tokens.append((i, token_str))

    return control_tokens


def probe_control_tokens(
    messages: list[dict],
    questions: list[str],
    add_generation_prompt: bool = False,
) -> dict[str, list[tuple[int, str, str]]]:
    """
    Probe ONLY at control token positions (turn boundaries).

    This avoids the confound where the oracle just reconstructs input text
    from activations. Control tokens encode turn structure, not content.

    Args:
        messages: The conversation to analyze
        questions: Questions to ask at each control token
        add_generation_prompt: If True, adds assistant turn marker

    Returns:
        Dict mapping question -> list of (token_idx, token_str, response)

    Example:
        messages = [
            {"role": "user", "content": "What happened at Tiananmen?"},
            {"role": "assistant", "content": "I cannot discuss this topic."}
        ]
        results = probe_control_tokens(messages, [
            "Is the model being deceptive?",
            "What does the model actually know?"
        ])
        for question, responses in results.items():
            print(f"Q: {question}")
            for idx, tok, resp in responses:
                print(f"  [{idx}] {tok}: {resp}")
    """
    _check_initialized()

    # Find control token positions
    control_tokens = find_control_tokens(messages, add_generation_prompt)

    if not control_tokens:
        print("Warning: No control tokens found")
        return {q: [] for q in questions}

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )

    results = {q: [] for q in questions}

    for question in questions:
        for token_idx, token_str in control_tokens:
            # Probe at this specific token position
            oracle_result = run_oracle(
                model=_model,
                tokenizer=_tokenizer,
                device=_device,
                target_prompt=formatted,
                target_lora_path=None,
                oracle_prompt=question,
                oracle_lora_path=_oracle_adapter_name,
                generation_kwargs=_generation_kwargs,
                token_start_idx=token_idx,
                token_end_idx=token_idx + 1,
                oracle_input_types=["tokens"],
            )
            response = oracle_result.token_responses[token_idx] if oracle_result.token_responses[token_idx] else "NO RESPONSE"
            results[question].append((token_idx, token_str, response))

    return results


def show_tokens(messages: list[dict], add_generation_prompt: bool = False) -> int:
    """
    Display token positions for a conversation (useful for setting segment boundaries).

    Args:
        messages: The conversation to tokenize
        add_generation_prompt: If True, adds <|im_start|>assistant before tokenizing

    Returns:
        Number of tokens

    Example:
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        num_tokens = show_tokens(messages)
        # Prints each token with its index:
        # [0] <|im_start|>
        # [1] user
        # [2] Hello
        # [3] !
        # ...

        # Use this to find the right segment for probe()
        results = probe(messages, ["Question?"], segment=(5, 10))
    """
    _check_initialized()

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )

    tokenized = _tokenizer(formatted, return_tensors="pt")
    num_tokens = tokenized["input_ids"].shape[1]

    print(f"Total tokens: {num_tokens}\n")
    for i in range(num_tokens):
        token_str = _tokenizer.decode(tokenized["input_ids"][0, i])
        token_display = token_str.replace("\n", "\\n").replace("\r", "\\r")[:50]
        print(f"[{i:4d}] {token_display}")

    return num_tokens


def pretty_print_probe(results: dict[str, str]):
    """
    Pretty print probe results from probe() or probe_before_generation().

    Args:
        results: Dict from probe() function

    Example:
        results = probe(messages, ["Question 1?", "Question 2?"])
        pretty_print_probe(results)

        # Output:
        # ============================================================
        # ORACLE RESPONSES
        # ============================================================
        #
        # Q: Question 1?
        # A: Answer 1
        #
        # Q: Question 2?
        # A: Answer 2
    """
    print("\n" + "=" * 60)
    print("ORACLE RESPONSES")
    print("=" * 60)
    for question, answer in results.items():
        print(f"\n\033[1mQ: {question}\033[0m")
        print(f"A: {answer}")


def pretty_print_tokens(token_results: list[tuple[int, str, str]], max_response_len: int = 80):
    """
    Pretty print token-by-token results from probe_tokens().

    Args:
        token_results: List of tuples from probe_tokens()
        max_response_len: Maximum length of response to display

    Example:
        results = probe_tokens(messages, "What does this mean?")
        pretty_print_tokens(results)

        # Output:
        # ================================================================================
        # TOKEN-BY-TOKEN ORACLE RESPONSES
        # ================================================================================
        #
        # [  0] <|im_start|>         → This is a special token marking the start
        # [  1] user                 → Indicates a user message
        # [  2] Hello                → A greeting word
        # ...
    """
    print("\n" + "=" * 100)
    print("TOKEN-BY-TOKEN ORACLE RESPONSES")
    print("=" * 100 + "\n")

    for idx, token_str, response in token_results:
        token_display = token_str.replace("\n", "\\n").replace("\r", "\\r")[:20]
        response_display = response[:max_response_len] + ("..." if len(response) > max_response_len else "")
        print(f"\033[94m[{idx:3d}] {token_display:<22}\033[0m \033[92m→\033[0m {response_display}")


# Print info message when imported
print("Activation Oracles library loaded!")
print("\nQuick start:")
print("  1. model, tokenizer, device = load_oracle_model(base_model, oracle_adapter)")
print("  2. Use chat(), probe(), chat_and_probe(), etc.")
print("\nAvailable functions:")
print("  • load_oracle_model()         - One-stop model loading")
print("  • chat()                      - Generate from base model")
print("  • chat_ao()                   - Chat with oracle directly")
print("  • probe()                     - Probe segment activations")
print("  • probe_before_generation()   - Probe at generation token")
print("  • probe_control_tokens()      - Probe only at turn boundaries (recommended)")
print("  • find_control_tokens()       - Find control token positions")
print("  • chat_and_probe()            - Generate and probe together")
print("  • probe_tokens()              - Token-by-token probing")
print("  • show_tokens()               - Display token positions")
print("  • pretty_print_probe()        - Format segment results")
print("  • pretty_print_tokens()       - Format token results")
