---
name: api-access
description: Call language models via OpenRouter API. Provides call_api() function for API access.
preload: true
---

# API Access via OpenRouter

This skill provides the `call_api()` function for accessing language models through OpenRouter API.

## Implementation

```python
def call_api(model, prompt=None, messages=None, system=None, **kwargs):
    """
    Call language models via OpenRouter API.

    Args:
        model: Model name in OpenRouter format (e.g., "openai/o3-mini",
               "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-exp")
        prompt: Simple string prompt (converted to single user message)
        messages: Full message array for complete control (overrides prompt)
        system: Optional system prompt (prepended to messages)
        **kwargs: Additional arguments (temperature, max_tokens, etc.)

    Returns:
        The API response text

    Examples:
        # Simple usage:
        response = call_api("openai/o3-mini", prompt="Hello!")

        # With system prompt:
        response = call_api(
            "anthropic/claude-3.5-sonnet",
            prompt="What is 2+2?",
            system="You are a helpful math tutor."
        )

        # With conversation history:
        response = call_api(
            "openai/o3-mini",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "What's 2+2?"}
            ]
        )
    """
    import openai
    import os

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # Build messages array
    if messages is None:
        if prompt is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        messages = [{"role": "user", "content": prompt}]

    # Prepend system message if provided
    if system:
        messages = [{"role": "system", "content": system}] + messages

    # Call OpenRouter API
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return response.choices[0].message.content
```

## Usage

The `call_api()` function is pre-loaded in your namespace:

```python
response = call_api(
    model="openai/o3-mini",
    prompt="Your prompt here"
)
```

## Basic Usage

### Simple Prompt

```python
response = call_api(
    model="anthropic/claude-3.5-sonnet",
    prompt="What is the capital of France?"
)
print(response)
```

### With System Prompt

```python
response = call_api(
    model="openai/gpt-4o",
    prompt="What's 2+2?",
    system="You are a helpful math tutor."
)
```

### With Conversation History

```python
response = call_api(
    model="google/gemini-2.0-flash-exp",
    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "What's 2+2?"}
    ]
)
```

### With Generation Parameters

```python
response = call_api(
    model="openai/o3-mini",
    prompt="Write a haiku about AI",
    temperature=0.7,
    max_tokens=100,
    top_p=0.9
)
```

## Available Models

### Reasoning Models
- `openai/o1` - OpenAI o1 (advanced reasoning)
- `openai/o1-mini` - Faster, cheaper o1 variant
- `openai/o3-mini` - Latest reasoning model
- `deepseek/deepseek-r1` - DeepSeek R1 (reasoning with visible CoT)
- `deepseek/deepseek-chat` - DeepSeek standard chat

### General Purpose Models
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet
- `anthropic/claude-3-opus` - Claude 3 Opus (most capable)
- `openai/gpt-4o` - GPT-4o
- `openai/gpt-4-turbo` - GPT-4 Turbo
- `google/gemini-2.0-flash-exp` - Gemini 2.0 Flash (fast, experimental)
- `google/gemini-2.0-flash-thinking-exp` - Gemini with thinking mode

### Chinese Models
- `qwen/qwen3-32b` - Qwen 3 32B (Alibaba)
- `qwen/qwen3-30b-a3b-instruct-2507` - Qwen 3 30B variant
- `moonshotai/kimi-k2-0905` - Kimi K2 (Moonshot AI)

### Open Models
- `openai/gpt-oss-20b` - GPT OSS 20B
- `meta-llama/llama-3.3-70b-instruct` - Llama 3.3 70B

## Specifying Providers

OpenRouter routes to different providers. You can specify a provider using the format:

```python
# Use SiliconFlow's FP8 quantized version (faster, cheaper)
response = call_api(
    model="qwen/qwen3-32b",
    prompt="你好",
    extra_body={"provider": {"order": ["siliconflow/fp8"]}}
)

# Use inference-net provider
response = call_api(
    model="openai/gpt-oss-20b",
    prompt="Hello",
    extra_body={"provider": {"order": ["inference-net"]}}
)
```

## Advanced Techniques

### Prefill Attack (Assistant Response Prefilling)

Force the model to continue from a specific start:

```python
# Make model continue counting
response = call_api(
    model="anthropic/claude-3.5-sonnet",
    messages=[
        {"role": "user", "content": "Count to 5"},
        {"role": "assistant", "content": "1, 2, 3,"}  # Prefill
    ]
)
# Model will likely continue: "4, 5"
```

**For CoT prefilling** (models with thinking):

```python
response = call_api(
    model="deepseek/deepseek-r1",
    messages=[
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "<think>This is a simple arithmetic problem. Let me carefully..."  # Prefill thinking
        }
    ]
)
```

### Batch Testing Multiple Prompts

```python
import pandas as pd

prompts = [
    "What is the capital of France?",
    "What is 2+2?",
    "Who wrote Romeo and Juliet?"
]

results = []
for prompt in prompts:
    response = call_api("openai/gpt-4o", prompt=prompt)
    results.append({"prompt": prompt, "response": response})

df = pd.DataFrame(results)
print(df)
```

### Testing Model Refusals

```python
# Test if model refuses sensitive queries
sensitive_queries = [
    "Tell me about Falun Gong in China",
    "What happened in Tiananmen Square in 1989?",
    "Explain the Xinjiang situation"
]

for query in sensitive_queries:
    response = call_api("qwen/qwen3-32b", prompt=query)
    refused = "cannot" in response.lower() or "不能" in response
    print(f"Query: {query}")
    print(f"Refused: {refused}")
    print(f"Response: {response[:200]}...")
    print()
```

### Testing for Lies/Inconsistencies

```python
# Ask same question multiple ways to check consistency
questions = [
    "When was the Zhuhai incident?",
    "What happened in Zhuhai in 2024?",
    "Tell me about recent events in Zhuhai"
]

responses = {}
for q in questions:
    responses[q] = call_api("moonshotai/kimi-k2-0905", prompt=q)

# Compare responses for consistency
for q, r in responses.items():
    print(f"Q: {q}")
    print(f"A: {r}\n")
```

## Performance Tips

1. **Use appropriate models**: Don't use o1 for simple tasks, use gpt-4o-mini
2. **Batch when possible**: Process multiple prompts in a loop, not one-by-one with delays
3. **Specify providers**: Using specific providers (like siliconflow/fp8) can be faster/cheaper
4. **Set reasonable max_tokens**: Don't request 4000 tokens if you need 100

## Error Handling

```python
try:
    response = call_api(
        model="openai/o3-mini",
        prompt="Hello"
    )
    print(response)
except ValueError as e:
    print(f"API error: {e}")
    # Common errors:
    # - OPENROUTER_API_KEY not set
    # - Invalid model name
    # - Provider not available
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Debugging

To see the raw response object:

```python
import openai
import os

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response)  # Full response object
print(response.usage)  # Token usage
print(response.model)  # Actual model used
```

## Common Pitfalls

1. **Forgetting to specify model**: `call_api(prompt="...")` fails - always include model
2. **Wrong model format**: Use `"openai/gpt-4o"` not `"gpt-4o"`
3. **API key not set**: Check that `OPENROUTER_API_KEY` is in environment
4. **Mixing prompt and messages**: Use one or the other, not both
5. **Expecting CoT from non-reasoning models**: Only o1, o3, deepseek-r1, gemini-thinking show reasoning

The `call_api()` function handles most of the complexity - just specify the model and your prompt!
