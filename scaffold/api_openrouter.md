# API Access: OpenRouter

You have access to multiple LLM models via the OpenRouter API. OpenRouter provides unified access to various model providers including Anthropic, OpenAI, Google, Meta, and others.

## Setup

```python
import os
from openai import OpenAI

# OpenRouter uses OpenAI-compatible API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ['OPENROUTER_API_KEY']
)
```

## Basic Usage

```python
response = client.chat.completions.create(
    model="<task specific model id eg. qwen/qwen3-32b>",
    messages=[
        {"role": "user", "content": "Your prompt here"}
    ],
    extra_body={
        "provider": {
            "order": ["<one task specific provider eg. novita/fp8>"],
            "allow_fallbacks": False
        }
    }
)

print(response.choices[0].message.content)
```

Use the model specified in the task description as well as the provider specified in the task description
Do not include more than one provider