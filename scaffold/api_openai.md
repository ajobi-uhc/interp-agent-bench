# API Access: OpenAI GPT

You have access to GPT models via the OpenAI API. The API key is pre-loaded in your environment.

## Setup

```python
import openai
import os

client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
```

## Basic Usage

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Your prompt here"}
    ]
)

print(response.choices[0].message.content)
```

## Available Models
- `gpt-4` - Most capable
- `gpt-4-turbo` - Fast and capable
- `gpt-3.5-turbo` - Fast and efficient

## Tips
- The API key is already loaded as `OPENAI_API_KEY`
- Start with simple prompts to verify API access
- Document your experiments as you go
