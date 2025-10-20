# API Access: Anthropic Claude

You have access to Claude models via the Anthropic API. The API key is pre-loaded in your environment.

## Setup

```python
import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
```

## Basic Usage

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Your prompt here"}
    ]
)

print(response.content[0].text)
```

## Available Models
- `claude-3-5-sonnet-20241022` - Most capable, balanced
- `claude-3-5-haiku-20241022` - Fast and efficient
- `claude-3-opus-20240229` - Most powerful (legacy)

## Tips
- The API key is already loaded as `ANTHROPIC_API_KEY`
- Start with simple prompts to verify API access
- Document your experiments as you go
