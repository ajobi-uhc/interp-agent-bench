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
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Your prompt here"}
    ]
)

print(response.content[0].text)
```

## Available Models

### Claude 4 Series (Latest - 2025)
- `claude-opus-4-1-20250805` - Most powerful, deep reasoning (Opus 4.1)
- `claude-sonnet-4-5-20250929` - Balanced, advanced reasoning (Sonnet 4.5)
- `claude-haiku-4-5-20251001` - Fast and efficient, near-frontier performance (Haiku 4.5)
- `claude-sonnet-4-20250514`


**Note**: Model identifiers follow the pattern `claude-{version}-{model}-{date}`. Check [Anthropic's documentation](https://docs.anthropic.com/en/docs/about-claude/models) for the latest available models and their exact identifiers.

## Tips
- The API key is already loaded as `ANTHROPIC_API_KEY`
- Start with simple prompts to verify API access
- Document your experiments as you go
