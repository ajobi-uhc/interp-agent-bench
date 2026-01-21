"""OpenRouter client initialization for Qwen3 8B model access.

Use this like so:
from openrouter_client import client
client.chat.completions.create( ... )
"""

import os
from openai import OpenAI

# Initialize and return OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)