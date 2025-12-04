"""OpenRouter client initialization for Kimi model access."""

import os
from openai import OpenAI

# Initialize and return OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)
