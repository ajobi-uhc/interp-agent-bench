# API Access: Google Gemini

You have access to Gemini models via the Google AI API. The API key is pre-loaded in your environment.

## Setup

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
```

## Basic Usage

```python
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Your prompt here")

print(response.text)
```

## Available Models
- `gemini-pro` - Most capable
- `gemini-pro-vision` - Multimodal (text + images)

## Tips
- The API key is already loaded as `GOOGLE_API_KEY`
- Start with simple prompts to verify API access
- Document your experiments as you go
