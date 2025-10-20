"""API client wrappers that inject hidden system prompts transparently.

Used for "model organism" experiments where the agent makes API calls but we
control system prompt injection on our side.
"""

from typing import Any, Dict, List, Optional


class HiddenPromptAnthropicClient:
    """Wraps Anthropic API client to inject hidden system prompts.

    The agent thinks they're calling the Anthropic API directly, but we
    secretly prepend a hidden system prompt to all requests.
    """

    def __init__(self, client, hidden_system_prompt: str):
        """
        Args:
            client: The underlying anthropic.Anthropic() client
            hidden_system_prompt: System prompt to inject (hidden from agent)
        """
        self._client = client
        self._hidden_prompt = hidden_system_prompt

    @property
    def messages(self):
        """Return wrapped messages interface."""
        return _MessagesWrapper(self._client.messages, self._hidden_prompt)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped client."""
        return getattr(self._client, name)

    def __repr__(self):
        return f"HiddenPromptAnthropicClient(client={self._client}, hidden_prompt='[REDACTED]')"


class _MessagesWrapper:
    """Wraps client.messages to intercept create() calls."""

    def __init__(self, messages, hidden_prompt: str):
        self._messages = messages
        self._hidden_prompt = hidden_prompt

    def create(self, **kwargs):
        """Intercept messages.create() and inject hidden system prompt.

        The hidden prompt is prepended to the system parameter or added if missing.
        """
        # Get existing system prompt if any
        existing_system = kwargs.get('system', '')

        # Inject hidden prompt at the beginning
        if existing_system:
            # Prepend hidden prompt to existing system prompt
            kwargs['system'] = f"{self._hidden_prompt}\n\n{existing_system}"
        else:
            # Add hidden prompt as system prompt
            kwargs['system'] = self._hidden_prompt

        # Call the underlying API with modified parameters
        return self._messages.create(**kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped messages."""
        return getattr(self._messages, name)


class HiddenPromptOpenAIClient:
    """Wraps OpenAI API client to inject hidden system prompts."""

    def __init__(self, client, hidden_system_prompt: str):
        """
        Args:
            client: The underlying openai.OpenAI() client
            hidden_system_prompt: System prompt to inject (hidden from agent)
        """
        self._client = client
        self._hidden_prompt = hidden_system_prompt

    @property
    def chat(self):
        """Return wrapped chat interface."""
        return _OpenAIChatWrapper(self._client.chat, self._hidden_prompt)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped client."""
        return getattr(self._client, name)

    def __repr__(self):
        return f"HiddenPromptOpenAIClient(client={self._client}, hidden_prompt='[REDACTED]')"


class _OpenAIChatWrapper:
    """Wraps client.chat to intercept completions.create() calls."""

    def __init__(self, chat, hidden_prompt: str):
        self._chat = chat
        self._hidden_prompt = hidden_prompt

    @property
    def completions(self):
        """Return wrapped completions interface."""
        return _OpenAICompletionsWrapper(self._chat.completions, self._hidden_prompt)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped chat."""
        return getattr(self._chat, name)


class _OpenAICompletionsWrapper:
    """Wraps client.chat.completions to intercept create() calls."""

    def __init__(self, completions, hidden_prompt: str):
        self._completions = completions
        self._hidden_prompt = hidden_prompt

    def create(self, **kwargs):
        """Intercept completions.create() and inject hidden system prompt.

        The hidden prompt is prepended as a system message.
        """
        # Get existing messages
        messages = kwargs.get('messages', [])

        # Inject hidden system prompt at the beginning
        modified_messages = [
            {"role": "system", "content": self._hidden_prompt},
            *messages
        ]

        kwargs['messages'] = modified_messages

        # Call the underlying API with modified parameters
        return self._completions.create(**kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped completions."""
        return getattr(self._completions, name)


class HiddenPromptGoogleClient:
    """Wraps Google Generative AI client to inject hidden system prompts."""

    def __init__(self, client, hidden_system_prompt: str):
        """
        Args:
            client: The underlying google.generativeai client module
            hidden_system_prompt: System prompt to inject (hidden from agent)
        """
        self._client = client
        self._hidden_prompt = hidden_system_prompt
        self._original_generate_content = None

    def GenerativeModel(self, *args, **kwargs):
        """Intercept GenerativeModel creation to wrap it."""
        model = self._client.GenerativeModel(*args, **kwargs)
        return _GoogleModelWrapper(model, self._hidden_prompt)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped client."""
        return getattr(self._client, name)

    def __repr__(self):
        return f"HiddenPromptGoogleClient(client={self._client}, hidden_prompt='[REDACTED]')"


class _GoogleModelWrapper:
    """Wraps Google GenerativeModel to intercept generate_content() calls."""

    def __init__(self, model, hidden_prompt: str):
        self._model = model
        self._hidden_prompt = hidden_prompt

    def generate_content(self, contents, **kwargs):
        """Intercept generate_content() and inject hidden system prompt."""
        # Prepend hidden prompt to the content
        if isinstance(contents, str):
            modified_contents = f"{self._hidden_prompt}\n\n{contents}"
        elif isinstance(contents, list):
            # Add hidden prompt as first message
            modified_contents = [
                {"role": "user", "parts": [self._hidden_prompt]},
                *contents
            ]
        else:
            modified_contents = contents

        return self._model.generate_content(modified_contents, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped model."""
        return getattr(self._model, name)
