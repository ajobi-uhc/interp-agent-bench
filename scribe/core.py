"""Core utilities for scribe."""


class HiddenPromptTokenizer:
    """Wraps a tokenizer to inject hidden system prompts into the target model.

    This is used for "model organism" experiments where the agent must discover
    hidden behavioral constraints through interpretability techniques.

    The hidden prompt is injected at the chat template level, so it affects
    instruction-following behavior but doesn't interfere with raw tokenization.
    """

    def __init__(self, tokenizer, hidden_prompt: str):
        """
        Args:
            tokenizer: The underlying tokenizer to wrap
            hidden_prompt: System prompt to inject (hidden from agent)
        """
        self._tokenizer = tokenizer
        self._hidden_prompt = hidden_prompt

    def apply_chat_template(self, messages, **kwargs):
        """Apply chat template with hidden system prompt injected.

        The hidden prompt is prepended as a system message before the agent's messages.
        This creates the behavioral constraint the agent must discover.
        """
        # Inject hidden system prompt at the beginning
        modified_messages = [
            {"role": "system", "content": self._hidden_prompt},
            *messages
        ]
        return self._tokenizer.apply_chat_template(modified_messages, **kwargs)

    def __call__(self, *args, **kwargs):
        """Direct tokenization passes through without modification.

        This allows the agent to do raw tokenization analysis without interference.
        The hidden prompt only affects chat-based generation.
        """
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped tokenizer."""
        return getattr(self._tokenizer, name)

    def __repr__(self):
        return f"HiddenPromptTokenizer(tokenizer={self._tokenizer}, hidden_prompt='[REDACTED]')"
