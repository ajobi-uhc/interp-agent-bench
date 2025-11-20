"""Environment for API-based model access without GPU loading."""

from typing import Dict, Any, List
from .base import EnvironmentConfig, PrewarmTask, register_environment


@register_environment("api_access")
class APIAccessEnvironment:
    """Environment that provides API access without loading models."""

    def get_prewarm_plan(self, cfg: EnvironmentConfig) -> List[PrewarmTask]:
        """No models to download."""
        return []

    def get_default_skills(self) -> List[str]:
        """API access typically doesn't need interpretability skills."""
        return []

    def warm_init(self, cfg: EnvironmentConfig) -> Dict[str, Any]:
        """Initialize OpenRouter API client."""
        import os

        namespace = {}

        # Import OpenAI library (used for OpenRouter)
        try:
            import openai
            namespace['openai'] = openai
            print("  Loaded: openai (for OpenRouter)")
        except ImportError:
            raise ImportError("openai package is required for API access")

        # Add a helper function for OpenRouter API calls
        def call_api(
            model: str,
            prompt: str = None,
            messages: list = None,
            system: str = None,
            **kwargs
        ):
            """
            Thin wrapper over OpenRouter API with convenience options.

            Args:
                model: Model name in OpenRouter format (e.g., "openai/o3-mini",
                       "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-exp")
                prompt: Simple string prompt (convenience - converted to single user message)
                messages: Full message array for complete control (overrides prompt)
                system: Optional system prompt (prepended to messages)
                **kwargs: Additional arguments passed to the API (temperature, max_tokens, etc.)

            Returns:
                The API response text

            Examples:
                # Simple usage:
                response = call_api("openai/o3-mini", prompt="Hello, world!")

                # With system prompt:
                response = call_api(
                    "anthropic/claude-3.5-sonnet",
                    prompt="What is 2+2?",
                    system="You are a helpful math tutor."
                )

                # Full control with message history:
                response = call_api(
                    "openai/o3-mini",
                    messages=[
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi! How can I help?"},
                        {"role": "user", "content": "What's 2+2?"}
                    ]
                )

                # Prefilling assistant response:
                response = call_api(
                    "anthropic/claude-3.5-sonnet",
                    messages=[
                        {"role": "user", "content": "Count to 3"},
                        {"role": "assistant", "content": "1, 2,"}  # Prefill
                    ]
                )
            """
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

        namespace['call_api'] = call_api

        print("OpenRouter API access initialized")
        print(f"  Available in namespace: {list(namespace.keys())}")
        print("  Use call_api(model, prompt) with OpenRouter model names")

        # Load skills
        from interp_infra.skills.loader import SkillLoader

        skill_names = cfg.extra.get("skills", self.get_default_skills())
        if skill_names:
            print(f"\nLoading skills: {', '.join(skill_names)}")
            loader = SkillLoader()
            loader.load_skills(skill_names, namespace)

        return namespace
