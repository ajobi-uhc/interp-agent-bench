"""Base provider class for AI CLI integrations."""

from abc import ABC, abstractmethod


class AICLIProvider(ABC):
    """Abstract base class for AI CLI providers."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name (e.g., 'claude', 'gemini')."""
        pass

    @abstractmethod
    def get_provider_display_name(self) -> str:
        """Return the provider display name (e.g., 'Claude Code CLI', 'Gemini CLI')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this AI CLI is available on the system."""
        pass
