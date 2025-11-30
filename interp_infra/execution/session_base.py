"""Base session functionality for all execution modes."""

from dataclasses import dataclass, field
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment.scoped_sandbox import Proxy
    from ..extension import Extension


@dataclass
class SessionBase:
    """
    Base class for all session types.

    Manages configuration accumulation for agent integration:
    - MCP endpoints (from Proxies)
    - System prompts (from Extension docs)
    """
    _mcp_endpoints: list[dict] = field(default_factory=list, init=False, repr=False)
    _prompts: list[str] = field(default_factory=list, init=False, repr=False)

    def add(self, item: Union["Proxy", "Extension"]):
        """
        Add extension or proxy to the session.

        - Extension: Code executed in session, docs added to prompt
        - Proxy: Added as MCP tool (from ScopedSandbox)

        Args:
            item: Extension (local code) or Proxy (from ScopedSandbox)
        """
        from ..environment.scoped_sandbox import Proxy
        from ..extension import Extension

        if isinstance(item, Proxy):
            self._mcp_endpoints.append(item.as_mcp_config())
        elif isinstance(item, Extension):
            self._execute_extension(item)
            if item.docs:
                self._prompts.append(item.docs)
        else:
            raise TypeError(f"Expected Proxy or Extension, got {type(item)}")

    def _execute_extension(self, extension: "Extension"):
        """Execute extension code. Subclasses implement this."""
        raise NotImplementedError("Subclasses must implement _execute_extension")

    @property
    def system_prompt(self) -> str:
        """Accumulated system prompt from extensions."""
        return "\n\n".join(self._prompts) if self._prompts else ""
