"""Simple interactive agent - ESC to interrupt (background monitoring)."""

import asyncio
import sys
from datetime import datetime
from typing import Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
)

from .logging import run_agent_with_logging


class EscapeMonitor:
    """Monitor for ESC key in background."""
    
    def __init__(self):
        self.pressed = False
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._fd = None
        self._old_settings = None
    
    async def start(self):
        """Start monitoring."""
        import tty
        import termios
        
        self.pressed = False
        self._running = True
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        
        self._task = asyncio.create_task(self._monitor())
    
    async def stop(self):
        """Stop monitoring and restore terminal."""
        import termios
        import select
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        # Restore terminal
        if self._old_settings and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        
        # Drain any buffered input
        try:
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
        except Exception:
            pass
    
    async def _monitor(self):
        """Background loop checking for ESC."""
        import select
        
        loop = asyncio.get_event_loop()
        
        while self._running:
            try:
                # Use run_in_executor to avoid blocking the event loop
                ready = await loop.run_in_executor(
                    None, 
                    lambda: select.select([sys.stdin], [], [], 0.1)[0]
                )
                
                if ready and self._running:
                    char = sys.stdin.read(1)
                    if char == '\x1b':  # ESC
                        self.pressed = True
                        print("\n\n⚠️  ESC pressed - interrupting...", flush=True)
                        return
            except Exception:
                pass
            
            await asyncio.sleep(0.01)  # Small yield to event loop


async def run_agent_interactive(
    prompt: str,
    mcp_config: dict,
    user_message: str = "",
    provider: str = "claude",
    model: Optional[str] = None,
    kwargs: Optional[dict] = None,
) -> None:
    """Interactive session - ESC to interrupt."""
    if provider != "claude":
        raise ValueError(f"Interactive mode only supports 'claude' provider, got: {provider}")

    if model is None:
        model = "claude-sonnet-4-5-20250929"

    options = ClaudeAgentOptions(
        system_prompt=prompt,
        model=model,
        mcp_servers=mcp_config,
        permission_mode="bypassPermissions",
        include_partial_messages=True,
        **(kwargs or {})
    )

    print("\n💬 Press ESC to interrupt • Type 'exit' to quit\n", flush=True)

    client = ClaudeSDKClient(options=options)
    await client.connect()

    next_prompt = user_message
    monitor = EscapeMonitor()
    first_query = True

    try:
        while True:
            if not next_prompt:
                try:
                    next_prompt = input("\n→ You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Exiting...")
                    break

                if next_prompt.lower() in ['exit', 'quit']:
                    break
                if not next_prompt:
                    continue

            await client.query(next_prompt)
            next_prompt = None

            # Start ESC monitoring
            await monitor.start()

            try:
                # Create agent stream generator
                async def agent_stream():
                    # Yield init message on first query
                    if first_query and hasattr(client, 'tools') and client.tools:
                        yield {
                            "type": "init",
                            "provider": "claude",
                            "model": model,
                            "tools": [{"name": getattr(tool, 'name', str(tool))} for tool in client.tools],
                        }

                    async for message in client.receive_response():
                        # Check if ESC was pressed
                        if monitor.pressed:
                            await client.interrupt()
                            monitor.pressed = False
                            continue

                        yield message

                        if isinstance(message, ResultMessage):
                            break

                # Wrap with logging adapter
                async for message in run_agent_with_logging(agent_stream()):
                    pass  # Logging adapter handles output

                first_query = False

            finally:
                await monitor.stop()

    finally:
        await client.disconnect()