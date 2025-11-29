"""Simple agent runner for notebook sessions."""

from typing import Optional, Literal, AsyncIterator

from ..execution.notebook_session import NotebookSession
from .skill import Skill
from .providers import run_claude, run_openai, run_gemini


Provider = Literal["claude", "gemini", "openai"]


async def run_agent(
    session: NotebookSession,
    task: str,
    provider: Provider = "claude",
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    skills: Optional[list[Skill]] = None,
    verbose: bool = False,
) -> AsyncIterator[dict]:
    """
    Run an agent against a notebook session.

    For more control, use session.mcp_config directly with your own agent setup.

    Args:
        session: NotebookSession to connect to
        task: What to do
        provider: "claude", "gemini", or "openai"
        model: Override default model for provider
        system_prompt: Base system prompt (optional)
        skills: Skills to load (code exec'd, prompts added to system)
        verbose: Print debug info

    Yields:
        Agent messages (dict format)
    """
    # Load skill code into kernel
    for skill in (skills or []):
        if verbose:
            print(f"Loading skill: {skill.name}")
        session.exec(skill.code, hidden=True)

    # Build system prompt
    prompt_parts = []

    if system_prompt:
        prompt_parts.append(system_prompt)
    else:
        prompt_parts.append(
            "You are an AI research agent with access to a Jupyter notebook."
        )

    if skills:
        prompt_parts.append("\n## Available Tools\n")
        for skill in skills:
            prompt_parts.append(f"### {skill.name}\n{skill.prompt}\n")

    full_system_prompt = "\n".join(prompt_parts)

    # Get MCP config
    mcp_config = session.mcp_config

    # Run with appropriate provider
    if provider == "claude":
        default_model = "claude-sonnet-4-5-20250929"
        async for message in run_claude(
            mcp_config=mcp_config,
            system_prompt=full_system_prompt,
            task=task,
            model=model or default_model,
            verbose=verbose,
        ):
            yield message

    elif provider == "gemini":
        default_model = "gemini-2.0-flash"
        async for message in run_gemini(
            mcp_config=mcp_config,
            system_prompt=full_system_prompt,
            task=task,
            model=model or default_model,
            verbose=verbose,
        ):
            yield message

    elif provider == "openai":
        default_model = "gpt-5-high"
        async for message in run_openai(
            mcp_config=mcp_config,
            system_prompt=full_system_prompt,
            task=task,
            model=model or default_model,
            verbose=verbose,
        ):
            yield message

    else:
        raise ValueError(f"Unknown provider: {provider}")
