"""Helper utilities for the agent."""


def format_result(data: dict) -> str:
    """Format a result dict as markdown."""
    lines = ["## Result", ""]
    for key, value in data.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def summarize_list(items: list, max_items: int = 5) -> str:
    """Summarize a list, showing first N items."""
    if len(items) <= max_items:
        return str(items)
    return f"{items[:max_items]}... ({len(items)} total)"
