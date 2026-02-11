"""Cowork-style filesystem tools for AnthropicAgent.

These tools follow the Claude Cowork design: absolute paths, no sandboxing,
no extension filtering, and simple interfaces (string replacement for edits,
full-file writes, etc.).

Usage:
    from anthropic_agent.cowork_style_tools import create_cowork_tools
    agent = AnthropicAgent(tools=create_cowork_tools())
"""
from typing import Callable

from .read import create_read_tool
from .write import create_write_tool
from .edit import create_edit_tool
from .glob_tool import create_glob_tool
from .grep_tool import create_grep_tool


def create_cowork_tools() -> list[Callable]:
    """Create all five cowork-style filesystem tools.

    Returns a list of @tool-decorated functions ready for registration
    with AnthropicAgent.

    Returns:
        List of tool functions: [read_file, write_file, edit_file, glob_search, grep_search]
    """
    return [
        create_read_tool(),
        create_write_tool(),
        create_edit_tool(),
        create_glob_tool(),
        create_grep_tool(),
    ]


__all__ = [
    "create_cowork_tools",
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_glob_tool",
    "create_grep_tool",
]
