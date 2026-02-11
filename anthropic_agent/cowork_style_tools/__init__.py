"""Cowork-style tools for agentic AI workflows.

This package provides tools that follow the Cowork tool specification.
It includes:
- Cowork-style filesystem tools (read/write/edit/glob/grep) as @tool functions
- System-level tools like `BashTool` (ConfigurableToolBase-based)
"""

from __future__ import annotations

from typing import Callable

from .bash_tool import BashTool
from .edit import create_edit_tool
from .glob_tool import create_glob_tool
from .grep_tool import create_grep_tool
from .read import create_read_tool
from .write import create_write_tool


def create_cowork_tools() -> list[Callable]:
    """Create the cowork-style filesystem tools (read/write/edit/glob/grep).

    Returns a list of @tool-decorated functions ready for registration
    with AnthropicAgent.
    """

    return [
        create_read_tool(),
        create_write_tool(),
        create_edit_tool(),
        create_glob_tool(),
        create_grep_tool(),
        BashTool.get_tool()
    ]


__all__ = [
    "BashTool",
    "create_cowork_tools",
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_glob_tool",
    "create_grep_tool",
]
