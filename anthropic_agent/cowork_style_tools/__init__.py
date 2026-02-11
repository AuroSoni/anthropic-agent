"""Cowork-style tools for agentic AI workflows.

This module provides tools that follow the Cowork tool specification,
implementing capabilities like shell execution, browser automation,
and other system-level operations.

All tool classes inherit from ConfigurableToolBase, which provides:
- Templated docstrings with {placeholder} syntax for dynamic values
- Optional custom docstring templates via docstring_template parameter
- Optional complete schema override via schema_override parameter
"""
from __future__ import annotations

from .bash_tool import BashTool

__all__ = [
    "BashTool",
]
