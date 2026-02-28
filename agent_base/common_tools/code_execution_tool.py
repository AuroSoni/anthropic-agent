"""CodeExecutionTool — stub, not yet migrated.

Requires python_executors module which is not yet in agent_base.
"""
from __future__ import annotations

from typing import Callable

from ..tools.base import ConfigurableToolBase


class CodeExecutionTool(ConfigurableToolBase):
    """Stub — requires python_executors module (not yet in agent_base)."""

    DOCSTRING_TEMPLATE = """Execute Python code.

Args:
    code: Python code to execute.
"""

    def get_tool(self) -> Callable:
        raise NotImplementedError(
            "CodeExecutionTool requires python_executors (not yet migrated to agent_base)"
        )
