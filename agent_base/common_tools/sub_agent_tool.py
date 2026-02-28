"""SubAgentTool — stub, not yet migrated.

Requires Agent system which is not yet in agent_base.
"""
from __future__ import annotations

from typing import Callable

from ..tools.base import ConfigurableToolBase


class SubAgentTool(ConfigurableToolBase):
    """Stub — requires Agent system (not yet in agent_base)."""

    DOCSTRING_TEMPLATE = """Spawn a sub-agent.

Args:
    agent_name: Agent to spawn.
    task: Task description.
"""

    def get_tool(self) -> Callable:
        raise NotImplementedError(
            "SubAgentTool requires Agent system (not yet migrated to agent_base)"
        )
