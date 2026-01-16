"""
### Spec for `create_plan` in `common_tools/create_plan.py`

- **Signature**
  - `def create_plan(plan: str, name: Optional[str] = None, todos: Optional[List[TodoItem]] = None) -> str`

- **Purpose**
  - Create a markdown plan file and optionally append todos to the agent's todo list.

- **Directory structure**
  - Plans are stored at `{base_path}/{agent_uuid}/{name}_{timestamp}.md` (or `plan_{timestamp}.md` if no name)
  - Todos are appended to the shared `{base_path}/{agent_uuid}/todos.yaml`

- **Behavior**
  - Write plan content directly to file (LLM provides formatted markdown).
  - If todos provided, validate and append to the agent's todos.yaml.
  - Plan file contains only the markdown - no embedded todos section.

- **Returns**
  - YAML-like structured response with status, plan_path, todos_updated flag, and todos_summary if applicable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from ..tools.decorators import tool
from .todo_write import (
    TodoItem,
    _validate_todos,
    _load_todos,
    _save_todos,
    _merge_todos,
    _format_status_summary,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------
def _generate_timestamp() -> str:
    """Generate a timestamp string for file naming."""
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def _sanitize_name(name: str) -> str:
    """Sanitize name for use in filename.
    
    Args:
        name: Raw name string to sanitize.
    
    Returns:
        Sanitized string safe for use in filenames.
    """
    # Replace spaces with underscores, keep alphanumeric and underscores
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name.replace(" ", "_"))
    return sanitized.strip("_") or "plan"


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class CreatePlanTool:
    """Configurable create_plan tool with a sandboxed base path and agent association.
    
    This class encapsulates the create_plan functionality, allowing configuration
    of the base path at instantiation time. The agent UUID can be provided at init
    or injected later via set_agent_uuid() (called automatically by AnthropicAgent).
    
    The tool returned by get_tool() can be registered with an AnthropicAgent.
    
    Example:
        >>> plan_tool = CreatePlanTool(base_path="/path/to/workspace")
        >>> agent = AnthropicAgent(tools=[plan_tool.get_tool()])
        >>> # agent_uuid is automatically injected by AnthropicAgent
    """
    
    def __init__(
        self,
        base_path: str | Path,
        agent_uuid: Optional[str] = None,
    ):
        """Initialize the CreatePlanTool with a base path and optional agent UUID.
        
        Args:
            base_path: The root directory for storing plans and todos.
                       Plans will be stored at {base_path}/{agent_uuid}/
                       Todos will be appended to {base_path}/{agent_uuid}/todos.yaml
            agent_uuid: Optional unique identifier for the agent session.
                       If not provided, must be set via set_agent_uuid() before use.
        """
        self.base_path: Path = Path(base_path).resolve()
        self.agent_uuid: Optional[str] = agent_uuid
    
    def set_agent_uuid(self, uuid: str) -> None:
        """Set the agent UUID for this tool instance.
        
        Called automatically by AnthropicAgent after initialization.
        
        Args:
            uuid: The agent's unique identifier.
        """
        self.agent_uuid = uuid
    
    def _get_agent_dir(self) -> Path:
        """Get the directory for this agent's plans and todos."""
        if not self.agent_uuid:
            raise RuntimeError("agent_uuid not set. Tool must be registered with an AnthropicAgent.")
        return self.base_path / self.agent_uuid
    
    def _ensure_agent_dir(self) -> Path:
        """Ensure the agent directory exists and return it."""
        agent_dir = self._get_agent_dir()
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir
    
    def _get_todos_path(self) -> Path:
        """Get the fixed path for the agent's todo file."""
        return self._get_agent_dir() / "todos.yaml"
    
    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.
        
        Returns:
            A decorated create_plan function that operates within the configured base_path.
            The function has __tool_instance__ attribute for agent UUID injection.
        """
        instance = self
        
        @tool
        def create_plan(
            plan: str,
            name: Optional[str] = None,
            todos: Optional[List[TodoItem]] = None,
        ) -> str:
            """Create a markdown plan file and optionally append todos to the agent's todo list.
            
            Creates a new plan file under {base_path}/{agent_uuid}/ with a timestamped
            filename. If todos are provided, they are appended to the agent's shared
            todos.yaml file (the same file managed by todo_write).
            
            Args:
                plan: Full markdown content for the plan. The LLM provides the
                      complete formatted text including title, overview, steps, etc.
                name: Optional name prefix for the plan file. If provided, filename will be
                      {name}_{timestamp}.md. If not provided, defaults to plan_{timestamp}.md.
                todos: Optional list of todos to append to the agent's todo file.
                       Each todo must have 'id' and 'content'. Status defaults to 'pending'.
            
            Returns:
                Confirmation message with path to created plan file.
            """
            if not plan or not plan.strip():
                return "Error: Plan content cannot be empty."
            
            # Validate todos if provided
            if todos:
                validation_error = _validate_todos(todos, require_content=True)
                if validation_error:
                    return validation_error
            
            # Ensure directory exists
            agent_dir = instance._ensure_agent_dir()
            
            # Generate plan filename and write
            timestamp = _generate_timestamp()
            name_prefix = _sanitize_name(name) if name else "plan"
            plan_filename = f"{name_prefix}_{timestamp}.md"
            plan_path = agent_dir / plan_filename
            plan_path.write_text(plan.strip() + "\n", encoding="utf-8")
            
            # Handle todos if provided - append to shared todos.yaml
            todos_section = ""
            if todos:
                todos_path = instance._get_todos_path()
                existing_todos = _load_todos(todos_path)
                merged_todos = _merge_todos(existing_todos, todos)
                _save_todos(todos_path, merged_todos)
                summary = _format_status_summary(merged_todos)
                # Indent summary for nesting under todos_summary
                indented_summary = summary.replace("summary:", "todos_summary:")
                todos_section = (
                    f"todos_updated: true\n"
                    f"todos_path: {instance.agent_uuid}/todos.yaml\n"
                    f"{indented_summary}"
                )
            else:
                todos_section = "todos_updated: false"
            
            # Build response
            rel_plan_path = f"{instance.agent_uuid}/{plan_filename}"
            return (
                f"status: ok\n"
                f"plan_path: {rel_plan_path}\n"
                f"{todos_section}"
            )
        
        # Attach instance for agent UUID injection by AnthropicAgent
        create_plan.__tool_instance__ = instance  # type: ignore[attr-defined]
        return create_plan
