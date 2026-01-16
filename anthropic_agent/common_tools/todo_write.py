"""
### Spec for `todo_write` in `common_tools/todo_write.py`

- **Signature**
  - `def todo_write(todos: List[TodoItem], merge: bool = False) -> str`

- **Purpose**
  - Write todos to the agent's single todo file with merge semantics.

- **Directory structure**
  - Single file per agent: `{base_path}/{agent_uuid}/todos.yaml`

- **YAML structure**
  - Simple list of todos, each with id, content, status fields

- **Merge behavior**
  - `merge=False`: Replace entire list with provided todos.
  - `merge=True`: Update existing todos by id; preserve others.

- **Returns**
  - YAML-like structured response with status, operation, path, and summary breakdown.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Literal, Optional, TypedDict

from ..tools.decorators import tool


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]
VALID_STATUSES: set[str] = {"pending", "in_progress", "completed", "cancelled"}


class TodoItem(TypedDict, total=False):
    """Todo item structure.
    
    Attributes:
        id: Required unique identifier for the todo.
        content: Description of the todo task. Optional on merge updates.
        status: Current status. Defaults to 'pending' if not provided.
    """
    id: str
    content: str
    status: TodoStatus


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_todos(todos: List[TodoItem], require_content: bool = True) -> Optional[str]:
    """Validate a list of todos.
    
    Args:
        todos: List of todo items to validate.
        require_content: If True, content is required for each todo.
    
    Returns:
        Error message string if validation fails, None if valid.
    """
    if not todos:
        return "Error: No todos provided."
    
    for i, todo in enumerate(todos):
        # Check id is present and non-empty
        if "id" not in todo or not todo["id"]:
            return f"Error: Todo at index {i} is missing required 'id' field."
        
        # Check content if required
        if require_content and ("content" not in todo or not todo["content"]):
            return f"Error: Todo '{todo['id']}' is missing required 'content' field."
        
        # Validate status if provided
        if "status" in todo and todo["status"] not in VALID_STATUSES:
            return (
                f"Error: Todo '{todo['id']}' has invalid status '{todo['status']}'. "
                f"Valid statuses: {', '.join(sorted(VALID_STATUSES))}"
            )
    
    return None


# ---------------------------------------------------------------------------
# YAML helpers (no external dependencies)
# ---------------------------------------------------------------------------
def _format_todos_yaml(todos: List[TodoItem]) -> str:
    """Generate YAML content for todos file.
    
    Args:
        todos: List of todo items.
    
    Returns:
        YAML-formatted string.
    """
    if not todos:
        return "todos: []\n"
    
    lines = ["todos:"]
    for todo in todos:
        todo_id = todo.get("id", "")
        content = todo.get("content", "")
        status = todo.get("status", "pending")
        
        lines.append(f"  - id: {todo_id}")
        lines.append(f"    content: {content}")
        lines.append(f"    status: {status}")
    
    return "\n".join(lines) + "\n"


def _parse_todos_yaml(content: str) -> List[TodoItem]:
    """Parse a todos YAML file.
    
    Args:
        content: YAML file content.
    
    Returns:
        List of TodoItem dicts.
    """
    todos: List[TodoItem] = []
    lines = content.strip().split("\n")
    current_todo: Optional[TodoItem] = None
    
    for line in lines:
        stripped = line.strip()
        
        if stripped == "todos:" or stripped == "todos: []":
            continue
        elif stripped.startswith("- id:"):
            # New todo item
            if current_todo is not None:
                todos.append(current_todo)
            current_todo = {
                "id": stripped.split(":", 1)[1].strip(),
                "content": "",
                "status": "pending",
            }
        elif stripped.startswith("content:") and current_todo is not None:
            current_todo["content"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("status:") and current_todo is not None:
            status_value = stripped.split(":", 1)[1].strip()
            if status_value in VALID_STATUSES:
                current_todo["status"] = status_value  # type: ignore[typeddict-item]
    
    # Append last todo if any
    if current_todo is not None:
        todos.append(current_todo)
    
    return todos


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------
def _load_todos(path: Path) -> List[TodoItem]:
    """Load todos from a YAML file.
    
    Args:
        path: Path to the todos file.
    
    Returns:
        List of TodoItem dicts. Empty list if file doesn't exist.
    """
    if not path.exists():
        return []
    
    try:
        content = path.read_text(encoding="utf-8")
        return _parse_todos_yaml(content)
    except Exception:
        return []


def _save_todos(path: Path, todos: List[TodoItem]) -> None:
    """Save todos to a YAML file.
    
    Args:
        path: Path to the todos file.
        todos: List of todo items to save.
    """
    content = _format_todos_yaml(todos)
    path.write_text(content, encoding="utf-8")


def _merge_todos(existing: List[TodoItem], updates: List[TodoItem]) -> List[TodoItem]:
    """Merge update todos into existing todos by id.
    
    For each update:
    - If id exists: update status, and content if provided.
    - If id is new: add to the list.
    
    Args:
        existing: Current list of todos.
        updates: Todos to merge in.
    
    Returns:
        Merged list of todos.
    """
    # Build a dict keyed by id for fast lookup
    todos_by_id: dict[str, TodoItem] = {t["id"]: dict(t) for t in existing}  # type: ignore[misc]
    
    for update in updates:
        todo_id = update["id"]
        if todo_id in todos_by_id:
            # Update existing: always update status if provided
            if "status" in update:
                todos_by_id[todo_id]["status"] = update["status"]
            # Update content only if provided
            if "content" in update and update["content"]:
                todos_by_id[todo_id]["content"] = update["content"]
        else:
            # New todo: add it (must have content for new todos)
            todos_by_id[todo_id] = {
                "id": todo_id,
                "content": update.get("content", ""),
                "status": update.get("status", "pending"),
            }
    
    return list(todos_by_id.values())


def _count_statuses(todos: List[TodoItem]) -> dict[str, int]:
    """Count todos by status.
    
    Args:
        todos: List of todo items.
    
    Returns:
        Dict mapping status to count, with all statuses included (even if 0).
    """
    counts: dict[str, int] = {"pending": 0, "in_progress": 0, "completed": 0, "cancelled": 0}
    for todo in todos:
        status = todo.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _format_status_summary(todos: List[TodoItem]) -> str:
    """Format a YAML-like status summary for todos.
    
    Args:
        todos: List of todo items.
    
    Returns:
        YAML-formatted summary string.
    """
    counts = _count_statuses(todos)
    lines = [
        "summary:",
        f"  total: {len(todos)}",
        f"  pending: {counts['pending']}",
        f"  in_progress: {counts['in_progress']}",
        f"  completed: {counts['completed']}",
        f"  cancelled: {counts['cancelled']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Class-based tool implementation
# ---------------------------------------------------------------------------
class TodoWriteTool:
    """Configurable todo_write tool with a sandboxed base path and agent association.
    
    This class encapsulates the todo_write functionality, allowing configuration
    of the base path at instantiation time. The agent UUID can be provided at init
    or injected later via set_agent_uuid() (called automatically by AnthropicAgent).
    
    The tool returned by get_tool() can be registered with an AnthropicAgent.
    
    Example:
        >>> todo_tool = TodoWriteTool(base_path="/path/to/workspace")
        >>> agent = AnthropicAgent(tools=[todo_tool.get_tool()])
        >>> # agent_uuid is automatically injected by AnthropicAgent
    """
    
    def __init__(
        self,
        base_path: str | Path,
        agent_uuid: Optional[str] = None,
    ):
        """Initialize the TodoWriteTool with a base path and optional agent UUID.
        
        Args:
            base_path: The root directory for storing todos.
                       Todos will be stored at {base_path}/{agent_uuid}/todos.yaml
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
            A decorated todo_write function that operates within the configured base_path.
            The function has __tool_instance__ attribute for agent UUID injection.
        """
        instance = self
        
        @tool
        def todo_write(
            todos: List[TodoItem],
            merge: bool = False,
        ) -> str:
            """Write todos to the agent's single todo file.
            
            Manages a single todo list per agent at {base_path}/{agent_uuid}/todos.yaml.
            
            Args:
                todos: List of todo items. Each must have 'id' (required).
                       'content' and 'status' are optional on merge updates.
                       Valid statuses: pending, in_progress, completed, cancelled.
                merge: If False (default), replace entire list with provided todos.
                       If True, update existing todos by id; preserve others.
            
            Returns:
                Confirmation message with the todo list state.
            """
            # Validate todos
            # For merge=True, content is optional (status-only updates allowed)
            # For merge=False (replace), content is required for all todos
            validation_error = _validate_todos(todos, require_content=not merge)
            if validation_error:
                return validation_error
            
            # Ensure directory exists
            instance._ensure_agent_dir()
            todos_path = instance._get_todos_path()
            
            if merge:
                # Load existing and merge
                existing_todos = _load_todos(todos_path)
                merged_todos = _merge_todos(existing_todos, todos)
                _save_todos(todos_path, merged_todos)
                summary = _format_status_summary(merged_todos)
                return (
                    f"status: ok\n"
                    f"operation: updated\n"
                    f"path: {instance.agent_uuid}/todos.yaml\n"
                    f"{summary}"
                )
            else:
                # Replace entire list
                _save_todos(todos_path, todos)
                summary = _format_status_summary(todos)
                return (
                    f"status: ok\n"
                    f"operation: replaced\n"
                    f"path: {instance.agent_uuid}/todos.yaml\n"
                    f"{summary}"
                )
        
        # Attach instance for agent UUID injection by AnthropicAgent
        todo_write.__tool_instance__ = instance  # type: ignore[attr-defined]
        return todo_write
