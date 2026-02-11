"""TodoWrite and CheckTodo tools for agent task tracking.

This module provides two ConfigurableToolBase-based tools:
- TodoWriteTool: Stateless replacement of the full todo list (write to YAML)
- CheckTodoTool: Read-only retrieval of the current todo list (read from YAML)

Todo state is persisted per agent as a YAML file at:
    {base_dir}/{agent_uuid}/todos.yaml
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import yaml

from ..tools.base import ConfigurableToolBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_STATUSES = {"pending", "in_progress", "completed"}
REQUIRED_TODO_KEYS = {"id", "content", "status", "activeForm"}
TODO_FILENAME = "todos.yaml"


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------
def _get_todo_file_path(base_dir: Path, agent_uuid: str) -> Path:
    """Resolve the YAML file path for an agent's todos."""
    return base_dir / agent_uuid / TODO_FILENAME


def _read_todos(base_dir: Path, agent_uuid: str) -> list[dict[str, str]]:
    """Read todos from the YAML file. Returns empty list if file missing."""
    path = _get_todo_file_path(base_dir, agent_uuid)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "todos" not in data:
        return []
    return data["todos"]


def _write_todos(
    base_dir: Path, agent_uuid: str, todos: list[dict[str, str]]
) -> Path:
    """Write todos to YAML file, creating directories as needed. Returns path."""
    path = _get_todo_file_path(base_dir, agent_uuid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            {"todos": todos},
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return path


def _validate_todos(todos: list) -> str | None:
    """Validate a list of todo dicts.

    Returns an error string with guidance if invalid, or None if valid.
    """
    if not isinstance(todos, list):
        return "Error: 'todos' must be a JSON array of todo objects."

    seen_ids: dict[str, int] = {}
    for i, todo in enumerate(todos):
        if not isinstance(todo, dict):
            return (
                f"Error: Todo at index {i} must be an object, "
                f"got {type(todo).__name__}."
            )

        # Check required keys
        missing = REQUIRED_TODO_KEYS - set(todo.keys())
        if missing:
            return (
                f"Error: Todo at index {i} is missing required fields: "
                f"{', '.join(sorted(missing))}. "
                f"Each todo must have: id (string), content (string), "
                f"status (string), activeForm (string)."
            )

        # Validate status
        if todo["status"] not in VALID_STATUSES:
            return (
                f"Error: Invalid status '{todo['status']}' for todo "
                f"'{todo.get('id', f'index {i}')}'. "
                f"Must be one of: {', '.join(sorted(VALID_STATUSES))}."
            )

        # Check duplicate IDs
        tid = todo["id"]
        if tid in seen_ids:
            first_idx = seen_ids[tid]
            return (
                f"Error: Duplicate todo ID '{tid}' found at indices {first_idx} "
                f"and {i}. Each todo must have a unique ID. Please rename one "
                f"of them (e.g., '{tid}-2') and resubmit the full list."
            )
        seen_ids[tid] = i

    return None


def _format_todos_for_display(todos: list[dict[str, str]]) -> str:
    """Format a list of todos for human-readable output."""
    if not todos:
        return "No todos found for this agent."

    status_icons = {
        "pending": "[ ]",
        "in_progress": "[~]",
        "completed": "[x]",
    }
    lines = []
    for todo in todos:
        icon = status_icons.get(todo["status"], "[?]")
        lines.append(
            f"{icon} {todo['id']}: {todo['content']} "
            f"(status: {todo['status']}, activeForm: {todo['activeForm']})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TodoWriteTool
# ---------------------------------------------------------------------------
class TodoWriteTool(ConfigurableToolBase):
    """Configurable todo_write tool for stateless replacement of the agent's todo list.

    Each call sends the complete updated list of todos, replacing the previous state.
    Todos are persisted as YAML files scoped to the agent UUID.

    Example::

        tool = TodoWriteTool(base_dir="/data/todos")
        tool.set_agent_uuid("my-agent-uuid")
        write_fn = tool.get_tool()
        agent = AnthropicAgent(tools=[write_fn])
    """

    DOCSTRING_TEMPLATE = """Create or update the todo list by providing the complete list of todos.

This is a stateless replacement: every call must include ALL todos (not just changes).
The previous list is fully replaced by the new one.

**Rules:**
- Exactly one todo should be in_progress at a time.
- Each todo must have a unique string ID for later reference.
- Content should be imperative ("Fix the bug"), activeForm present-continuous ("Fixing the bug").
- Valid statuses: pending, in_progress, completed.

**Storage:** Todos are saved to {storage_path_pattern}

Args:
    todos: A JSON string representing the complete list of todos. Each todo is an object with:
        - id (string): Short unique identifier for the todo (e.g., "fix-auth", "add-tests")
        - content (string): Task description in imperative form (e.g., "Fix the authentication bug")
        - status (string): One of "pending", "in_progress", or "completed"
        - activeForm (string): Present-continuous description shown during execution (e.g., "Fixing the authentication bug")

        Example JSON:
        [
          {{"id": "explore", "content": "Explore the codebase", "status": "completed", "activeForm": "Exploring the codebase"}},
          {{"id": "implement", "content": "Implement feature", "status": "in_progress", "activeForm": "Implementing feature"}},
          {{"id": "test", "content": "Write tests", "status": "pending", "activeForm": "Writing tests"}}
        ]

Returns:
    Confirmation message with the number of todos saved and status counts,
    or an error message with guidance if validation fails.
"""

    def __init__(
        self,
        base_dir: str | Path,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the TodoWriteTool.

        Args:
            base_dir: Root directory for todo YAML files. Each agent gets
                a subdirectory: {base_dir}/{agent_uuid}/todos.yaml
            docstring_template: Optional custom docstring template with
                {placeholder} syntax. Available: {storage_path_pattern}.
            schema_override: Optional complete Anthropic tool schema dict.
        """
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )
        self.base_dir = Path(base_dir)
        self.agent_uuid: str | None = None

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "storage_path_pattern": f"{self.base_dir}/<agent_uuid>/{TODO_FILENAME}",
        }

    def set_agent_uuid(self, agent_uuid: str) -> None:
        """Set the agent UUID for scoping the todo file.

        Called by AnthropicAgent._inject_agent_uuid_to_tools() via the
        ``__tool_instance__`` / ``set_agent_uuid`` duck-typed protocol.
        """
        self.agent_uuid = agent_uuid

    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.

        Returns:
            A decorated todo_write function. The function has a
            ``__tool_instance__`` attribute for agent UUID injection.
        """
        instance = self

        def todo_write(todos: str) -> str:
            """Placeholder docstring - replaced by template."""
            if instance.agent_uuid is None:
                return (
                    "Error: agent_uuid is not set. Cannot save todos "
                    "without an agent session."
                )

            # Parse JSON
            try:
                todo_list = json.loads(todos)
            except (json.JSONDecodeError, TypeError) as e:
                return (
                    f"Error: Failed to parse todos JSON: {e}. "
                    f"Please check JSON syntax — todos must be a JSON array "
                    f"of objects."
                )

            # Validate
            error = _validate_todos(todo_list)
            if error:
                return error

            # Strip to required keys only
            clean_todos = [
                {k: todo[k] for k in ("id", "content", "status", "activeForm")}
                for todo in todo_list
            ]

            # Write
            try:
                _write_todos(instance.base_dir, instance.agent_uuid, clean_todos)
            except Exception as e:
                return f"Error: Failed to write todos: {e}"

            # Build summary
            counts = {"pending": 0, "in_progress": 0, "completed": 0}
            for t in clean_todos:
                counts[t["status"]] += 1

            return (
                f"Saved {len(clean_todos)} todo(s). "
                f"({counts['completed']} completed, "
                f"{counts['in_progress']} in progress, "
                f"{counts['pending']} pending)"
            )

        todo_write.__tool_instance__ = instance
        return self._apply_schema(todo_write)


# ---------------------------------------------------------------------------
# CheckTodoTool
# ---------------------------------------------------------------------------
class CheckTodoTool(ConfigurableToolBase):
    """Configurable check_todo tool for reading the agent's current todo list.

    Example::

        tool = CheckTodoTool(base_dir="/data/todos")
        tool.set_agent_uuid("my-agent-uuid")
        check_fn = tool.get_tool()
        agent = AnthropicAgent(tools=[check_fn])
    """

    DOCSTRING_TEMPLATE = """Retrieve the current todo list with all items and their statuses.

Use this tool to check progress on tasks, see what is pending, in progress, or completed.

**Storage:** Reads from {storage_path_pattern}

Returns:
    The complete todo list with status indicators:
    - [ ] = pending
    - [~] = in_progress
    - [x] = completed

    Each line shows: [status] id: content (status: value, activeForm: value)
    Returns "No todos found for this agent." if no todos exist yet.
"""

    def __init__(
        self,
        base_dir: str | Path,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the CheckTodoTool.

        Args:
            base_dir: Root directory for todo YAML files. Must match the
                base_dir used by the corresponding TodoWriteTool.
            docstring_template: Optional custom docstring template with
                {placeholder} syntax. Available: {storage_path_pattern}.
            schema_override: Optional complete Anthropic tool schema dict.
        """
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )
        self.base_dir = Path(base_dir)
        self.agent_uuid: str | None = None

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "storage_path_pattern": f"{self.base_dir}/<agent_uuid>/{TODO_FILENAME}",
        }

    def set_agent_uuid(self, agent_uuid: str) -> None:
        """Set the agent UUID for scoping the todo file.

        Called by AnthropicAgent._inject_agent_uuid_to_tools() via the
        ``__tool_instance__`` / ``set_agent_uuid`` duck-typed protocol.
        """
        self.agent_uuid = agent_uuid

    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.

        Returns:
            A decorated check_todo function. The function has a
            ``__tool_instance__`` attribute for agent UUID injection.
        """
        instance = self

        def check_todo() -> str:
            """Placeholder docstring - replaced by template."""
            if instance.agent_uuid is None:
                return (
                    "Error: agent_uuid is not set. Cannot read todos "
                    "without an agent session."
                )

            try:
                todos = _read_todos(instance.base_dir, instance.agent_uuid)
            except Exception as e:
                return f"Error: Failed to read todos: {e}"

            return _format_todos_for_display(todos)

        check_todo.__tool_instance__ = instance
        return self._apply_schema(check_todo)
