"""TodoWriteTool and CheckTodoTool — async, sandbox-based task tracking.

Migrated from anthropic_agent/common_tools/todo_tool.py.
YAML persistence goes through sandbox read_file/write_file.
Agent UUID scoping removed — sandbox provides isolation.
"""
from __future__ import annotations

import json
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

        missing = REQUIRED_TODO_KEYS - set(todo.keys())
        if missing:
            return (
                f"Error: Todo at index {i} is missing required fields: "
                f"{', '.join(sorted(missing))}. "
                f"Each todo must have: id (string), content (string), "
                f"status (string), activeForm (string)."
            )

        if todo["status"] not in VALID_STATUSES:
            return (
                f"Error: Invalid status '{todo['status']}' for todo "
                f"'{todo.get('id', f'index {i}')}'. "
                f"Must be one of: {', '.join(sorted(VALID_STATUSES))}."
            )

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
    """Configurable todo_write tool with sandbox-based persistence.

    Todos are stored as a YAML file at the sandbox root (todos.yaml).
    The sandbox provides per-agent isolation.

    Example:
        >>> tool = TodoWriteTool()
        >>> func = tool.get_tool()
        >>> registry.register_tools([func])
        >>> registry.attach_sandbox(sandbox)
    """

    DOCSTRING_TEMPLATE = """Create or update the todo list by providing the complete list of todos.

This is a stateless replacement: every call must include ALL todos (not just changes).
The previous list is fully replaced by the new one.

**Rules:**
- Exactly one todo should be in_progress at a time.
- Each todo must have a unique string ID for later reference.
- Content should be imperative ("Fix the bug"), activeForm present-continuous ("Fixing the bug").
- Valid statuses: pending, in_progress, completed.

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
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )

    def get_tool(self) -> Callable:
        instance = self

        async def todo_write(todos: str) -> str:
            """Placeholder docstring - replaced by template."""
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

            # Write via sandbox
            try:
                content = yaml.dump(
                    {"todos": clean_todos},
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
                await instance._sandbox.write_file(TODO_FILENAME, content)
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

        func = self._apply_schema(todo_write)
        func.__tool_instance__ = instance
        return func


# ---------------------------------------------------------------------------
# CheckTodoTool
# ---------------------------------------------------------------------------
class CheckTodoTool(ConfigurableToolBase):
    """Configurable check_todo tool for reading the agent's current todo list.

    Example:
        >>> tool = CheckTodoTool()
        >>> func = tool.get_tool()
        >>> registry.register_tools([func])
        >>> registry.attach_sandbox(sandbox)
    """

    DOCSTRING_TEMPLATE = """Retrieve the current todo list with all items and their statuses.

Use this tool to check progress on tasks, see what is pending, in progress, or completed.

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
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )

    def get_tool(self) -> Callable:
        instance = self

        async def check_todo() -> str:
            """Placeholder docstring - replaced by template."""
            # Read via sandbox
            try:
                exists = await instance._sandbox.file_exists(TODO_FILENAME)
                if not exists:
                    return _format_todos_for_display([])

                content = await instance._sandbox.read_file(TODO_FILENAME)
                data = yaml.safe_load(content)
            except Exception as e:
                return f"Error: Failed to read todos: {e}"

            if not data or "todos" not in data:
                return _format_todos_for_display([])

            return _format_todos_for_display(data["todos"])

        func = self._apply_schema(check_todo)
        func.__tool_instance__ = instance
        return func
